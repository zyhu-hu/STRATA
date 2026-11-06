#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Build hourly ne128pg2 initial-condition stores from ne1024pg2 SCREAM output.

The hourly ne128 checkpoints (e.g. the large-scale emulator) were fine-tuned
on SCREAM coarsened from ne1024pg2 to ne128pg2. This script applies the same
coarsening to the requested hours and writes a store that
``scripts/ace/run_screamcast_ne128.py --ic-zarr`` reads:

- each 3-D field at the 24 model levels, physical SCREAM levels
  ``35:128:4`` (0-based, of 128);
- per cube face, the 2048x2048 fine raster is averaged over 8x8 blocks
  (``torch.nn.functional.avg_pool2d``, float32, unweighted). This is the
  operation that produced the training data; do not area-weight;
- the snapshot at the requested hour (no time averaging);
- statics ``phis``, ``sgh30``, ``landfrac`` coarsened the same way;
- ``lat``/``lon`` from ``latlon_ne128pg2.nc`` and ``hyam``/``hybm`` from
  ``data/scream_vertical_coordinate.nc`` at the same 24 levels.

Two kinds of source are accepted:

``--scream-netcdf-dir DIR``
    The SCREAM (EAMxx) 10-minute ``INSTANT`` output files, e.g.
    ``output.scream.U.10min.INSTANT.nmins_x10.1994-10-13-00000.nc``. Each
    file holds one or more variables as ``(time, ncol)`` or
    ``(time, ncol, lev)`` with ``ncol`` in the native ne1024pg2 order. The
    file for each variable and hour is found by name (time stamp suffix) and
    the record by the file's ``time`` variable. Statics come from the E3SM
    topography file on the same grid (``--topo-file``, variables ``PHIS``,
    ``SGH30``, ``LANDFRAC``).
``--scream-zarr PATH``
    An ne1024pg2 cubesphere zarr made from that output with levels
    ``3:128:4`` (32 levels) and one frame every 10 minutes; ``--zarr-t0`` is
    the time of frame 0. Statics come from the store.

Usage (one initial condition)::

    PYTHONPATH=. python3 scripts/ace/build_scream_ne128_ic.py \\
        --scream-netcdf-dir /path/to/run \\
        --topo-file /path/to/USGS-gtopo30_ne1024np4pg2_x6t-SGH.c20210614.nc \\
        --latlon $AUX_DATA_ROOT/latlon_ne128pg2.nc \\
        --start 1994-10-13T00:00:00 \\
        scream_ne128_1994101300.zarr

``--end`` adds the following hours (e.g. to score a rollout against the
coarsened truth). ``--reference STORE`` compares the result with another
store at the same hours, e.g. a reference initial condition built from the
training data, and prints the largest differences.

Memory: one cube face of one 3-D variable at a time, about 2 GB.
"""

import argparse
import datetime
import glob
import os
import re
import shlex
import sys

import numpy as np
import torch
import torch.nn.functional as F
import zarr

from screamcast.cubesphere_transforms import (
    reorder_2d_tensor_to_cubesphere,
    reorder_cubesphere_to_2d_tensor,
)

VARS_3D = ("PotentialTemperature", "U", "V", "z_mid", "omega", "qv")
VARS_2D = ("T_2m", "ps")
STATICS = ("phis", "sgh30", "landfrac")
NPG = 2
FACTOR = 8
COARSE_NE = 128
# Model levels: physical SCREAM levels 35, 39, ..., 127 (0-based, of 128).
MODEL_LEVELS = slice(35, 128, 4)
N_LEVELS = len(range(*MODEL_LEVELS.indices(128)))
SCREAM_DT = np.timedelta64(10, "m")
_EPOCH = np.datetime64("1970-01-01T00:00:00")


def coarsen_face(x: np.ndarray, fine_ne: int) -> np.ndarray:
    """``[C, fine face cells]`` -> ``[C, coarse face cells]`` by 8x8 block mean.

    Cells are in cubesphere storage order ``(elem_i, elem_j, gp_i, gp_j)``;
    the mean is taken on the face raster, as for the training data.
    """
    r = reorder_cubesphere_to_2d_tensor(torch.from_numpy(x), ne=fine_ne, npg=NPG)
    pooled = F.avg_pool2d(r.unsqueeze(0).float(), kernel_size=FACTOR, stride=FACTOR)
    back = reorder_2d_tensor_to_cubesphere(
        pooled.squeeze(0), ne=fine_ne // FACTOR, npg=NPG
    )
    return back.numpy()


def _fine_ne(ncol: int) -> int:
    side = int(round((ncol / 6) ** 0.5))
    if 6 * side * side != ncol or side % (NPG * FACTOR):
        raise ValueError(f"ncol={ncol} is not a cubed sphere coarsenable by {FACTOR}")
    return side // NPG


class ScreamZarrSource:
    """ne1024pg2 cubesphere zarr: ``[t, 32, ncol]`` at levels ``3:128:4``."""

    # store rows 8..31 hold physical levels 35..127
    STORE_LEVELS = slice(8, 32)

    def __init__(self, path: str, t0):
        self._g = zarr.open_group(path, mode="r", use_consolidated=False)
        self._t0 = np.datetime64(t0, "s")
        nlev = self._g[VARS_3D[0]].shape[1]
        if nlev != 32:
            raise ValueError(f"expected the 32-level 3:128:4 store, got {nlev} levels")
        self.ncol = self._g[VARS_3D[0]].shape[-1]

    def _frame(self, time) -> int:
        k, rem = divmod(np.datetime64(time, "s") - self._t0, SCREAM_DT)
        nt = self._g[VARS_3D[0]].shape[0]
        if rem or not 0 <= k < nt:
            raise ValueError(
                f"{time} is not a 10-minute frame of the store "
                f"(frame 0 = {self._t0}, {nt} frames)"
            )
        return int(k)

    def read(self, var: str, time, cols: slice) -> np.ndarray:
        """``[24, len(cols)]`` (3-D) or ``[1, len(cols)]`` (2-D), float32."""
        t = self._frame(time)
        if var in VARS_3D:
            x = self._g[var][t, self.STORE_LEVELS, cols]
        else:
            x = self._g[var][t, cols][None]
        return np.asarray(x, dtype=np.float32)

    def read_static(self, var: str, cols: slice) -> np.ndarray:
        return np.asarray(self._g[var][cols], dtype=np.float32)[None]


class ScreamNetcdfSource:
    """SCREAM (EAMxx) 10-minute INSTANT netCDF output plus the topography file."""

    _STAMP = re.compile(r"\.(\d{4})-(\d{2})-(\d{2})-(\d{5})\.nc$")
    _TOPO_NAMES = {"phis": "PHIS", "sgh30": "SGH30", "landfrac": "LANDFRAC"}

    def __init__(self, directory: str, topo_file: str):
        import netCDF4

        self._nc = netCDF4
        # stream (file name without the time stamp) -> sorted [(start, path)]
        self._streams = {}
        for path in glob.glob(os.path.join(directory, "*.nc")):
            m = self._STAMP.search(os.path.basename(path))
            if m is None:
                continue
            y, mo, d, sec = (int(g) for g in m.groups())
            start = np.datetime64(
                f"{y:04d}-{mo:02d}-{d:02d}T00:00:00"
            ) + np.timedelta64(sec, "s")
            stream = os.path.basename(path)[: m.start()]
            self._streams.setdefault(stream, []).append((start, path))
        if not self._streams:
            raise ValueError(f"no time-stamped *.nc files in {directory}")
        for files in self._streams.values():
            files.sort()
        self._open = {}
        self._topo = netCDF4.Dataset(topo_file)
        self._topo.set_auto_mask(False)
        self.ncol = self._topo.dimensions["ncol"].size

    def _dataset(self, path):
        if path not in self._open:
            ds = self._nc.Dataset(path)
            ds.set_auto_mask(False)
            self._open[path] = ds
        return self._open[path]

    @staticmethod
    def _check_fill(v, x):
        if "_FillValue" in v.ncattrs() and np.any(x == v.getncattr("_FillValue")):
            raise ValueError(f"{v.name}: fill values in the requested record")
        return x

    @staticmethod
    def _record(ds, time):
        """Index of the record of ``ds`` at ``time`` (to the minute), or None."""
        import cftime

        tv = ds.variables["time"]
        cal = tv.calendar if "calendar" in tv.ncattrs() else "standard"
        t = datetime.datetime.fromisoformat(str(np.datetime64(time, "s")))

        def num(dt):
            c = cftime.datetime(*dt.timetuple()[:6], calendar=cal)
            return float(cftime.date2num(c, tv.units, calendar=cal))

        target = num(t)
        half_minute = num(t + datetime.timedelta(seconds=30)) - target
        values = np.asarray(tv[:], dtype=np.float64)
        hit = np.flatnonzero(np.abs(values - target) < half_minute)
        return int(hit[0]) if hit.size else None

    def _locate(self, var: str, time):
        """``(netCDF variable, record)`` holding ``var`` at ``time``."""
        t = np.datetime64(time, "s")
        tried = []
        for files in self._streams.values():
            # the latest file of this stream that starts at or before t
            starts = [s for s, _ in files]
            i = np.searchsorted(np.array(starts), t, side="right") - 1
            if i < 0:
                continue
            ds = self._dataset(files[i][1])
            if var not in ds.variables:
                continue
            rec = self._record(ds, t)
            tried.append(files[i][1])
            if rec is not None:
                return ds.variables[var], rec
        raise ValueError(f"no record of {var} at {t} (searched {tried or 'no file'})")

    def read(self, var: str, time, cols: slice) -> np.ndarray:
        v, rec = self._locate(var, time)
        if var in VARS_3D:
            if v.dimensions != ("time", "ncol", "lev") or v.shape[2] != 128:
                raise ValueError(
                    f"{var}: expected (time, ncol, lev=128), got {v.dimensions} {v.shape}"
                )
            x = np.asarray(v[rec, cols, :], dtype=np.float32)[:, MODEL_LEVELS]
            return np.ascontiguousarray(self._check_fill(v, x).T)
        if v.dimensions != ("time", "ncol"):
            raise ValueError(f"{var}: expected (time, ncol), got {v.dimensions}")
        return self._check_fill(v, np.asarray(v[rec, cols], dtype=np.float32))[None]

    def read_static(self, var: str, cols: slice) -> np.ndarray:
        v = self._topo.variables[self._TOPO_NAMES[var]]
        return self._check_fill(v, np.asarray(v[cols], dtype=np.float32))[None]


def build_store(source, times, out: str, latlon, hyab, coarse_ne: int = COARSE_NE):
    """Coarsen ``source`` at ``times`` into a new hourly store at ``out``.

    ``latlon`` = (lat, lon) of the coarse cells in degrees, ``hyab`` = (hyam,
    hybm) at the model levels.
    """
    fine_ne = _fine_ne(source.ncol)
    if fine_ne != coarse_ne * FACTOR:
        raise ValueError(f"source is ne{fine_ne}, expected ne{coarse_ne * FACTOR}")
    fine_face = (fine_ne * NPG) ** 2
    coarse_face = (coarse_ne * NPG) ** 2
    ncol = 6 * coarse_face
    nt = len(times)
    hours = [
        int((np.datetime64(t, "s") - _EPOCH) // np.timedelta64(1, "h")) for t in times
    ]

    g = zarr.open_group(out, mode="w-")
    g.attrs.update(
        {
            "cubesphere_ne": coarse_ne,
            "cubesphere_npg": NPG,
            "time_units": "hours since 1970-01-01 00:00:00",
            "source": f"SCREAM ne{fine_ne}pg{NPG}, {FACTOR}x{FACTOR} block mean, "
            f"physical levels {MODEL_LEVELS.start}:{MODEL_LEVELS.stop}:{MODEL_LEVELS.step}",
            "history": f"{datetime.datetime.now(datetime.timezone.utc).isoformat()}: "
            + shlex.join([sys.executable, *sys.argv]),
        }
    )
    g.create_array("time", shape=(nt,), chunks=(nt,), dtype="int64")
    g["time"][:] = np.asarray(hours, dtype=np.int64)
    for name, arr in zip(("lat", "lon"), latlon):
        g.create_array(name, shape=(ncol,), chunks=(ncol,), dtype="float32")
        g[name][:] = np.asarray(arr, dtype=np.float32)
    for name, arr in zip(("hyam", "hybm"), hyab):
        g.create_array(name, shape=(N_LEVELS,), chunks=(N_LEVELS,), dtype="float64")
        g[name][:] = np.asarray(arr, dtype=np.float64)
    for v in VARS_3D:
        g.create_array(
            v,
            shape=(nt, N_LEVELS, ncol),
            chunks=(1, N_LEVELS, coarse_face),
            dtype="float32",
        )
    for v in VARS_2D:
        g.create_array(v, shape=(nt, ncol), chunks=(1, coarse_face), dtype="float32")

    def faces(read):
        out = []
        for f in range(6):
            x = read(slice(f * fine_face, (f + 1) * fine_face))
            out.append(coarsen_face(x, fine_ne))
        return np.concatenate(out, axis=-1)

    for v in STATICS:
        g.create_array(v, shape=(ncol,), chunks=(coarse_face,), dtype="float32")
        g[v][:] = faces(lambda cols, v=v: source.read_static(v, cols))[0]
    for k, t in enumerate(times):
        for v in VARS_3D + VARS_2D:
            x = faces(lambda cols, v=v, t=t: source.read(v, t, cols))
            if not np.isfinite(x).all():
                raise ValueError(f"non-finite values in {v} at {t}")
            g[v][k] = x if v in VARS_3D else x[0]
        print(f"[{k + 1}/{nt}] {t}", flush=True)
    zarr.consolidate_metadata(out)


def compare_stores(out: str, reference: str) -> float:
    """Print max |out - reference| per variable at the hours of ``out``;
    return the largest one relative to the reference's standard deviation."""
    a = zarr.open_group(out, mode="r")
    b = zarr.open_group(reference, mode="r")
    tb = list(np.asarray(b["time"][:]))
    rows = [tb.index(h) for h in np.asarray(a["time"][:])]
    worst = 0.0
    for v in VARS_3D + VARS_2D + STATICS:
        x = np.asarray(a[v][:], dtype=np.float64)
        y = np.asarray(b[v][rows] if v not in STATICS else b[v][:], dtype=np.float64)
        d = float(np.abs(x - y).max())
        rel = d / (float(y.std()) or 1.0)
        worst = max(worst, rel)
        print(
            f"  {v:22s} max|diff| {d:.3e}  /std {rel:.2e}  bitwise {np.array_equal(x, y)}"
        )
    return worst


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--scream-netcdf-dir", help="SCREAM 10-min INSTANT output files")
    src.add_argument("--scream-zarr", help="ne1024pg2 cubesphere zarr (levels 3:128:4)")
    p.add_argument("--topo-file", help="E3SM topography on ne1024pg2 (netCDF source)")
    p.add_argument("--zarr-t0", help="time of frame 0 of --scream-zarr")
    p.add_argument(
        "--start", required=True, help="first hour, e.g. 1994-10-13T00:00:00"
    )
    p.add_argument("--end", help="last hour, inclusive (default: --start)")
    p.add_argument(
        "--latlon",
        default=os.path.join(os.environ.get("AUX_DATA_ROOT", ""), "latlon_ne128pg2.nc"),
    )
    p.add_argument(
        "--vertical-coordinate",
        default=os.path.join(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            ),
            "data",
            "scream_vertical_coordinate.nc",
        ),
    )
    p.add_argument("--reference", help="store to compare the result with")
    p.add_argument("out")
    args = p.parse_args(argv)
    if args.scream_netcdf_dir and not args.topo_file:
        p.error("--scream-netcdf-dir needs --topo-file")
    if args.scream_zarr and not args.zarr_t0:
        p.error("--scream-zarr needs --zarr-t0")
    return args


def main(argv=None) -> None:
    import xarray as xr

    args = parse_args(argv)
    if os.path.exists(args.out):
        raise SystemExit(f"{args.out} exists; choose a new output path")
    start = np.datetime64(args.start, "s")
    end = np.datetime64(args.end or args.start, "s")
    if (start - _EPOCH) % np.timedelta64(1, "h") or end < start:
        raise SystemExit("--start must be on the hour and --end >= --start")
    times = list(np.arange(start, end + np.timedelta64(1, "h"), np.timedelta64(1, "h")))
    if args.scream_netcdf_dir:
        source = ScreamNetcdfSource(args.scream_netcdf_dir, args.topo_file)
    else:
        source = ScreamZarrSource(args.scream_zarr, args.zarr_t0)
    with xr.open_dataset(args.latlon) as ll:
        latlon = (ll["lat"].values, ll["lon"].values)
    with xr.open_dataset(args.vertical_coordinate) as vc:
        hyab = (vc["hyam"].values[MODEL_LEVELS], vc["hybm"].values[MODEL_LEVELS])
    build_store(source, times, args.out, latlon, hyab)
    print(f"wrote {args.out}")
    if args.reference:
        print(f"compared with {args.reference}:")
        compare_stores(args.out, args.reference)


if __name__ == "__main__":
    main()
