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
"""Tests for scripts/ace/build_scream_ne128_ic.py on a small cubed sphere.

The fine grid is ne16pg2 (coarse ne2pg2). Fields are linear in the face
raster position, so the exact 8x8 block mean is the value at the block
centre; levels and time frames carry their own index in the value. All sums
stay below 2**24, so float32 block means are exact.
"""

import itertools

import numpy as np
import pytest
import zarr

netCDF4 = pytest.importorskip("netCDF4")

from scripts.ace.build_scream_ne128_ic import (  # noqa: E402
    STATICS,
    VARS_2D,
    VARS_3D,
    ScreamNetcdfSource,
    ScreamZarrSource,
    build_store,
    compare_stores,
)

FINE_NE, COARSE_NE, NPG = 16, 2, 2
FINE_NCOL = 6 * (FINE_NE * NPG) ** 2
COARSE_NCOL = 6 * (COARSE_NE * NPG) ** 2
T0 = np.datetime64("1994-10-13T00:00:00")
N_FRAMES = 13  # 10-min frames 00:00 .. 02:00


def _cells(ne):
    """(flat index, face, raster row, raster col) in cubesphere storage order."""
    for i, (f, ei, ej, gi, gj) in enumerate(
        itertools.product(range(6), range(ne), range(ne), range(NPG), range(NPG))
    ):
        yield i, f, ei * NPG + gi, ej * NPG + gj


def _spatial(ne, at_block_centre=False):
    """1000*face + 7*row + 3*col; at_block_centre evaluates it at the centre of
    each fine 8x8 block of a coarse cell (fine coordinates)."""
    out = np.empty(6 * (ne * NPG) ** 2, dtype=np.float64)
    for i, f, r, c in _cells(ne):
        if at_block_centre:
            r, c = 8 * r + 3.5, 8 * c + 3.5
        out[i] = 1000 * f + 7 * r + 3 * c
    return out


FINE = _spatial(FINE_NE).astype(np.float32)
EXPECTED = _spatial(COARSE_NE, at_block_centre=True)
VAR_OFFSET = {v: 20000 * (k + 1) for k, v in enumerate(VARS_3D + VARS_2D + STATICS)}


def _value(var, frame=None, level=None):
    """Fine field of ``var`` at a 10-min ``frame`` and physical ``level``."""
    x = FINE + VAR_OFFSET[var]
    if frame is not None:
        x = x + 1000 * frame
    if level is not None:
        x = x + 10 * level
    return x.astype(np.float32)


@pytest.fixture(scope="module")
def zarr_source(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("fine") / "fine.zarr")
    g = zarr.open_group(path, mode="w")
    for v in VARS_3D:
        a = g.create_array(v, shape=(N_FRAMES, 32, FINE_NCOL), dtype="float32")
        for k in range(N_FRAMES):
            # store row j = physical level 3 + 4j
            a[k] = np.stack([_value(v, k, 3 + 4 * j) for j in range(32)])
    for v in VARS_2D:
        a = g.create_array(v, shape=(N_FRAMES, FINE_NCOL), dtype="float32")
        for k in range(N_FRAMES):
            a[k] = _value(v, k)
    for v in STATICS:
        g.create_array(v, shape=(FINE_NCOL,), dtype="float32")[:] = _value(v)
    return ScreamZarrSource(path, str(T0))


def _write_stream(path, variables, start, n_records):
    """One EAMxx-style output file: 10-min records, noleap calendar."""
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("time", None)
        ds.createDimension("ncol", FINE_NCOL)
        ds.createDimension("lev", 128)
        tv = ds.createVariable("time", "f8", ("time",))
        tv.units = "days since 1994-10-01 00:00:00"
        tv.calendar = "noleap"
        frame0 = int((start - T0) // np.timedelta64(10, "m"))
        days0 = (start - np.datetime64("1994-10-01T00:00:00")) / np.timedelta64(1, "D")
        tv[:] = days0 + np.arange(n_records) / 144.0
        for v in variables:
            if v in VARS_3D:
                x = ds.createVariable(v, "f4", ("time", "ncol", "lev"))
                for r in range(n_records):
                    x[r] = np.stack(
                        [_value(v, frame0 + r, lev) for lev in range(128)], axis=-1
                    )
            else:
                x = ds.createVariable(v, "f4", ("time", "ncol"))
                for r in range(n_records):
                    x[r] = _value(v, frame0 + r)


@pytest.fixture(scope="module")
def netcdf_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("run")
    for h in range(2):  # hourly files of 6 records: 00:00-00:50, 01:00-01:50
        start = T0 + np.timedelta64(h, "h")
        stamp = f"1994-10-13-{3600 * h:05d}"
        for v in VARS_3D:
            _write_stream(
                d / f"output.scream.{v}.10min.INSTANT.nmins_x10.{stamp}.nc",
                [v],
                start,
                6,
            )
        _write_stream(
            d / f"output.scream.T02D.10min.INSTANT.nmins_x10.{stamp}.nc",
            ["ps"],
            start,
            6,
        )
        _write_stream(
            d / f"output.scream.T12D.10min.INSTANT.nmins_x10.{stamp}.nc",
            ["T_2m"],
            start,
            6,
        )
    with netCDF4.Dataset(d / "topo.nc", "w") as ds:
        ds.createDimension("ncol", FINE_NCOL)
        for v, name in ScreamNetcdfSource._TOPO_NAMES.items():
            ds.createVariable(name, "f8", ("ncol",))[:] = _value(v)
    return d


def _build(source, tmp_path, times, name="out.zarr"):
    out = str(tmp_path / name)
    latlon = (np.arange(COARSE_NCOL) * 0.25, np.arange(COARSE_NCOL) * 0.5)
    hyab = (np.arange(24) * 0.01, np.arange(24) * 0.02)
    build_store(source, times, out, latlon, hyab, coarse_ne=COARSE_NE)
    return out


def _check_store(out, hours):
    g = zarr.open_consolidated(out)
    ref_hours = (
        np.datetime64("1994-10-13T00", "h") - np.datetime64("1970-01-01T00", "h")
    ).astype(int)
    assert list(g["time"][:]) == [ref_hours + h for h in hours]
    for k, h in enumerate(hours):
        frame = 6 * h
        for v in VARS_3D:
            got = np.asarray(g[v][k], dtype=np.float64)
            for lev_idx, phys in enumerate(range(35, 128, 4)):
                want = EXPECTED + VAR_OFFSET[v] + 1000 * frame + 10 * phys
                np.testing.assert_allclose(got[lev_idx], want, rtol=0, atol=1e-2)
        for v in VARS_2D:
            want = EXPECTED + VAR_OFFSET[v] + 1000 * frame
            np.testing.assert_allclose(np.asarray(g[v][k]), want, rtol=0, atol=1e-2)
    for v in STATICS:
        np.testing.assert_allclose(
            np.asarray(g[v][:]), EXPECTED + VAR_OFFSET[v], rtol=0, atol=1e-2
        )
    assert g["U"].shape == (len(hours), 24, COARSE_NCOL)
    np.testing.assert_array_equal(g["hyam"][:], np.arange(24) * 0.01)
    assert "history" in g.attrs


def test_zarr_source_block_mean_levels_and_hours(zarr_source, tmp_path):
    out = _build(zarr_source, tmp_path, [T0, T0 + np.timedelta64(2, "h")])
    _check_store(out, [0, 2])


def test_netcdf_source_matches_zarr_source(zarr_source, netcdf_dir, tmp_path):
    src = ScreamNetcdfSource(str(netcdf_dir), str(netcdf_dir / "topo.nc"))
    times = [T0, T0 + np.timedelta64(1, "h")]
    a = _build(src, tmp_path, times, "nc.zarr")
    _check_store(a, [0, 1])
    b = _build(zarr_source, tmp_path, times, "zarr.zarr")
    assert compare_stores(a, b) == 0.0


def test_sources_reject_missing_times(zarr_source, netcdf_dir):
    with pytest.raises(ValueError, match="not a 10-minute frame"):
        zarr_source.read("ps", T0 + np.timedelta64(5, "m"), slice(0, 4))
    with pytest.raises(ValueError, match="not a 10-minute frame"):
        zarr_source.read("ps", T0 + np.timedelta64(3, "h"), slice(0, 4))
    src = ScreamNetcdfSource(str(netcdf_dir), str(netcdf_dir / "topo.nc"))
    with pytest.raises(ValueError, match="no record of ps"):
        src.read("ps", T0 + np.timedelta64(2, "h"), slice(0, 4))
    with pytest.raises(ValueError, match="no record of ps"):
        src.read("ps", T0 - np.timedelta64(1, "h"), slice(0, 4))


def test_netcdf_source_rejects_fill_values(tmp_path):
    _write_stream(
        tmp_path / "output.scream.T02D.10min.INSTANT.nmins_x10.1994-10-13-00000.nc",
        ["ps"],
        T0,
        1,
    )
    with netCDF4.Dataset(tmp_path / "topo.nc", "w") as ds:
        ds.createDimension("ncol", FINE_NCOL)
        x = ds.createVariable("PHIS", "f4", ("ncol",), fill_value=-999.0)
        x[:] = 1.0
        x[5] = -999.0
    src = ScreamNetcdfSource(str(tmp_path), str(tmp_path / "topo.nc"))
    with pytest.raises(ValueError, match="fill values"):
        src.read_static("phis", slice(0, 64))
