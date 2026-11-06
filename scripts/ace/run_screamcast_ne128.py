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
"""Hourly autoregressive rollout of an ne128pg2 checkpoint (e.g. the large-scale
emulator ``ckpt_scream_lora3ds_replay_409k_merged.pth``).

Uses the halo/topology/writer machinery of ``run_screamcast_nudged.py`` with
the grid, data source and step size of the hourly ne128 models:

- 6 faces of 256x256. With the default ``--tile-size 256 --halo-width 32``
  each face is one tile padded to 320, the geometry these models train on;
  the halo ring comes from the neighbouring faces through a KNN interpolator
  built from the padded SCRIP grid (``ne128halo16pg2_scrip.nc``). The model
  output is cropped back to the interior (no halo adjoint). ``total_tiles``
  is 6, so the world size must divide 6.
- The initial condition is read from hourly ne128pg2 zarr stores whose
  ``time`` coordinate is integer hours since 1970-01-01 and whose variables
  are stored per cell (3-D ``[time, level, ncol]``, 2-D ``[time, ncol]``,
  static ``[ncol]``). coszr is computed by the model at every step; the
  static fields and the lat/lon grid are taken from the first store.
- The model step is ``--dt-hours`` (1 h for these checkpoints; the
  checkpoint does not record it).
- Channel level suffixes index store levels directly (``--level-offset 0``);
  a checkpoint whose suffixes start at 8 needs ``--level-offset 8``.

Batch size: the two polar faces are centred on the poles, where the
spherical-centroid longitude of the tile frame (used for the wind rotation
and the stereographic RoPE) is set by rounding in the reduction, so it
depends on how many tiles share a forward pass. The hourly ne128
checkpoints trained with one tile per forward pass, so
``--inference-batch-size`` defaults to 1, which gives the polar faces the
frame they were trained with, for any number of ranks. Larger batches run
slightly faster but rotate the polar-face frames by up to about 12 degrees,
which degrades the polar-face forecasts.

``AUX_DATA_ROOT`` must hold ``latlon_ne128pg2.nc``, ``ne128halo16pg2_scrip.nc``
and the ne128 static fields ``phis.npy``, ``sgh30.npy``, ``landfrac.npy``
(flat cell order, 393216 values), which ``ScreamcastModel.from_checkpoint``
reads. Keep it separate from the ne1024 one: the .npy names are the same.

Example (one GPU)::

    AUX_DATA_ROOT=<dir with the ne128 grid files and static .npy> \\
    PYTHONPATH=. python3 scripts/ace/run_screamcast_ne128.py \\
        --run-name ls_rollout --checkpoint ckpt_scream_lora3ds_replay_409k_merged.pth \\
        --ic-zarr lsic_ne128_2016080100.zarr --initial-time 2016-08-01T00:00:00 \\
        --n-steps 24 out_ls.zarr

The output store records the timestamp and the exact command in its
``history`` attribute.
"""

import argparse
import datetime
import logging
import os
import shlex
import sys
from collections import OrderedDict

import numpy as np
import torch
import zarr

from screamcast.cubesphere_transforms import unstructured_to_6faces
from screamcast.dali_ext_src import open_consolidated_zarr_group
from screamcast.distributed_halo import (
    DistributedTileKNNHaloPadding_AllGather,
    TileTopology,
)
from screamcast.distributed_state_fetch import fetch_local_tiles
from screamcast.earth2studio_wrappers import ScreamcastModel, ScreamDataSource
from screamcast.zarr_writer import FaceZarrWriteStep, prepare_output_store
from scripts.ace.run_screamcast_nudged import (
    _split_variable_levels,
    channel_chunked,
    load_scrip_lonlat,
    run_scream,
)

logger = logging.getLogger("run_screamcast_ne128")

_EPOCH = np.datetime64("1970-01-01T00:00:00")


class HourlyCubeSphereSource(ScreamDataSource):
    """Initial-condition source over hourly ne128pg2 stores.

    Each store's ``time`` coordinate is integer hours since 1970-01-01. A
    request is served from the one store whose time range covers it.
    """

    NE = 128
    NPG = 2
    MODEL_DT_MINUTES = 60.0

    def __init__(self, zarr_paths, level_offset: int = 0, storage_env: str = ""):
        self._groups = []
        self._t0s = []
        self._nts = []
        for p in zarr_paths:
            g = open_consolidated_zarr_group(p, storage_env)
            self._groups.append(g)
            self._t0s.append(int(g["time"][0]))
            self._nts.append(int(g["time"].shape[0]))
        self._level_offset = int(level_offset)
        self._active_t0 = None
        # _var_group is bound to the covering store in __call__.
        super().__init__(var_group={})

    @property
    def first_group(self):
        """The first store (grid, statics and hybrid coefficients)."""
        return self._groups[0]

    def _parse_variable(self, var_str: str) -> tuple:
        """Map a channel's level suffix to a store level (suffix - offset)."""
        name, level = super()._parse_variable(var_str)
        if level is not None:
            level = level - self._level_offset
            if level < 0:
                raise ValueError(
                    f"channel {var_str!r} maps to store level {level} < 0 "
                    f"(level_offset={self._level_offset} too large?)"
                )
        return name, level

    @staticmethod
    def _hours_since_epoch(t) -> int:
        delta = np.datetime64(t, "s") - _EPOCH
        if delta % np.timedelta64(1, "h") != np.timedelta64(0, "s"):
            raise ValueError(f"time {t} is not hour-aligned; the stores are hourly")
        return int(delta // np.timedelta64(1, "h"))

    def _select_store(self, t) -> int:
        """Index of the store whose time range covers ``t``."""
        hours = self._hours_since_epoch(t)
        for i, (t0, nt) in enumerate(zip(self._t0s, self._nts)):
            if t0 <= hours < t0 + nt:
                return i
        raise ValueError(
            f"time {t} (hour {hours}) is not covered by the stores "
            f"(start hours {self._t0s}, lengths {self._nts})"
        )

    def __call__(self, coords: dict):
        """Bind the covering store, then read through ScreamDataSource."""
        stores = {self._select_store(t) for t in np.atleast_1d(coords["time"])}
        if len(stores) != 1:
            raise NotImplementedError("a request must not span several stores")
        i = stores.pop()
        g = self._groups[i]
        self._var_group = {k: g for k in g.array_keys()}
        self._active_t0 = self._t0s[i]
        return super().__call__(coords)

    def _datetime_to_tidx(self, t) -> int:
        if self._active_t0 is None:
            raise RuntimeError("_datetime_to_tidx called before a store was bound")
        return self._hours_since_epoch(t) - self._active_t0


def faces_from_store(flat: np.ndarray, ne: int, npg: int) -> np.ndarray:
    """Per-cell store order ``[..., 6*ne*ne*npg*npg]`` -> faces ``[..., 6, S, S]``.

    Within a face the store holds ``(elem_i, elem_j, gp_i, gp_j)``; face pixel
    ``(elem_i*npg + gp_i, elem_j*npg + gp_j)``.
    """
    side = ne * npg
    lead = flat.shape[:-1]
    x = flat.reshape(*lead, 6, ne, ne, npg, npg)
    x = np.moveaxis(x, -2, -3)  # (..., 6, ne, npg, ne, npg)
    return x.reshape(*lead, 6, side, side)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument(
        "--ic-zarr",
        action="append",
        required=True,
        help="hourly ne128pg2 store with the initial condition (repeat the "
        "flag for several stores; the first one supplies grid and statics)",
    )
    p.add_argument("--initial-time", required=True, help="e.g. 2016-08-01T00:00:00")
    p.add_argument("--n-steps", type=int, default=24)
    p.add_argument("--dt-hours", type=float, default=1.0)
    p.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    p.add_argument(
        "--wind-center-mode",
        choices=("mean", "spherical_centroid"),
        default=None,
        help="override DiTConfig.wind_center_mode for checkpoints whose "
        "embedded config predates the field (default: use the checkpoint's)",
    )
    p.add_argument("--level-offset", type=int, default=0)
    p.add_argument("--tile-size", type=int, default=256)
    p.add_argument("--halo-width", type=int, default=32)
    p.add_argument("--halo-knn-k", type=int, default=4)
    p.add_argument("--halo-channel-chunk", type=int, default=64)
    _aux = os.environ.get("AUX_DATA_ROOT", "data")
    p.add_argument(
        "--grid-tgt",
        default=os.path.join(_aux, "ne128halo16pg2_scrip.nc"),
        help="SCRIP file of the halo-padded ne128pg2 grid",
    )
    p.add_argument(
        "--grid-tgt-halo-elems",
        type=int,
        default=16,
        help="halo width in ELEMENTS that --grid-tgt was generated with",
    )
    p.add_argument("--storage-env", default="", help="zarr storage profile (s3)")
    p.add_argument(
        "--inference-batch-size",
        type=int,
        default=1,
        help="tiles per forward pass; 1 matches training (see the module docstring)",
    )
    p.add_argument("--fetch-channel-chunk", type=int, default=32)
    p.add_argument("--output-steps", type=int, default=1)
    p.add_argument("--output-chunk-size", type=int, default=64)
    p.add_argument("--output-variables", nargs="+", default=None, metavar="VAR")
    p.add_argument(
        "--output-levels",
        type=lambda s: [int(x) for x in s.split(",")],
        default=None,
        metavar="L0,L1,...",
        help="channel-suffix levels to write for 3-D variables (default: all)",
    )
    p.add_argument("output", help="output zarr path (must not exist)")
    return p.parse_args(argv)


def _select_output_variables(out_vars, output_variables, output_levels):
    grouped = _split_variable_levels(out_vars)
    if output_variables is not None:
        unknown = set(output_variables) - grouped.keys()
        if unknown:
            raise ValueError(
                f"Unknown output variables {sorted(unknown)}; "
                f"available: {sorted(grouped)}"
            )
        grouped = OrderedDict(
            (k, v) for k, v in grouped.items() if k in output_variables
        )
    if output_levels is not None:
        keep = set(output_levels)
        filtered = OrderedDict()
        for k, entries in grouped.items():
            if entries[0][0] is None:
                filtered[k] = entries
            else:
                kept = [(lvl, ci) for lvl, ci in entries if lvl in keep]
                if kept:
                    filtered[k] = kept
        grouped = filtered
    return grouped


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args(argv)
    if os.path.exists(args.output):
        raise FileExistsError(f"{args.output} exists; refusing to overwrite")
    t0 = np.datetime64(args.initial_time)
    if args.precision == "fp32":
        # Full fp32: no TF32 in matmuls or cuDNN convolutions.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    # The halo exchange and state fetch use collectives even at world size 1.
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(29500 + os.getpid() % 500))
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    torch.distributed.init_process_group(backend="nccl")
    rank = torch.distributed.get_rank()
    world = torch.distributed.get_world_size()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    device = torch.device("cuda", torch.cuda.current_device())

    ds = HourlyCubeSphereSource(
        args.ic_zarr, level_offset=args.level_offset, storage_env=args.storage_env
    )
    store0 = ds.first_group
    ne, npg = HourlyCubeSphereSource.NE, HourlyCubeSphereSource.NPG
    face_size = ne * npg

    model = ScreamcastModel.from_checkpoint(
        args.checkpoint,
        bf16=args.precision == "bf16",
        dit_overrides=(
            {"wind_center_mode": args.wind_center_mode}
            if args.wind_center_mode
            else None
        ),
    )
    model.inference_batch_size = args.inference_batch_size
    model = model.to(device).eval()
    model.dt = np.timedelta64(int(round(args.dt_hours * 3600)), "s")

    # Grid and static fields of the IC store replace the ones from_checkpoint
    # loaded; set_latlon below regrids them onto the padded tiles.
    lat_faces = faces_from_store(np.asarray(store0["lat"][:]), ne, npg)
    lon_faces = faces_from_store(np.asarray(store0["lon"][:]), ne, npg)
    model._src_latlon = torch.from_numpy(
        np.stack([lat_faces, lon_faces], axis=-1).astype(np.float32)
    ).to(device)
    if model._static_forcing_names:
        statics = np.stack(
            [
                faces_from_store(np.asarray(store0[v][:], dtype=np.float32), ne, npg)
                for v in model._static_forcing_names
            ]
        )
        model._src_static_forcing = torch.from_numpy(statics).to(device)

    topology = TileTopology(
        world_size=world,
        rank=rank,
        face_size=face_size,
        tile_size=args.tile_size,
        halo_width=args.halo_width,
    )
    pad_width_data = args.grid_tgt_halo_elems * npg
    if args.halo_width > pad_width_data:
        raise ValueError(
            f"--halo-width {args.halo_width} px exceeds the {pad_width_data} px "
            f"halo of {args.grid_tgt}"
        )
    grid_tgt = load_scrip_lonlat(args.grid_tgt)
    ne_padded = ne + 2 * args.grid_tgt_halo_elems
    lon = unstructured_to_6faces(torch.from_numpy(grid_tgt["lon"]), ne_padded, npg)
    lat = unstructured_to_6faces(torch.from_numpy(grid_tgt["lat"]), ne_padded, npg)
    interior = slice(pad_width_data, pad_width_data + face_size)
    lat_face = lat[:, interior, interior].numpy()
    lon_face = lon[:, interior, interior].numpy()
    dlon = (lon_face - lon_faces + 180.0) % 360.0 - 180.0
    if not np.allclose(lat_face, lat_faces, atol=1e-3) or np.abs(dlon).max() > 1e-3:
        raise ValueError(
            f"interior of {args.grid_tgt} does not match the IC store grid"
        )
    halo_exchange = DistributedTileKNNHaloPadding_AllGather.from_padded_face_grid(
        topology=topology,
        lon=lon,
        lat=lat,
        pad_width_data=pad_width_data,
        device=device,
        k=args.halo_knn_k,
    ).to(device)
    pad = channel_chunked(args.halo_channel_chunk)(halo_exchange)
    unpad = channel_chunked(args.halo_channel_chunk)(halo_exchange.topology.crop)
    model.set_latlon(lat_deg=halo_exchange.lat_deg, lon_deg=halo_exchange.lon_deg)

    in_coords = dict(model.input_coords())
    out_vars = list(model.output_coords(dict(in_coords))["variable"])
    grouped = _select_output_variables(
        out_vars, args.output_variables, args.output_levels
    )
    # The writer indexes hyam/hybm by channel suffix: with a level offset,
    # prefix NaNs so suffix k maps to store level k - offset.
    hyam = np.asarray(store0["hyam"][:], dtype=np.float32)
    hybm = np.asarray(store0["hybm"][:], dtype=np.float32)
    if args.level_offset:
        nan_pad = np.full(args.level_offset, np.nan, dtype=np.float32)
        hyam = np.concatenate([nan_pad, hyam])
        hybm = np.concatenate([nan_pad, hybm])

    x0, coords = fetch_local_tiles(
        ds, in_coords, topology, device, t0, args.fetch_channel_chunk
    )
    logger.info("IC %s: %d tiles x %d channels", t0, x0.shape[0], x0.shape[1])
    if rank == 0:
        prepare_output_store(
            output_path=args.output,
            grouped_variables=grouped,
            n_times=1,
            n_steps=len(range(0, args.n_steps, args.output_steps)),
            nside=face_size,
            tile_size=args.output_chunk_size,
            time_values=np.array([t0]),
            lat=lat_face,
            lon=lon_face,
            dt=model.dt * args.output_steps,
            hyam=hyam,
            hybm=hybm,
            attrs={
                "run_name": args.run_name,
                "checkpoint": args.checkpoint,
                "initial_time": str(t0),
                "grid": "ne128pg2",
                "dt_hours": args.dt_hours * args.output_steps,
                "precision": args.precision,
                "pixel_ordering": "cubesphere_faces_2d",
                "history": (
                    f"{datetime.datetime.now(datetime.timezone.utc).isoformat()}: "
                    + shlex.join([sys.executable, *sys.argv])
                ),
            },
        )
        zarr.consolidate_metadata(args.output)
    torch.distributed.barrier()
    writer = FaceZarrWriteStep(
        output_steps=args.output_steps,
        topology=topology,
        grouped_variables=grouped,
        store=zarr.open_group(args.output, mode="a"),
    )
    run_scream(x0, coords, args.n_steps, model, pad, unpad, topology, write_step=writer)
    writer.close()
    torch.distributed.barrier()
    if rank == 0:
        zarr.consolidate_metadata(args.output)
    logger.info("rollout complete: %s", args.output)
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
