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
"""Initial-condition reads of scripts/ace/run_screamcast_ne128.py.

Stores are built from a known face field through the documented per-cell
layout (face, elem_i, elem_j, gp_i, gp_j) with explicit loops, so the
expected faces never go through the code under test.
"""

import numpy as np
import pytest
import zarr

from scripts.ace.run_screamcast_ne128 import (
    HourlyCubeSphereSource,
    faces_from_store,
    parse_args,
)

NE, NPG, NLEV = 3, 2, 4
SIDE = NE * NPG
NCOL = 6 * NE * NE * NPG * NPG


class _TinySource(HourlyCubeSphereSource):
    NE = NE


def _flat_from_faces(faces: np.ndarray) -> np.ndarray:
    """[..., 6, SIDE, SIDE] -> [..., NCOL] in store cell order."""
    flat = np.empty(faces.shape[:-3] + (NCOL,), dtype=faces.dtype)
    for f in range(6):
        for ei in range(NE):
            for ej in range(NE):
                for gi in range(NPG):
                    for gj in range(NPG):
                        col = (((f * NE + ei) * NE + ej) * NPG + gi) * NPG + gj
                        flat[..., col] = faces[..., f, ei * NPG + gi, ej * NPG + gj]
    return flat


def _faces_value(t_hour: int, level: int) -> np.ndarray:
    """Distinct value per (hour, level, face, x, y)."""
    f, x, y = np.meshgrid(np.arange(6), np.arange(SIDE), np.arange(SIDE), indexing="ij")
    return (t_hour * 1e4 + level * 1e3 + f * 100 + x * 10 + y).astype(np.float32)


def _make_store(path, start_hour: int, n_hours: int) -> str:
    g = zarr.open_group(str(path), mode="w")
    hours = np.arange(start_hour, start_hour + n_hours, dtype=np.int64)
    g.create_array("time", data=hours)
    u = np.stack([[_faces_value(h, k) for k in range(NLEV)] for h in hours])
    g.create_array("U", data=_flat_from_faces(u))
    ps = np.stack([_faces_value(h, 99) for h in hours])
    g.create_array("ps", data=_flat_from_faces(ps))
    g.create_array("phis", data=_flat_from_faces(_faces_value(0, 7)))
    zarr.consolidate_metadata(str(path))
    return str(path)


def _iso(hour: int) -> np.datetime64:
    return np.datetime64("1970-01-01T00:00:00") + np.timedelta64(hour, "h")


@pytest.fixture
def stores(tmp_path):
    return [
        _make_store(tmp_path / "a.zarr", 1000, 3),
        _make_store(tmp_path / "b.zarr", 1003, 2),
    ]


def test_faces_from_store_matches_cell_layout():
    faces = _faces_value(5, 2)
    np.testing.assert_array_equal(
        faces_from_store(_flat_from_faces(faces), NE, NPG), faces
    )


@pytest.mark.parametrize("hour", [1000, 1002, 1003, 1004])
@pytest.mark.parametrize("level_offset", [0, 1])
def test_source_reads_covering_store_and_level(stores, hour, level_offset):
    ds = _TinySource(stores, level_offset=level_offset)
    variables = [f"U_{1 + level_offset}", f"U_{3 + level_offset}", "ps", "phis"]
    coords = {
        "time": [_iso(hour)],
        "variable": variables,
        "face": np.arange(6),
        "x": np.arange(SIDE),
        "y": np.arange(SIDE),
    }
    got = ds(coords).values[0]  # [variable, face, x, y]
    expected = [
        _faces_value(hour, 1),
        _faces_value(hour, 3),
        _faces_value(hour, 99),
        _faces_value(0, 7),
    ]
    np.testing.assert_array_equal(got, np.stack(expected))
    # A tile read (not the whole sphere) takes the per-cell index path.
    tile = dict(coords, face=np.array([4]), x=np.arange(1, 4), y=np.arange(2, 6))
    np.testing.assert_array_equal(
        ds(tile).values[0, 0, 0], _faces_value(hour, 1)[4, 1:4, 2:6]
    )


def test_source_rejects_bad_requests(stores):
    ds = _TinySource(stores)
    base = {
        "variable": ["ps"],
        "face": np.arange(6),
        "x": np.arange(SIDE),
        "y": np.arange(SIDE),
    }
    with pytest.raises(ValueError, match="not covered"):
        ds(dict(base, time=[_iso(1005)]))
    with pytest.raises(ValueError, match="hour-aligned"):
        ds(dict(base, time=[_iso(1001) + np.timedelta64(30, "m")]))
    with pytest.raises(NotImplementedError):
        ds(dict(base, time=[_iso(1002), _iso(1003)]))
    with pytest.raises(ValueError, match="level_offset"):
        _TinySource(stores, level_offset=2)._parse_variable("U_1")


def test_parse_args_defaults_are_the_training_geometry():
    args = parse_args(
        [
            "--run-name",
            "r",
            "--checkpoint",
            "c.pth",
            "--ic-zarr",
            "a.zarr",
            "--ic-zarr",
            "b.zarr",
            "--initial-time",
            "2016-08-01T00:00:00",
            "out.zarr",
        ]
    )
    assert args.ic_zarr == ["a.zarr", "b.zarr"]
    assert args.output == "out.zarr"
    assert (args.tile_size, args.halo_width, args.dt_hours) == (256, 32, 1.0)
    assert args.precision == "bf16" and args.wind_center_mode is None
    # one tile per forward pass, as in training (polar-face frame)
    assert args.inference_batch_size == 1
