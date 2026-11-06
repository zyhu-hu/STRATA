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
"""``DiTConfig.wind_center_mode`` survives a checkpoint config and reaches the
wind rotation.

The hourly ne128 checkpoints train on whole cube faces and were trained with
the spherical-centroid frame center. The mean center puts a polar face's
center near 50 degrees latitude instead of the pole; if the field is dropped
from the embedded config, those checkpoints run with misplaced wind frames.
"""

import math
import warnings

import numpy as np
import pytest
import torch

import train as train_module
from screamcast.config import DiTConfig, PixelDiTConfig


def _north_polar_face(n: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    """Lat/lon (radians, ``[1, n, n]``) of an equiangular cube face centered
    on the north pole, built from the gnomonic projection directly."""
    a = (np.arange(n) + 0.5) / n * (np.pi / 2) - np.pi / 4
    ta, tb = np.meshgrid(np.tan(a), np.tan(a), indexing="ij")
    r = np.sqrt(ta**2 + tb**2 + 1.0)
    lat = np.arcsin(1.0 / r)
    lon = np.arctan2(tb, ta)
    return (
        torch.from_numpy(lat).float().unsqueeze(0),
        torch.from_numpy(lon).float().unsqueeze(0),
    )


def test_from_dict_keeps_wind_center_mode():
    cfg = DiTConfig.from_dict(
        {"do_rotate_wind": True, "wind_center_mode": "spherical_centroid"}
    )
    assert cfg.wind_center_mode == "spherical_centroid"


def test_from_dict_warns_when_rotate_wind_config_lacks_center():
    with pytest.warns(UserWarning, match="wind_center_mode"):
        cfg = DiTConfig.from_dict({"do_rotate_wind": True})
    assert cfg.wind_center_mode == "mean"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        DiTConfig.from_dict({"do_rotate_wind": False})


def test_unknown_wind_center_mode_rejected():
    with pytest.raises(ValueError, match="wind_center_mode"):
        DiTConfig(wind_center_mode="centroid")


def _center_lat(network) -> float:
    lat, lon = _north_polar_face()
    lat0, _ = network._tile_center_fn(lat, lon)
    return float(lat0.squeeze())


@pytest.mark.parametrize("model_type", ["dit3d", "pixeldit"])
def test_factory_wind_center_on_polar_face(model_type):
    """The configured center reaches the network built by the train factory:
    by symmetry the spherical centroid of a pole-centered face is the pole;
    the mean center undershoots it by tens of degrees."""
    lats = {}
    for mode in ("mean", "spherical_centroid"):
        dit_cfg = DiTConfig(
            embed_dim=64,
            n_layers=1,
            num_heads=2,
            do_rotate_wind=True,
            wind_center_mode=mode,
        )
        kwargs = dict(
            in_channels=6,
            out_channels=6,
            nside=64,
            tile_size=32,
            dit_cfg=dit_cfg,
            do_bf16_mixed=False,
            depth_levels=4,
            wind_channel_indices=(0, 1),
            grid_type="healpix",
        )
        with torch.device("meta"):
            if model_type == "pixeldit":
                network = train_module.build_strata(
                    pixel_cfg=PixelDiTConfig(embed_dim=32, n_layers=1), **kwargs
                )
            else:
                network = train_module.build_backbone(**kwargs)
        lats[mode] = _center_lat(network)
    assert lats["spherical_centroid"] == pytest.approx(math.pi / 2, abs=1e-4)
    assert lats["mean"] < math.radians(60.0)
