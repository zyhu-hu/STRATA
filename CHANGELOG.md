# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

- Add `scripts/ace/build_scream_ne128_ic.py`: hourly ne128pg2 initial-condition
  stores from SCREAM ne1024pg2 output (EAMxx netCDF or the cubesphere zarr),
  coarsened as the fine-tuning data was (8x8 block mean per face, physical
  levels 35:128:4). `docs/ne128_scream_rollout.md` walks through building an
  initial condition and rolling out.
- Add `scripts/ace/run_screamcast_ne128.py`: hourly rollout of the ne128pg2
  checkpoints (e.g. the large-scale emulator) from hourly ne128 zarr stores,
  one 320-px halo-padded tile per face, coszr computed per step, output in
  the face-layout zarr of `run_screamcast_nudged.py` with a `history`
  attribute. It runs one tile per forward pass by default,
  as in training: the polar-face tile frame depends on the batch size.
- Add `DiTConfig.wind_center_mode` (`"mean"` default, `"spherical_centroid"`)
  and `ScreamcastModel.from_checkpoint(..., dit_overrides=...)`. The hourly
  ne128 checkpoints train on whole cube faces with the spherical-centroid
  wind-rotation frame. Their embedded configs record the field, but older
  code dropped it on load and ran them with the mean center, which is wrong
  on the polar faces; they now load correctly without an override. Configs
  with `do_rotate_wind` but no `wind_center_mode` now warn.
- Fix the slow pixel-stage depthwise conv (#6): pin physicsnemo main @
  `fae6bb2b`, which contains NVIDIA/physicsnemo#2140. Its Strata
  `DepthwiseConv` defaults to `chunk_size="auto"` instead of fixed
  2-channel chunks, so every config that leaves
  `use_chunked_depthwise_conv` at its default (including checkpoints that
  predate the field) gets the fast path with no config or checkpoint change.
  physicsnemo now requires torch >= 2.13, so the Docker base moves from NGC
  PyTorch 26.01 to 26.09; NATTEN goes to 0.21.7 (built as C++20 for the
  torch 2.14 headers) and earth2grid to v2025.11.1, which fixes NaNs in
  lat/lon regridding west of the first source longitude.
- The model architecture now comes from NVIDIA PhysicsNeMo
  (`physicsnemo.experimental.models.strata`); the local `dit_3d.py` /
  `dit_3d_pixel.py` implementations were replaced by thin wrappers
  (`screamcast/strata_wrappers.py`) that keep geometry, wind rotation, and
  legacy-checkpoint compatibility (`screamcast/checkpoint_compat.py`).
  Pre-migration checkpoints load transparently. The `nvidia-modulus`
  dependency was replaced by `nvidia-physicsnemo` (pinned source archive
  until the 2.2 release).
- Initial public Apache-2.0 release of Strata.
