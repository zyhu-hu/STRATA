# Hourly ne128 rollouts from SCREAM initial conditions

How to run the hourly ne128pg2 checkpoints (the large-scale emulator,
`ckpt_scream_lora3ds_replay_409k_merged.pth`, and its parent
`ckpt_scream_lora3ds_403k_merged.pth`) from a SCREAM initial condition, starting
from the SCREAM ne1024pg2 output.

There are two steps:

1. `scripts/ace/build_scream_ne128_ic.py` coarsens SCREAM to an hourly ne128pg2 store.
2. `scripts/ace/run_screamcast_ne128.py` rolls the model out from that store.

## What you need

| Item | Notes |
|---|---|
| Checkpoint `.pth` (1.3 GB each) | Model weights and the training configuration; no optimizer state. |
| `latlon_ne128pg2.nc` | Cell centres of the ne128pg2 grid. |
| `ne128halo16pg2_scrip.nc` | ne128pg2 grid with a 16-element halo, used for the cross-face tile halo. |
| `phis.npy`, `sgh30.npy`, `landfrac.npy` | ne128pg2 static fields (393216 values each). `ScreamcastModel.from_checkpoint` reads them; the rollout then takes the statics from the initial-condition store. |
| `scream_ne128_ic_1994101300.zarr` (optional) | Reference initial condition built from the training data, to check your own build (step 1). |
| SCREAM ne1024pg2 output | The 10-minute `INSTANT` output files containing `PotentialTemperature`, `U`, `V`, `z_mid`, `omega`, `qv`, `ps`, `T_2m`, plus the E3SM topography file on the same grid (`USGS-gtopo30_ne1024np4pg2_x6t-SGH.c20210614.nc`: `PHIS`, `SGH30`, `LANDFRAC`). |
| Software | The Docker image from this repository and a GPU (tested on H100). |

Put the `.nc` and `.npy` files in one directory and point `AUX_DATA_ROOT` at it.
Keep this directory separate from the ne1024 auxiliary directory, since the
`.npy` names are the same.

Use the grid files as shipped; do not regenerate them. On the two polar faces
the model's tile frame depends on the last bits of these coordinates (see
"Batch size" below).

## Step 1: build the initial condition

```bash
export AUX_DATA_ROOT=/path/to/ne128_aux
PYTHONPATH=. python3 scripts/ace/build_scream_ne128_ic.py \
    --scream-netcdf-dir /path/to/scream/run \
    --topo-file /path/to/USGS-gtopo30_ne1024np4pg2_x6t-SGH.c20210614.nc \
    --start 1994-10-13T00:00:00 \
    scream_ne128_1994101300.zarr
```

What the script does:

- **Time.** It takes the SCREAM snapshot at each requested hour, with no time
  averaging. `--end` adds later hours, for example to score a rollout against
  coarsened SCREAM.
- **Levels.** 3-D fields are taken at physical SCREAM levels `35:128:4`
  (0-based, 24 levels).
- **Coarsening.** On each cube face the 2048×2048 ne1024pg2 raster is averaged
  over 8×8 blocks, without area weighting. The fine-tuning data was made the
  same way, and the area weights inside a block differ by less than 0.6 %.
- **Statics.** `phis`, `sgh30` and `landfrac` are coarsened the same way from
  the topography file.

Run time and memory: the script holds one cube face of one variable at a time
(about 2 GB of memory). From the netCDF output it reads all 128 levels, about
80 GB per hour. From the zarr below it reads about 15 GB per hour, which took
about 40 s per hour on one node.

If you also have the ne1024pg2 cubesphere zarr (levels `3:128:4`, one frame
every 10 minutes), use `--scream-zarr PATH --zarr-t0 <time of frame 0>` in
place of the netCDF and topography arguments.

To check your build against the shipped reference, add
`--reference scream_ne128_ic_1994101300.zarr` and build that hour. The script
prints the largest difference per variable. Both inputs come from the same
SCREAM output and the same float32 block mean, so the differences should be
zero or at float32 rounding level.

## Step 2: roll out

```bash
PYTHONPATH=. python3 scripts/ace/run_screamcast_ne128.py \
    --run-name ls_1994101300 \
    --checkpoint /path/to/ckpt_scream_lora3ds_replay_409k_merged.pth \
    --ic-zarr scream_ne128_1994101300.zarr \
    --initial-time 1994-10-13T00:00:00 \
    --n-steps 24 \
    ls_1994101300.zarr
```

- The model step is 1 hour. Output is written every step, on the six cube
  faces (`[time, step, level, face, x, y]`), with a `history` attribute.
- For several ranks, run with `torchrun --nproc_per_node N`. The output does
  not depend on the number of ranks.
- `--precision fp32` runs in float32. The default is bf16, as in training.
- `python3 scripts/ace/run_screamcast_ne128.py --help` lists the other options
  (output variables and levels, tile and halo sizes).

## Batch size

Keep the default `--inference-batch-size 1`. The two polar faces are centred
on the poles. There, the longitude of the tile frame (used for the wind
rotation and the positional encoding) comes from rounding in a reduction, so
it depends on how many tiles share a forward pass. The checkpoints were
trained with one tile per forward pass, and a batch size of 1 reproduces that
frame. Larger batches rotate the polar-face frames by up to about 12 degrees,
which degrades the polar forecasts.
