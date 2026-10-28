# Synthetic session dataloader

`synthetic_sessions.py` makes multiple observations of the same synthetic scene
from a library of cropped ROI footprints. It places each crop within a common
FOV, then changes the scene's geometry and appearance for each session. The ROIs
retain their positions within the FOV. Match `roi_ids` to find the same neuron
across sessions, even though the rows are shuffled independently.

The defaults place 128 base ROIs in a 128-by-128 FOV. Each session independently
warps the crops with `roi_warp_std_px=0.75`, then applies shared session motion.
Dropout and visibility thresholds determine the observed count in each session.

The sparse archive contains ROI crops. It supplies no FOV images, original
neuron positions, or pixel sizes. We sample positions and render each FOV as a
weighted sum of all the placed footprints. There is no added background, and
`fov_noise_std` defaults to zero. Missing ROI extractions remain visible in the
FOV. Set `um_per_pixel` explicitly; `1.0` below is a testing convention.

```python
import torch
from roicat.model_training import synthetic_sessions

dir_library = synthetic_sessions.prepare_sparse_roi_library(
    filepath_npz="/n/holylabs/kempner_rhakim_lab/Lab/rhakim/projects/prj_roicat_paper/data_model_training/sf_sparse_36x36_20220503.npz",
    dir_library="/n/netscratch/bsabatini_lab/Lab/rhakim/temp/hyper_roinet/roi_library",
)
config = synthetic_sessions.SyntheticSessionConfig(um_per_pixel=1.0)
dataset = synthetic_sessions.SyntheticSessionDataset(
    dir_roi_library=dir_library,
    config=config,
    n_scenes=1000,
    seed=0,
)
dataloader = torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=4,
    shuffle=True,
    num_workers=2,
    collate_fn=synthetic_sessions.collate_synthetic_sessions,
    generator=torch.Generator().manual_seed(0),
)
batch = next(iter(dataloader))
```

Run extraction once on a compute node before creating workers. The function
streams the compressed CSR arrays to scratch storage and renames the completed
directory into place. Workers open read-only memory maps and load only selected
ROI rows into dense arrays. Reuse checks the source path, size, and modification
time. For spawned workers, construct and iterate the DataLoader inside a script's
`if __name__ == "__main__":` guard.

## Session generation

A dataset item is fixed for `(seed, scene_index)` regardless of access order or
worker count. Change the seed to generate a different collection of scenes.
Generation uses local NumPy and Torch generators and leaves global random state
untouched.

1. Select distinct templates, normalize their peak intensities, and place them
   within the FOV. Render the base FOV from all the footprints.
2. Independently warp each ROI crop with `augmentation.WarpPoints` and put it
   back at its assigned position. Rebuild the FOV from these deformed footprints.
   `roi_warp_std_px` controls the crop displacement strength in pixels per axis.
   Sample shared affine motion and smooth deformation; `elastic_std_px` controls
   the shared displacement strength. `n_elastic_warps` and
   `elastic_radius_bounds` control the Gaussian fields for both kinds of motion.
3. Apply one sampling grid to the stacked ROIs and FOV with
   `helpers.remap_images`. Apply shared Gaussian blur and intensity gain.
4. Use `augmentation.AddGaussianNoise` for independent multiplicative ROI noise
   and optional additive FOV noise. Multiplicative noise keeps empty ROI pixels
   at zero. Keep clean footprints as prediction targets.
5. Omit poorly visible ROIs and randomly drop extractions. Shuffle the retained
   rows, keeping their identities and source-template indices together.

`WarpPoints.generate_sampling_grid` exposes the sampled geometry. Both updated
augmentations accept an optional local Torch generator and remain usable in the
existing TorchScript ROInet pipeline.

## Batch fields

`B` is the batch size, `S` the session count, `N` the largest observed ROI count
in the batch, and `(H,W)` the FOV dimensions.

| Key | Shape | Meaning |
| --- | --- | --- |
| `roi_images` | `(B,S,N,H,W)` | Positioned footprints with independent observation noise |
| `roi_images_clean` | `(B,S,N,H,W)` | Footprints after shared geometry, blur, and gain, before independent noise |
| `roi_ids` | `(B,S,N)` | Neuron identities across sessions |
| `roi_library_indices` | `(B,S,N)` | Source-template row indices |
| `roi_valid` | `(B,S,N)` | Observed rows; use to mask attention and losses |
| `centroids_yx` | `(B,S,N,2)` | Intensity-weighted positions in pixels |
| `fov_image` | `(B,S,1,H,W)` | Rendered FOV, including neurons with missing extractions |
| `fov_valid` | `(B,S,1,H,W)` | Pixels whose sampling coordinates are inside the base FOV, before blur |
| `grid_session_to_base_yx` | `(B,S,H,W,2)` | Shared-motion sampling coordinates in `(y,x)` pixels; excludes independent crop deformation |
| `roi_ids_base`, `roi_library_indices_base` | `(B,N_base)` | Base identities and template indices |
| `centroids_base_yx` | `(B,N_base,2)` | Positions before session changes |
| `is_observed` | `(B,S,N_base)` | Observed identities in base order |
| `scene_ids`, `um_per_pixel` | `(B,)` | Scene index and pixel size |

Padded identities and source indices are `-1`; padded images and positions are
zero. Sessions with no observed ROIs are supported. Always mask padded rows when
comparing identities: two padded `-1` values would otherwise appear to match.
Multiply centroid coordinates by the corresponding `um_per_pixel` to get microns.

Identities are unique across scenes within one dataset. They do not identify the
same cells across separately constructed datasets. A source template can also be
reused as a different placed neuron in another scene. For training and validation,
partition source rows first and pass disjoint `indices_roi_library` subsets.

The sampling grid describes shared motion in base coordinates. Independent crop
deformation happens before this grid and is excluded from it. A positive sampling
translation shifts the observed image in the opposite direction. The grid does
not directly describe forward cell motion, and deformation can fold. Visibility
uses retained footprint mass before blur or gain, which also changes with spatial
magnification.

These dense tiles are intended for initial encoder experiments. Larger FOVs will
need a sparse or local-patch representation. The preview uses
`helpers.rand_cmap` and `visualization.compute_colored_FOV`, with all sessions
rendered together to preserve colors across missing and reordered observations.
The animation shows independently generated sessions and is saved with
`helpers.save_webp`. The numerical colored FOVs are saved alongside its inputs.

Run checks on a compute node:

```bash
python -m pytest tests/test_synthetic_sessions.py -q
```
