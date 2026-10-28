"""Make synthetic tracking sessions from a library of cropped ROI footprints.

A scene places several ROIs in a common FOV. Each session independently deforms
the ROI crops, then applies shared spatial motion, blur, and gain to the ROIs and
their rendered FOV. Identities remain available when an extraction is missing.
"""

from typing import Any, Optional, Sequence

import dataclasses
import json
import os
from pathlib import Path
import shutil
import tempfile
import zipfile

import numpy as np
import scipy.ndimage
import scipy.sparse
import torch

from .. import helpers
from . import augmentation


def prepare_sparse_roi_library(filepath_npz: str, dir_library: str) -> str:
    """Extract a sparse ROI archive into arrays that workers can memory-map.

    Call once before constructing a DataLoader. Extraction streams the archive
    without loading the full library into RAM. Existing extracted arrays are
    reused when the source path, file size, and modification time match.

    Args:
        filepath_npz (str):
            SciPy CSR archive with shape ``(n_rois, height_roi * width_roi)``.
        dir_library (str):
            Destination directory. Use scratch storage for large libraries.

    Returns:
        (str):
            Absolute path to the directory containing the extracted CSR arrays.

    Raises:
        ValueError:
            The archive is not CSR, or the destination contains arrays extracted
            from a different source file.
    """
    filepath_source = Path(filepath_npz).resolve(strict=True)
    dir_destination = Path(dir_library).resolve()
    stat_source = filepath_source.stat()
    metadata_source = {
        "filepath_npz": str(filepath_source),
        "size_bytes": stat_source.st_size,
        "mtime_ns": stat_source.st_mtime_ns,
    }
    filenames_arrays = (
        "data.npy",
        "indices.npy",
        "indptr.npy",
        "shape.npy",
        "format.npy",
    )

    ## Check the source before reusing extracted arrays.
    if dir_destination.exists():
        metadata_cached = json.loads((dir_destination / "source.json").read_text())
        if metadata_cached != metadata_source:
            raise ValueError("ROI library cache does not match the source archive.")
        for filename_array in filenames_arrays:
            if not (dir_destination / filename_array).is_file():
                raise FileNotFoundError(dir_destination / filename_array)
        return str(dir_destination)

    ## Extract to a temporary directory so workers cannot read incomplete files.
    dir_destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{dir_destination.name}_", dir=dir_destination.parent
    ) as dir_temporary:
        with zipfile.ZipFile(file=filepath_source) as archive:
            for filename_array in filenames_arrays:
                with archive.open(name=filename_array) as stream_source:
                    with (Path(dir_temporary) / filename_array).open(
                        "wb"
                    ) as stream_output:
                        shutil.copyfileobj(
                            fsrc=stream_source, fdst=stream_output, length=2**20
                        )
        format_sparse = np.load(
            file=Path(dir_temporary) / "format.npy", allow_pickle=False
        )
        if format_sparse.item() != b"csr":
            raise ValueError("The ROI archive must use CSR sparse format.")
        (Path(dir_temporary) / "source.json").write_text(
            json.dumps(metadata_source, indent=2) + "\n"
        )
        os.rename(src=dir_temporary, dst=dir_destination)
    return str(dir_destination)


@dataclasses.dataclass(frozen=True)
class SyntheticSessionConfig:
    """Control ROI placement and differences between synthetic sessions.

    Args:
        um_per_pixel (float):
            Pixel size in microns. The sparse archive does not store this value.
        n_sessions (int):
            Number of observations of each base scene.
        n_rois (int):
            Number of distinct ROI templates placed in each scene.
        shape_roi (tuple[int, int]):
            Crop dimensions in ``(height, width)`` order.
        shape_fov (tuple[int, int]):
            Scene dimensions in ``(height, width)`` order.
        max_translation_px (float):
            Maximum sampling translation along each axis. A positive sampling
            translation moves the observed image in the opposite direction.
        max_rotation_deg (float):
            Maximum absolute sampling rotation about the scene center.
        scale_bounds (tuple[float, float]):
            Bounds on spatial magnification.
        elastic_std_px (float):
            Standard deviation of the smooth displacement along each axis, in
            pixels. The displacement is centered and scaled to this value.
        n_elastic_warps (int):
            Number of Gaussian displacement fields sampled by ``WarpPoints``
            for shared motion and each independent crop deformation.
        elastic_radius_bounds (tuple[float, float]):
            Bounds on Gaussian radii in normalized image coordinates, where the
            image extends from -1 to 1 along each axis.
        roi_warp_std_px (float):
            Standard deviation of independent smooth displacement within each
            ROI crop, in pixels per axis. Applied before shared session motion.
        blur_sigma_bounds (tuple[float, float]):
            Bounds on shared Gaussian blur sigma, in pixels.
        gain_bounds (tuple[float, float]):
            Bounds on shared intensity gain.
        fov_noise_std (float):
            Standard deviation of optional additive FOV noise. Zero leaves the
            background empty except for ROI signal and its blur.
        roi_noise_std (float):
            Standard deviation of independent multiplicative footprint noise.
        dropout_probability (float):
            Probability of omitting an ROI extraction. Its signal remains in
            the FOV image.
        min_visible_fraction (float):
            Minimum retained ROI mass after deformation, relative to base mass.
            Computed before blur, gain, and independent observation noise.
    """

    um_per_pixel: float
    n_sessions: int = 2
    n_rois: int = 128
    shape_roi: tuple[int, int] = (36, 36)
    shape_fov: tuple[int, int] = (128, 128)
    max_translation_px: float = 4.0
    max_rotation_deg: float = 5.0
    scale_bounds: tuple[float, float] = (0.95, 1.05)
    elastic_std_px: float = 0.5
    n_elastic_warps: int = 3
    elastic_radius_bounds: tuple[float, float] = (0.2, 0.6)
    roi_warp_std_px: float = 0.75
    blur_sigma_bounds: tuple[float, float] = (0.0, 0.75)
    gain_bounds: tuple[float, float] = (0.8, 1.2)
    fov_noise_std: float = 0.0
    roi_noise_std: float = 0.05
    dropout_probability: float = 0.2
    min_visible_fraction: float = 0.5

    def __post_init__(self) -> None:
        """Validate image dimensions and perturbation bounds before loading data."""
        ## Validate counts and dimensions.
        if not isinstance(self.n_sessions, int) or self.n_sessions < 2:
            raise ValueError("Require at least two sessions.")
        if not isinstance(self.n_rois, int) or self.n_rois < 1:
            raise ValueError("Require at least one base ROI.")
        if not isinstance(self.n_elastic_warps, int) or self.n_elastic_warps < 1:
            raise ValueError("n_elastic_warps must be a positive integer.")
        for shape_image in (self.shape_roi, self.shape_fov):
            if len(shape_image) != 2 or any(
                not isinstance(size_axis, int) or size_axis < 2
                for size_axis in shape_image
            ):
                raise ValueError("Image shapes must contain two integers >= 2.")
        if any(
            size_roi > size_fov
            for size_roi, size_fov in zip(self.shape_roi, self.shape_fov)
        ):
            raise ValueError("ROI crops must fit inside the synthetic FOV.")

        ## Validate paired bounds and positive physical scales.
        for bounds in (
            self.scale_bounds,
            self.elastic_radius_bounds,
            self.blur_sigma_bounds,
            self.gain_bounds,
        ):
            if len(bounds) != 2 or bounds[0] > bounds[1]:
                raise ValueError(
                    "Bounds must contain an ordered (minimum, maximum) pair."
                )
        values_positive = (
            self.um_per_pixel,
            *self.scale_bounds,
            *self.elastic_radius_bounds,
            *self.gain_bounds,
        )
        if any(not np.isfinite(value) or value <= 0 for value in values_positive):
            raise ValueError(
                "Pixel scale, spatial scales, radii, and gains must be finite and positive."
            )

        ## Noise and deformation magnitudes may be zero; probabilities are bounded.
        values_nonnegative = (
            self.max_translation_px,
            self.max_rotation_deg,
            self.elastic_std_px,
            self.roi_warp_std_px,
            *self.blur_sigma_bounds,
            self.fov_noise_std,
            self.roi_noise_std,
        )
        if any(not np.isfinite(value) or value < 0 for value in values_nonnegative):
            raise ValueError("Perturbation magnitudes must be finite and nonnegative.")
        if not 0 <= self.dropout_probability <= 1:
            raise ValueError("dropout_probability must lie in [0, 1].")
        if not 0 < self.min_visible_fraction <= 1:
            raise ValueError("min_visible_fraction must lie in (0, 1].")


def _calculate_centroids(roi_images: torch.Tensor) -> torch.Tensor:
    """Calculate intensity-weighted positions of nonempty ROI footprints.

    Args:
        roi_images (torch.Tensor):
            Positioned footprints with shape ``(n_rois, height, width)``.

    Returns:
        (torch.Tensor):
            Centroids in pixel ``(y, x)`` coordinates. Shape: ``(n_rois, 2)``.
    """
    coords_y = torch.arange(end=roi_images.shape[-2], dtype=roi_images.dtype)
    coords_x = torch.arange(end=roi_images.shape[-1], dtype=roi_images.dtype)
    masses_roi = roi_images.sum(dim=(-2, -1))
    centroids_y = (roi_images.sum(dim=-1) * coords_y[None, :]).sum(dim=-1) / masses_roi
    centroids_x = (roi_images.sum(dim=-2) * coords_x[None, :]).sum(dim=-1) / masses_roi
    return torch.stack(tensors=(centroids_y, centroids_x), dim=-1)


def _sample_elastic_displacement(
    transform: augmentation.WarpPoints, std_px: float, generator: torch.Generator
) -> torch.Tensor:
    """Sample a centered smooth displacement with a specified pixel strength.

    Args:
        transform (augmentation.WarpPoints):
            Configured deformation and output image dimensions.
        std_px (float):
            Standard deviation along each image axis, in pixels.
        generator (torch.Generator):
            Local CPU generator for deformation parameters.

    Returns:
        (torch.Tensor):
            Pixel displacements in ``(y, x)`` order, shape ``(height, width, 2)``.
    """
    ## Convert the repo augmentation's normalized coordinates to pixel coordinates.
    grid_normalized_xy = transform.generate_sampling_grid(
        device=torch.device("cpu"), generator=generator
    )[0]
    shape_image_xy = torch.as_tensor(
        data=transform.img_size_out[::-1], dtype=torch.float32
    )
    grid_pixels_yx = ((grid_normalized_xy + 1) * (shape_image_xy - 1) / 2).flip(
        dims=(-1,)
    )
    coords_y, coords_x = torch.meshgrid(
        torch.arange(end=transform.img_size_out[0], dtype=torch.float32),
        torch.arange(end=transform.img_size_out[1], dtype=torch.float32),
        indexing="ij",
    )
    displacement_yx = grid_pixels_yx - torch.stack(tensors=(coords_y, coords_x), dim=-1)

    ## Center each axis and set its strength independently of the Gaussian radii.
    displacement_yx = displacement_yx - displacement_yx.mean(dim=(0, 1), keepdim=True)
    std_displacement_yx = displacement_yx.std(dim=(0, 1), keepdim=True).clamp(min=1e-6)
    return displacement_yx * (std_px / std_displacement_yx)


def _place_roi_crops(
    roi_crops: torch.Tensor, positions_yx: torch.Tensor, shape_fov: tuple[int, int]
) -> torch.Tensor:
    """Place cropped footprints at their assigned positions within a shared FOV.

    Args:
        roi_crops (torch.Tensor):
            Crops with shape ``(n_rois, height_roi, width_roi)``.
        positions_yx (torch.Tensor):
            Top-left crop positions, shape ``(n_rois, 2)``.
        shape_fov (tuple[int, int]):
            FOV dimensions in ``(height, width)`` order.

    Returns:
        (torch.Tensor):
            Positioned footprints, shape ``(n_rois, height_fov, width_fov)``.
    """
    roi_images = torch.zeros(size=(len(roi_crops), *shape_fov), dtype=roi_crops.dtype)
    for index_roi, (position_y, position_x) in enumerate(positions_yx.tolist()):
        roi_images[
            index_roi,
            position_y : position_y + roi_crops.shape[-2],
            position_x : position_x + roi_crops.shape[-1],
        ] = roi_crops[index_roi]
    return roi_images


def _make_sampling_grid(
    config: SyntheticSessionConfig,
    rng: np.random.Generator,
    generator: torch.Generator,
    transform_elastic: augmentation.WarpPoints,
) -> torch.Tensor:
    """Combine affine motion and smooth deformation in one sampling grid.

    Args:
        config (SyntheticSessionConfig):
            Scene dimensions and deformation parameters.
        rng (np.random.Generator):
            Local generator for affine parameters.
        generator (torch.Generator):
            Local CPU generator for ``WarpPoints``.
        transform_elastic (augmentation.WarpPoints):
            Configured smooth deformation augmentation.

    Returns:
        (torch.Tensor):
            Base-scene pixel coordinates to sample at each output-session pixel,
            in ``(y, x)`` order. Shape: ``(height, width, 2)``.
    """
    ## Sample rotation, magnification, and translation around the FOV center.
    angle_radians = np.deg2rad(
        rng.uniform(low=-config.max_rotation_deg, high=config.max_rotation_deg)
    )
    scale = rng.uniform(low=config.scale_bounds[0], high=config.scale_bounds[1])
    matrix_rotation_xy = (
        np.asarray(
            [
                [np.cos(angle_radians), -np.sin(angle_radians)],
                [np.sin(angle_radians), np.cos(angle_radians)],
            ],
            dtype=np.float32,
        )
        / scale
    )
    center_xy = (np.asarray(config.shape_fov[::-1], dtype=np.float32) - 1) / 2
    translation_xy = rng.uniform(
        low=-config.max_translation_px, high=config.max_translation_px, size=2
    )
    matrix_affine_xy = np.column_stack(
        (
            matrix_rotation_xy,
            center_xy - matrix_rotation_xy @ center_xy + translation_xy,
        )
    ).astype(np.float32)

    ## The repo helper returns pixel coordinates in (x, y) order.
    grid_affine_xy = helpers.warp_matrix_to_remappingIdx(
        warp_matrix=matrix_affine_xy, x=config.shape_fov[1], y=config.shape_fov[0]
    )
    grid_session_to_base_yx = torch.as_tensor(data=grid_affine_xy).flip(dims=(-1,))
    if config.elastic_std_px == 0:
        return grid_session_to_base_yx

    displacement_yx = _sample_elastic_displacement(
        transform=transform_elastic, std_px=config.elastic_std_px, generator=generator
    )
    return grid_session_to_base_yx + displacement_yx


def _warp_images(images: torch.Tensor, grid_yx: torch.Tensor) -> torch.Tensor:
    """Apply the same spatial deformation to every image in a stack.

    Args:
        images (torch.Tensor):
            Images with shape ``(n_images, height, width)``.
        grid_yx (torch.Tensor):
            Input sampling coordinates in pixels, with shape ``(height, width, 2)``.

    Returns:
        (torch.Tensor):
            Warped images with the same shape as ``images``.
    """
    ## Treat images as channels so remap_images uses one grid for the whole stack.
    images_warped = helpers.remap_images(
        images=images,
        remappingIdx=grid_yx.flip(dims=(-1,)),
        backend="torch",
        interpolation_method="linear",
        border_mode="constant",
        border_value=0,
        device="cpu",
    )
    return images_warped.reshape(
        images.shape
    )  ## Restore singleton image axes removed by the helper.


class SyntheticSessionDataset(torch.utils.data.Dataset):
    """Generate positioned ROI observations from memory-mapped sparse templates.

    Each item contains one scene and its sessions. The scene is fixed for a
    given ``(seed, index_scene)`` regardless of access order or worker count.
    Sessions shuffle ROI rows independently; match ``roi_ids`` to find the same
    neuron in different sessions. IDs are unique across scenes within this dataset.

    Args:
        dir_roi_library (str):
            Directory returned by :func:`prepare_sparse_roi_library`.
        config (SyntheticSessionConfig):
            Scene dimensions and session perturbations.
        n_scenes (int):
            Number of synthetic scenes in the dataset.
        seed (int):
            Nonnegative seed for scene generation.
        indices_roi_library (Optional[Sequence[int]]):
            Allowed source rows. Use disjoint subsets for training and validation.
    """

    def __init__(
        self,
        dir_roi_library: str,
        config: SyntheticSessionConfig,
        n_scenes: int = 1000,
        seed: int = 0,
        indices_roi_library: Optional[Sequence[int]] = None,
    ) -> None:
        ## Check dimensions without opening the large sparse data arrays.
        self.dir_roi_library = str(Path(dir_roi_library).resolve(strict=True))
        self.config = config
        self.n_scenes = n_scenes
        self.seed = seed
        self._roi_library = None
        shape_library = np.load(
            file=Path(self.dir_roi_library) / "shape.npy", allow_pickle=False
        )
        if shape_library.shape != (2,) or shape_library[1] != np.prod(config.shape_roi):
            raise ValueError(
                "Library shape must be (n_rois, product(config.shape_roi))."
            )
        self.shape_roi_library = tuple(int(size_axis) for size_axis in shape_library)
        if not isinstance(n_scenes, int) or n_scenes < 1:
            raise ValueError("n_scenes must be a positive integer.")
        if not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a nonnegative integer.")

        ## Restrict template selection when making separate training and validation sets.
        self.indices_roi_library = None
        if indices_roi_library is not None:
            indices_library = np.asarray(a=indices_roi_library)
            if indices_library.ndim != 1 or not np.issubdtype(
                indices_library.dtype, np.integer
            ):
                raise ValueError("indices_roi_library must be a 1D integer sequence.")
            if np.any(indices_library < 0) or np.any(
                indices_library >= self.shape_roi_library[0]
            ):
                raise ValueError("Source ROI indices are outside the library.")
            if len(np.unique(ar=indices_library)) != len(indices_library):
                raise ValueError("Source ROI indices must be unique.")
            self.indices_roi_library = indices_library.astype(dtype=np.int64, copy=True)
        self.n_templates_available = (
            self.shape_roi_library[0]
            if self.indices_roi_library is None
            else len(self.indices_roi_library)
        )
        if self.n_templates_available < config.n_rois:
            raise ValueError("Need at least config.n_rois distinct source templates.")

        ## Reuse the repo's deformation and noise augmentations with local generators.
        self.transform_elastic = augmentation.WarpPoints(
            r=list(config.elastic_radius_bounds),
            n_warps=config.n_elastic_warps,
            prob=1.0,
            img_size_in=list(config.shape_fov),
            img_size_out=list(config.shape_fov),
            mode="bilinear",
        )
        self.transform_roi_warp = augmentation.WarpPoints(
            r=list(config.elastic_radius_bounds),
            n_warps=config.n_elastic_warps,
            prob=1.0,
            img_size_in=list(config.shape_roi),
            img_size_out=list(config.shape_roi),
            mode="bilinear",
        )
        self.transform_roi_noise = augmentation.AddGaussianNoise(
            std=config.roi_noise_std, level_bounds=(1.0, 1.0), prob=1.0
        )
        self.transform_fov_noise = augmentation.AddGaussianNoise(
            std=config.fov_noise_std, level_bounds=(1.0, 1.0), prob=1.0
        )

    def __len__(self) -> int:
        """Return the number of synthetic scenes."""
        return self.n_scenes

    def __getstate__(self) -> dict[str, Any]:
        """Reopen memory maps in spawned workers instead of copying the library."""
        state_dataset = self.__dict__.copy()
        state_dataset["_roi_library"] = None
        return state_dataset

    def _open_library(self) -> scipy.sparse.csr_array:
        """Open read-only CSR arrays in the worker that requests a scene."""
        if self._roi_library is None:
            dir_library = Path(self.dir_roi_library)
            arrays_library = {
                name_array: np.load(
                    file=dir_library / f"{name_array}.npy",
                    mmap_mode="r",
                    allow_pickle=False,
                )
                for name_array in ("data", "indices", "indptr")
            }
            if (
                arrays_library["data"].ndim != 1
                or arrays_library["indices"].shape != arrays_library["data"].shape
                or arrays_library["indptr"].shape != (self.shape_roi_library[0] + 1,)
                or arrays_library["indptr"][0] != 0
                or arrays_library["indptr"][-1] != len(arrays_library["data"])
            ):
                raise ValueError("Malformed CSR library arrays.")
            self._roi_library = scipy.sparse.csr_array(
                arg1=(
                    arrays_library["data"],
                    arrays_library["indices"],
                    arrays_library["indptr"],
                ),
                shape=self.shape_roi_library,
                copy=False,
            )
        return self._roi_library

    def _render_base_scene(
        self, index_scene: int, rng: np.random.Generator
    ) -> dict[str, torch.Tensor]:
        """Select templates, place them in the FOV, and assign their identities."""
        ## Read only selected rows and scale each crop to unit peak intensity.
        indices_source = rng.choice(
            a=self.n_templates_available, size=self.config.n_rois, replace=False
        )
        if self.indices_roi_library is not None:
            indices_source = self.indices_roi_library[indices_source]
        roi_crops = self._open_library()[indices_source].toarray().astype(np.float32)
        roi_crops = roi_crops.reshape(-1, *self.config.shape_roi)
        if (
            not np.isfinite(roi_crops).all()
            or np.any(roi_crops < 0)
            or np.any(roi_crops.max(axis=(1, 2)) <= 0)
        ):
            raise ValueError(
                f"Selected source rows must be finite, nonnegative, nonempty ROIs: {indices_source.tolist()}"
            )
        roi_crops = roi_crops / roi_crops.max(axis=(1, 2), keepdims=True)

        ## Position whole crops in a shared FOV, preserving their relative locations.
        positions_crop_yx = torch.as_tensor(
            data=rng.integers(
                low=0,
                high=np.asarray(self.config.shape_fov)
                - np.asarray(self.config.shape_roi)
                + 1,
                size=(self.config.n_rois, 2),
            ),
            dtype=torch.int64,
        )
        roi_crops = torch.as_tensor(data=roi_crops)
        roi_images_base = _place_roi_crops(
            roi_crops=roi_crops,
            positions_yx=positions_crop_yx,
            shape_fov=self.config.shape_fov,
        )

        ## Render the FOV from all neurons. There is no added ambient background.
        intensities_roi = torch.as_tensor(
            data=rng.uniform(low=0.5, high=1.0, size=self.config.n_rois),
            dtype=torch.float32,
        )
        fov_image_base = (roi_images_base * intensities_roi[:, None, None]).sum(dim=0)
        fov_peak_base = fov_image_base.max()
        fov_image_base = fov_image_base / fov_peak_base
        roi_ids_base = torch.arange(
            start=index_scene * self.config.n_rois,
            end=(index_scene + 1) * self.config.n_rois,
            dtype=torch.int64,
        )
        return {
            "roi_images": roi_images_base,
            "roi_crops": roi_crops,
            "positions_crop_yx": positions_crop_yx,
            "intensities_roi": intensities_roi,
            "fov_peak": fov_peak_base,
            "fov_image": fov_image_base[None, ...],
            "roi_ids": roi_ids_base,
            "roi_library_indices": torch.as_tensor(
                data=indices_source, dtype=torch.int64
            ),
        }

    def _render_session(
        self, scene_base: dict[str, torch.Tensor], seed_session: np.random.SeedSequence
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Deform a base scene and choose which ROI extractions are observed.

        Returns:
            (tuple):
                session (dict[str, torch.Tensor]):
                    Positioned footprints, FOV image, identities, and geometry.
                is_observed (torch.Tensor):
                    Boolean observations in base-ROI order. Shape: ``(n_rois,)``.
        """
        ## Use independent generators for geometry and observation noise.
        rng_session = np.random.default_rng(seed=seed_session)
        seed_geometry, seed_noise, seed_roi_warp = seed_session.spawn(n_children=3)
        generator_geometry = torch.Generator(device="cpu").manual_seed(
            int(seed_geometry.generate_state(n_words=1, dtype=np.uint64)[0])
        )
        generator_noise = torch.Generator(device="cpu").manual_seed(
            int(seed_noise.generate_state(n_words=1, dtype=np.uint64)[0])
        )
        grid_yx = _make_sampling_grid(
            config=self.config,
            rng=rng_session,
            generator=generator_geometry,
            transform_elastic=self.transform_elastic,
        )

        ## Deform each ROI independently within its crop, then rebuild the FOV.
        roi_images_session_base = scene_base["roi_images"]
        fov_image_session_base = scene_base["fov_image"]
        if self.config.roi_warp_std_px > 0:
            generator_roi_warp = torch.Generator(device="cpu").manual_seed(
                int(seed_roi_warp.generate_state(n_words=1, dtype=np.uint64)[0])
            )
            coords_y, coords_x = torch.meshgrid(
                torch.arange(end=self.config.shape_roi[0], dtype=torch.float32),
                torch.arange(end=self.config.shape_roi[1], dtype=torch.float32),
                indexing="ij",
            )
            grid_crop_yx = torch.stack(tensors=(coords_y, coords_x), dim=-1)
            roi_crops_warped = []
            for roi_crop in scene_base["roi_crops"]:
                displacement_yx = _sample_elastic_displacement(
                    transform=self.transform_roi_warp,
                    std_px=self.config.roi_warp_std_px,
                    generator=generator_roi_warp,
                )
                roi_crops_warped.append(
                    _warp_images(
                        images=roi_crop[None, ...],
                        grid_yx=grid_crop_yx + displacement_yx,
                    )[0]
                )
            roi_images_session_base = _place_roi_crops(
                roi_crops=torch.stack(tensors=roi_crops_warped, dim=0),
                positions_yx=scene_base["positions_crop_yx"],
                shape_fov=self.config.shape_fov,
            )
            fov_image_session_base = (
                roi_images_session_base * scene_base["intensities_roi"][:, None, None]
            ).sum(dim=0)[None, ...] / scene_base["fov_peak"]

        ## Share spatial deformation, blur, and gain across all ROIs and the FOV.
        images_warped = _warp_images(
            images=torch.cat(
                tensors=(roi_images_session_base, fov_image_session_base), dim=0
            ),
            grid_yx=grid_yx,
        )
        masses_roi_base = scene_base["roi_images"].sum(dim=(-2, -1))
        fractions_visible = images_warped[:-1].sum(dim=(-2, -1)) / masses_roi_base
        sigma_blur = rng_session.uniform(
            low=self.config.blur_sigma_bounds[0], high=self.config.blur_sigma_bounds[1]
        )
        gain = rng_session.uniform(
            low=self.config.gain_bounds[0], high=self.config.gain_bounds[1]
        )
        images_clean = (
            torch.as_tensor(
                data=scipy.ndimage.gaussian_filter(
                    input=images_warped.numpy(),
                    sigma=(0, sigma_blur, sigma_blur),
                    mode="constant",
                )
            )
            * gain
        )

        ## Multiplicative footprint noise leaves empty pixels at zero.
        roi_images_observed = images_clean[:-1]
        if self.config.roi_noise_std > 0:
            factors_noise_roi = self.transform_roi_noise(
                tensor=torch.ones_like(input=roi_images_observed),
                generator=generator_noise,
            )
            roi_images_observed = (roi_images_observed * factors_noise_roi).clamp(min=0)
        fov_image_observed = images_clean[-1:]
        if self.config.fov_noise_std > 0:
            fov_image_observed = self.transform_fov_noise(
                tensor=fov_image_observed, generator=generator_noise
            ).clamp(min=0)

        ## Missing extractions remain in the FOV; shuffle the retained ROI rows.
        is_observed = (
            (fractions_visible >= self.config.min_visible_fraction)
            & (roi_images_observed.sum(dim=(-2, -1)) > 0)
            & torch.as_tensor(
                data=rng_session.random(size=self.config.n_rois)
                >= self.config.dropout_probability
            )
        )
        indices_observed = torch.as_tensor(
            data=rng_session.permutation(np.flatnonzero(is_observed.numpy())),
            dtype=torch.int64,
        )
        roi_images_session = roi_images_observed[indices_observed]
        mask_fov_valid = (grid_yx >= 0).all(dim=-1) & (
            grid_yx <= torch.as_tensor(data=self.config.shape_fov) - 1
        ).all(dim=-1)
        session = {
            "roi_images": roi_images_session,
            "roi_images_clean": images_clean[indices_observed],
            "roi_ids": scene_base["roi_ids"][indices_observed],
            "roi_library_indices": scene_base["roi_library_indices"][indices_observed],
            "centroids_yx": _calculate_centroids(roi_images=roi_images_session),
            "fov_image": fov_image_observed,
            "fov_valid": mask_fov_valid[None, ...],
            "grid_session_to_base_yx": grid_yx,
        }
        return session, is_observed

    def __getitem__(self, index_scene: int) -> dict[str, Any]:
        """Render one scene and its independently perturbed sessions.

        Args:
            index_scene (int):
                Scene index between zero and ``n_scenes - 1``.

        Returns:
            (dict[str, Any]):
                scene_id (int): Dataset index.
                um_per_pixel (float): Pixel size in microns.
                roi_ids_base (torch.Tensor): Identities, shape ``(n_rois,)``.
                roi_library_indices_base (torch.Tensor): Source rows, shape ``(n_rois,)``.
                centroids_base_yx (torch.Tensor): Base positions, shape ``(n_rois, 2)``.
                is_observed (torch.Tensor): Visibility, shape ``(n_sessions, n_rois)``.
                sessions (list[dict[str, torch.Tensor]]):
                    Each session contains footprints ``roi_images`` and
                    ``roi_images_clean`` with shape ``(n_observed, height, width)``,
                    identities and source rows with shape ``(n_observed,)``, and
                    ``centroids_yx`` with shape ``(n_observed, 2)``. ``fov_image``
                    and boolean ``fov_valid`` have shape ``(1, height, width)``.
                    ``grid_session_to_base_yx`` has shape ``(height, width, 2)``;
                    it describes shared session motion in base-scene coordinates.
                    Independent ROI crop deformation is applied before this grid
                    and is not included in it.
                    ``fov_valid`` marks coordinates inside the base FOV before blur.

        Raises:
            ValueError:
                A sampled template is empty, negative, or contains nonfinite values.
        """
        if (
            not isinstance(index_scene, (int, np.integer))
            or not 0 <= index_scene < self.n_scenes
        ):
            raise IndexError("Scene index is outside the dataset.")
        seeds_scene = np.random.SeedSequence(
            entropy=[self.seed, int(index_scene)]
        ).spawn(n_children=self.config.n_sessions + 1)
        scene_base = self._render_base_scene(
            index_scene=int(index_scene), rng=np.random.default_rng(seed=seeds_scene[0])
        )

        ## Keep observation masks in base order even though session rows are shuffled.
        sessions = []
        masks_observed = []
        for seed_session in seeds_scene[1:]:
            session, is_observed = self._render_session(
                scene_base=scene_base, seed_session=seed_session
            )
            sessions.append(session)
            masks_observed.append(is_observed)
        return {
            "scene_id": int(index_scene),
            "um_per_pixel": self.config.um_per_pixel,
            "roi_ids_base": scene_base["roi_ids"],
            "roi_library_indices_base": scene_base["roi_library_indices"],
            "centroids_base_yx": _calculate_centroids(
                roi_images=scene_base["roi_images"]
            ),
            "is_observed": torch.stack(tensors=masks_observed, dim=0),
            "sessions": sessions,
        }


def collate_synthetic_sessions(scenes: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    """Pad each session's observed ROIs to a common size for batching.

    Args:
        scenes (list[dict[str, Any]]):
            Dataset items with the same FOV shape, session count, and base-ROI count.

    Returns:
        (dict[str, torch.Tensor]):
            roi_images, roi_images_clean (torch.Tensor):
                Footprints with shape ``(batch, sessions, max_rois, height, width)``.
            roi_ids, roi_library_indices (torch.Tensor):
                Identities and source rows with shape ``(batch, sessions, max_rois)``.
            roi_valid (torch.Tensor):
                Boolean observed-ROI mask with shape ``(batch, sessions, max_rois)``.
            centroids_yx (torch.Tensor):
                Positions in pixels with shape ``(batch, sessions, max_rois, 2)``.
            fov_image, fov_valid (torch.Tensor):
                FOVs and valid-pixel masks with shape ``(batch, sessions, 1, height, width)``.
            grid_session_to_base_yx (torch.Tensor):
                Shared-motion sampling grids with shape
                ``(batch, sessions, height, width, 2)``. Independent crop
                deformation is applied before these grids.
            scene_ids, um_per_pixel (torch.Tensor):
                Scene indices and pixel sizes with shape ``(batch,)``.
            roi_ids_base, roi_library_indices_base (torch.Tensor):
                Base identities and source rows with shape ``(batch, n_rois)``.
            centroids_base_yx (torch.Tensor):
                Base positions with shape ``(batch, n_rois, 2)``.
            is_observed (torch.Tensor):
                Observations in base order with shape ``(batch, sessions, n_rois)``.

    Padded images and positions are zero; padded IDs and source indices are -1.
    Use ``roi_valid`` to mask attention and losses. Sessions with no observed ROIs
    are supported. Multiply pixel coordinates by ``um_per_pixel`` to get microns.
    """
    ## Verify dimensions before allocating the padded batch.
    if not scenes:
        raise ValueError("Cannot collate an empty batch.")
    n_sessions = len(scenes[0]["sessions"])
    shape_fov = scenes[0]["sessions"][0]["fov_image"].shape
    shape_base = scenes[0]["roi_ids_base"].shape
    for scene in scenes:
        if (
            len(scene["sessions"]) != n_sessions
            or scene["roi_ids_base"].shape != shape_base
            or any(
                session["fov_image"].shape != shape_fov for session in scene["sessions"]
            )
        ):
            raise ValueError(
                "Batch scenes must share session count, base-ROI count, and FOV shape."
            )
    n_rois_max = max(
        len(session["roi_ids"]) for scene in scenes for session in scene["sessions"]
    )
    shape_roi_slots = (len(scenes), n_sessions, n_rois_max)

    ## Allocate zero images and masks, with -1 for padded identities.
    batch = {
        "roi_images": torch.zeros(
            size=(*shape_roi_slots, *shape_fov[-2:]), dtype=torch.float32
        ),
        "roi_images_clean": torch.zeros(
            size=(*shape_roi_slots, *shape_fov[-2:]), dtype=torch.float32
        ),
        "roi_ids": torch.full(size=shape_roi_slots, fill_value=-1, dtype=torch.int64),
        "roi_library_indices": torch.full(
            size=shape_roi_slots, fill_value=-1, dtype=torch.int64
        ),
        "centroids_yx": torch.zeros(size=(*shape_roi_slots, 2), dtype=torch.float32),
        "roi_valid": torch.zeros(size=shape_roi_slots, dtype=torch.bool),
        "scene_ids": torch.as_tensor(
            data=[scene["scene_id"] for scene in scenes], dtype=torch.int64
        ),
        "um_per_pixel": torch.as_tensor(
            data=[scene["um_per_pixel"] for scene in scenes], dtype=torch.float32
        ),
    }
    for key in (
        "roi_ids_base",
        "roi_library_indices_base",
        "centroids_base_yx",
        "is_observed",
    ):
        batch[key] = torch.stack(tensors=[scene[key] for scene in scenes], dim=0)

    ## FOVs and geometry already have a common shape across sessions.
    for key in ("fov_image", "fov_valid", "grid_session_to_base_yx"):
        batch[key] = torch.stack(
            tensors=[
                torch.stack(
                    tensors=[session[key] for session in scene["sessions"]], dim=0
                )
                for scene in scenes
            ],
            dim=0,
        )

    ## Copy observed rows; the remaining slots retain their padding values.
    for index_scene, scene in enumerate(scenes):
        for index_session, session in enumerate(scene["sessions"]):
            n_observed = len(session["roi_ids"])
            batch["roi_valid"][index_scene, index_session, :n_observed] = True
            for key in (
                "roi_images",
                "roi_images_clean",
                "roi_ids",
                "roi_library_indices",
                "centroids_yx",
            ):
                batch[key][index_scene, index_session, :n_observed] = session[key]
    return batch
