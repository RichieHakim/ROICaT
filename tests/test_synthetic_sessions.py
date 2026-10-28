"""Coordinate, correspondence, missing-observation, and worker checks for synthetic sessions."""

import dataclasses
import json

import numpy as np
import pytest
import scipy.ndimage
import scipy.sparse
import torch

from roicat import helpers
from roicat.model_training import augmentation, synthetic_sessions


@pytest.fixture
def dir_library(tmp_path):
    """Prepare distinct nonempty asymmetric ROI templates without external data."""
    crops = np.zeros(shape=(24, 7, 7), dtype=np.float32)
    for index_roi in range(len(crops)):
        crops[index_roi, 2, 2] = 1
        crops[index_roi, 3, 3] = 0.3 + 0.02 * index_roi
        crops[index_roi, 4, 2 + index_roi % 3] = 0.5
    filepath_npz = tmp_path / "rois.npz"
    scipy.sparse.save_npz(
        file=filepath_npz, matrix=scipy.sparse.csr_array(crops.reshape(len(crops), -1))
    )
    return synthetic_sessions.prepare_sparse_roi_library(
        filepath_npz=str(filepath_npz), dir_library=str(tmp_path / "library")
    )


@pytest.fixture
def config_identity():
    """Disable all perturbations to expose placement and identity bookkeeping."""
    return synthetic_sessions.SyntheticSessionConfig(
        um_per_pixel=1.25,
        n_rois=6,
        shape_roi=(7, 7),
        shape_fov=(24, 31),
        max_translation_px=0,
        max_rotation_deg=0,
        scale_bounds=(1, 1),
        elastic_std_px=0,
        roi_warp_std_px=0,
        blur_sigma_bounds=(0, 0),
        gain_bounds=(1, 1),
        fov_noise_std=0,
        roi_noise_std=0,
        dropout_probability=0,
    )


def _assert_batches_equal(batch_first, batch_second):
    """Compare complete batches, including IDs, noise, masks, and sampling grids."""
    assert batch_first.keys() == batch_second.keys()
    for key in batch_first:
        torch.testing.assert_close(
            actual=batch_first[key], expected=batch_second[key], rtol=0, atol=0
        )


def _reference_warp(image, grid_yx):
    """Independent bilinear sampling grid using SciPy's pixel-coordinate convention."""
    return scipy.ndimage.map_coordinates(
        input=image,
        coordinates=np.moveaxis(grid_yx, -1, 0),
        order=1,
        mode="grid-constant",
        cval=0,
        prefilter=False,
    )


def test_cache_reuse_and_source_mismatch(dir_library, tmp_path):
    """Reuse the same cache without accepting a different archive silently."""
    filepath_npz = tmp_path / "rois.npz"
    filepath_data = tmp_path / "library" / "data.npy"
    timestamp = filepath_data.stat().st_mtime_ns
    assert (
        synthetic_sessions.prepare_sparse_roi_library(
            filepath_npz=str(filepath_npz), dir_library=dir_library
        )
        == dir_library
    )
    assert filepath_data.stat().st_mtime_ns == timestamp
    manifest = json.loads((tmp_path / "library" / "source.json").read_text())
    assert manifest["filepath_npz"] == str(filepath_npz)
    filepath_other = tmp_path / "different.npz"
    filepath_other.write_bytes(filepath_npz.read_bytes())
    with pytest.raises(ValueError, match="does not match"):
        synthetic_sessions.prepare_sparse_roi_library(
            filepath_npz=str(filepath_other), dir_library=dir_library
        )


def test_non_csr_cache_is_not_completed(tmp_path):
    """A failed preparation does not leave a cache that workers could reuse."""
    filepath_npz = tmp_path / "csc.npz"
    scipy.sparse.save_npz(file=filepath_npz, matrix=scipy.sparse.csc_array(np.eye(4)))
    with pytest.raises(ValueError, match="CSR"):
        synthetic_sessions.prepare_sparse_roi_library(
            filepath_npz=str(filepath_npz), dir_library=str(tmp_path / "cache")
        )
    assert not (tmp_path / "cache").exists()


def test_identity_sessions_preserve_footprints_and_correspondences(
    dir_library, config_identity
):
    """Session row permutations retain IDs, source templates, and native coordinates."""
    dataset = synthetic_sessions.SyntheticSessionDataset(
        dir_roi_library=dir_library, config=config_identity, seed=11
    )
    scene = dataset[0]
    session_source, session_target = scene["sessions"]
    assert scene["is_observed"].all()
    assert (
        len(torch.unique(input=scene["roi_library_indices_base"]))
        == config_identity.n_rois
    )
    assert not torch.equal(
        input=session_source["roi_ids"], other=session_target["roi_ids"]
    )
    for index_source, roi_id in enumerate(session_source["roi_ids"]):
        index_target = int(
            torch.where(condition=session_target["roi_ids"] == roi_id)[0][0]
        )
        torch.testing.assert_close(
            actual=session_source["roi_images"][index_source],
            expected=session_target["roi_images"][index_target],
            rtol=0,
            atol=2e-6,
        )
        torch.testing.assert_close(
            actual=session_source["centroids_yx"][index_source],
            expected=scene["centroids_base_yx"][int(roi_id)],
            rtol=0,
            atol=2e-5,
        )
        assert (
            session_source["roi_library_indices"][index_source]
            == session_target["roi_library_indices"][index_target]
        )
    assert session_source["fov_valid"].all()
    ## At least two cells occupy different positions within the shared tile.
    assert torch.pdist(input=scene["centroids_base_yx"]).max() > 5


def test_shared_warp_matches_independent_scipy_reference(dir_library, config_identity):
    """Both the FOV and each cell follow the supplied shared sampling grid, including elastic motion."""
    config_warped = dataclasses.replace(
        config_identity,
        max_translation_px=2.5,
        max_rotation_deg=7,
        elastic_std_px=0.4,
        min_visible_fraction=0.01,
    )
    base = synthetic_sessions.SyntheticSessionDataset(
        dir_roi_library=dir_library, config=config_identity, seed=5
    )[0]
    warped = synthetic_sessions.SyntheticSessionDataset(
        dir_roi_library=dir_library, config=config_warped, seed=5
    )[0]
    session_base = base["sessions"][0]
    for session in warped["sessions"]:
        grid = session["grid_session_to_base_yx"].numpy()
        expected_fov = _reference_warp(
            image=session_base["fov_image"][0].numpy(), grid_yx=grid
        )
        np.testing.assert_allclose(
            actual=session["fov_image"][0].numpy(),
            desired=expected_fov,
            rtol=1e-4,
            atol=4e-6,
        )
        for index_roi, roi_id in enumerate(session["roi_ids"]):
            index_base = int(
                torch.where(condition=session_base["roi_ids"] == roi_id)[0][0]
            )
            expected_roi = _reference_warp(
                image=session_base["roi_images"][index_base].numpy(), grid_yx=grid
            )
            np.testing.assert_allclose(
                actual=session["roi_images"][index_roi].numpy(),
                desired=expected_roi,
                rtol=1e-4,
                atol=4e-6,
            )


def test_subpixel_translation_and_axis_order():
    """A half-pixel sampling grid in x splits an impulse horizontally, not vertically."""
    image = torch.zeros(size=(1, 5, 7), dtype=torch.float32)
    image[0, 2, 3] = 1
    coords_y, coords_x = torch.meshgrid(torch.arange(5), torch.arange(7), indexing="ij")
    grid_yx = torch.stack(tensors=(coords_y, coords_x + 0.5), dim=-1).float()
    result = synthetic_sessions._warp_images(images=image, grid_yx=grid_yx)
    expected = torch.zeros_like(input=image)
    expected[0, 2, 2:4] = 0.5
    torch.testing.assert_close(actual=result, expected=expected, rtol=0, atol=1e-6)


def test_shared_blur_and_gain(dir_library, config_identity):
    """Shared appearance effects transform each footprint and the FOV consistently."""
    base = synthetic_sessions.SyntheticSessionDataset(
        dir_roi_library=dir_library, config=config_identity, seed=3
    )[0]
    transformed = synthetic_sessions.SyntheticSessionDataset(
        dir_roi_library=dir_library,
        config=dataclasses.replace(
            config_identity, blur_sigma_bounds=(0.7, 0.7), gain_bounds=(0.8, 0.8)
        ),
        seed=3,
    )[0]
    for session_base, session_transformed in zip(
        base["sessions"], transformed["sessions"]
    ):
        for key in ("fov_image", "roi_images"):
            expected = (
                scipy.ndimage.gaussian_filter(
                    input=session_base[key].numpy(),
                    sigma=(0, 0.7, 0.7),
                    mode="constant",
                )
                * 0.8
            )
            np.testing.assert_allclose(
                actual=session_transformed[key].numpy(),
                desired=expected,
                atol=1e-6,
                rtol=1e-5,
            )


def test_dropout_removes_extractions_but_preserves_fov(dir_library, config_identity):
    """A neuron remains visually present when its extraction is omitted."""
    visible = synthetic_sessions.SyntheticSessionDataset(
        dir_roi_library=dir_library, config=config_identity, seed=7
    )[0]
    missing = synthetic_sessions.SyntheticSessionDataset(
        dir_roi_library=dir_library,
        config=dataclasses.replace(config_identity, dropout_probability=1),
        seed=7,
    )[0]
    assert not missing["is_observed"].any()
    for session_visible, session_missing in zip(
        visible["sessions"], missing["sessions"]
    ):
        assert session_missing["roi_images"].shape == (0, *config_identity.shape_fov)
        assert session_missing["centroids_yx"].shape == (0, 2)
        torch.testing.assert_close(
            actual=session_missing["fov_image"],
            expected=session_visible["fov_image"],
            rtol=0,
            atol=0,
        )
    batch = synthetic_sessions.collate_synthetic_sessions(scenes=[missing])
    assert batch["roi_images"].shape == (1, 2, 0, *config_identity.shape_fov)
    assert not batch["roi_valid"].any()


def test_padding_and_true_pairs(dir_library, config_identity):
    """Padding never creates false identity matches or nonzero model inputs."""
    config = dataclasses.replace(config_identity, dropout_probability=0.45)
    dataset = synthetic_sessions.SyntheticSessionDataset(
        dir_roi_library=dir_library, config=config, seed=4
    )
    scenes = [dataset[0], dataset[1]]
    batch = synthetic_sessions.collate_synthetic_sessions(scenes=scenes)
    assert batch["roi_images"].shape[:2] == (2, 2)
    assert batch["fov_image"].shape == (2, 2, 1, *config.shape_fov)
    assert batch["grid_session_to_base_yx"].shape == (2, 2, *config.shape_fov, 2)
    assert (~batch["roi_valid"]).any()
    assert (batch["roi_ids"][~batch["roi_valid"]] == -1).all()
    assert (batch["roi_images"][~batch["roi_valid"]] == 0).all()
    assert (batch["centroids_yx"][~batch["roi_valid"]] == 0).all()
    for index_scene, scene in enumerate(scenes):
        for index_session, session in enumerate(scene["sessions"]):
            mask = batch["roi_valid"][index_scene, index_session]
            torch.testing.assert_close(
                actual=batch["roi_ids"][index_scene, index_session][mask],
                expected=session["roi_ids"],
            )
            for roi_id in session["roi_ids"]:
                assert scene["is_observed"][
                    index_session, int(roi_id) - index_scene * config.n_rois
                ]
    valid_ids_first = batch["roi_ids"][0][batch["roi_valid"][0]]
    valid_ids_second = batch["roi_ids"][1][batch["roi_valid"][1]]
    assert not torch.isin(
        elements=valid_ids_first, test_elements=valid_ids_second
    ).any()


def test_clean_targets_exclude_independent_roi_noise(dir_library, config_identity):
    """Observation noise changes the input footprints but not clean geometry targets."""
    clean = synthetic_sessions.SyntheticSessionDataset(
        dir_roi_library=dir_library, config=config_identity, seed=2
    )[0]
    noisy = synthetic_sessions.SyntheticSessionDataset(
        dir_roi_library=dir_library,
        config=dataclasses.replace(config_identity, roi_noise_std=0.4),
        seed=2,
    )[0]
    for session_clean, session_noisy in zip(clean["sessions"], noisy["sessions"]):
        torch.testing.assert_close(
            actual=session_noisy["roi_images_clean"],
            expected=session_clean["roi_images"],
            rtol=0,
            atol=0,
        )
        assert not torch.equal(
            input=session_noisy["roi_images"], other=session_noisy["roi_images_clean"]
        )
        assert (session_noisy["roi_images"] >= 0).all()


def test_determinism_across_access_order_and_spawned_worker(
    dir_library, config_identity
):
    """Noise and source selection depend on the scene index, not RNG/worker scheduling."""
    config = dataclasses.replace(
        config_identity,
        elastic_std_px=0.3,
        roi_warp_std_px=1.0,
        roi_noise_std=0.2,
        fov_noise_std=0.1,
        dropout_probability=0.2,
    )
    dataset = synthetic_sessions.SyntheticSessionDataset(
        dir_roi_library=dir_library, config=config, n_scenes=4, seed=9
    )
    reference = synthetic_sessions.collate_synthetic_sessions(
        scenes=[dataset[0], dataset[1]]
    )
    dataset[3]
    _assert_batches_equal(
        batch_first=reference,
        batch_second=synthetic_sessions.collate_synthetic_sessions(
            scenes=[dataset[0], dataset[1]]
        ),
    )
    ## An already-open library must be reopened in a spawned worker.
    assert dataset.__getstate__()["_roi_library"] is None
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=2,
        num_workers=1,
        multiprocessing_context="spawn",
        collate_fn=synthetic_sessions.collate_synthetic_sessions,
    )
    batches = list(loader)
    _assert_batches_equal(batch_first=reference, batch_second=batches[0])
    different = synthetic_sessions.SyntheticSessionDataset(
        dir_roi_library=dir_library, config=config, seed=10
    )[0]
    assert not torch.equal(
        input=dataset[0]["roi_library_indices_base"],
        other=different["roi_library_indices_base"],
    )


def test_disjoint_template_subsets(dir_library, config_identity):
    """Training and validation subsets cannot sample each other's source templates."""
    train = synthetic_sessions.SyntheticSessionDataset(
        dir_roi_library=dir_library,
        config=config_identity,
        indices_roi_library=np.arange(12),
    )
    validation = synthetic_sessions.SyntheticSessionDataset(
        dir_roi_library=dir_library,
        config=config_identity,
        indices_roi_library=np.arange(12, 24),
    )
    assert (train[0]["roi_library_indices_base"] < 12).all()
    assert (validation[0]["roi_library_indices_base"] >= 12).all()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dropout_probability": -0.1},
        {"um_per_pixel": 0},
        {"n_sessions": 1},
        {"gain_bounds": (1, 0.5)},
        {"shape_roi": (25, 7)},
    ],
)
def test_invalid_scene_config(config_identity, kwargs):
    """Reject invalid physical scales, probabilities, bounds, and image sizes early."""
    with pytest.raises(ValueError):
        dataclasses.replace(config_identity, **kwargs)


def test_invalid_source_templates(tmp_path, config_identity):
    """An empty source ROI cannot silently become an unmatched synthetic cell."""
    filepath_npz = tmp_path / "empty.npz"
    scipy.sparse.save_npz(
        file=filepath_npz, matrix=scipy.sparse.csr_array((6, 49), dtype=np.float32)
    )
    dir_library = synthetic_sessions.prepare_sparse_roi_library(
        filepath_npz=str(filepath_npz), dir_library=str(tmp_path / "empty_library")
    )
    dataset = synthetic_sessions.SyntheticSessionDataset(
        dir_roi_library=dir_library, config=config_identity
    )
    with pytest.raises(ValueError, match="nonempty"):
        dataset[0]


def test_default_fov_has_no_added_background(dir_library, config_identity):
    """Empty pixels stay zero when FOV noise and blur are disabled."""
    assert synthetic_sessions.SyntheticSessionConfig(um_per_pixel=1).fov_noise_std == 0
    dataset = synthetic_sessions.SyntheticSessionDataset(
        dir_roi_library=dir_library, config=config_identity, seed=3
    )
    scene = dataset[0]
    for session in scene["sessions"]:
        mask_empty = session["roi_images"].sum(dim=0) == 0
        assert mask_empty.any()
        assert (session["fov_image"][0][mask_empty] == 0).all()


def test_dataset_preserves_global_torch_generator(dir_library, config_identity):
    """Local augmentation generators keep data loading from changing model randomness."""
    dataset = synthetic_sessions.SyntheticSessionDataset(
        dir_roi_library=dir_library,
        config=dataclasses.replace(
            config_identity,
            elastic_std_px=0.4,
            roi_warp_std_px=1.0,
            roi_noise_std=0.1,
            fov_noise_std=0.1,
        ),
    )
    state_before = torch.get_rng_state()
    dataset[0]
    assert torch.equal(input=torch.get_rng_state(), other=state_before)


@pytest.mark.parametrize("shape_image", [(9, 15), (15, 9)])
def test_warp_points_rectangular_identity(shape_image):
    """Rectangular images retain height, width, and their coordinate orientation."""
    transform = augmentation.WarpPoints(
        prob=0.0,
        img_size_in=list(shape_image),
        img_size_out=list(shape_image),
        mode="bilinear",
    )
    image = torch.arange(end=np.prod(shape_image), dtype=torch.float32).reshape(
        1, *shape_image
    )
    image_warped = transform(tensor=image, generator=torch.Generator().manual_seed(2))
    assert image_warped.shape == image.shape
    torch.testing.assert_close(actual=image_warped, expected=image, rtol=0, atol=2e-5)


@pytest.mark.parametrize("shape_output", [(36, 36), (48, 48)])
def test_warp_points_preserves_existing_square_transform(shape_output):
    """Compare the refactored grid against the previous square-image implementation."""
    transform = augmentation.WarpPoints(
        n_warps=3, prob=1.0, img_size_out=list(shape_output), mode="bilinear"
    )
    image = torch.arange(end=36 * 36, dtype=torch.float32).reshape(1, 36, 36) / (
        36 * 36
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(6)
        torch.rand(1)  ## Previous probability draw.
        parameters = torch.rand(5, transform.n_warps)
        centers_x = parameters[0] * transform.cx_range + transform.cx[0]
        centers_y = parameters[1] * transform.cy_range + transform.cy[0]
        deltas_x = parameters[2] * transform.dx_range + transform.dx[0]
        deltas_y = parameters[3] * transform.dy_range + transform.dy[0]
        radii = parameters[4] * transform.r_range + transform.r[0]
        weights = transform.gaus2D(x=centers_x, y=centers_y, sigma=radii)
        displacement = weights[None, ...] * torch.stack(
            tensors=(deltas_x, deltas_y), dim=0
        ).reshape(2, 1, 1, transform.n_warps)
        grid_legacy = (
            transform.meshgrid_out[..., 0:1] + displacement.sum(dim=3, keepdim=True)
        ).permute(3, 2, 1, 0)
        image_expected = torch.nn.functional.grid_sample(
            input=image[None, ...],
            grid=grid_legacy,
            mode="bilinear",
            align_corners=True,
        )[0]
        torch.manual_seed(6)
        image_actual = transform(tensor=image)
    torch.testing.assert_close(
        actual=image_actual, expected=image_expected, rtol=0, atol=2e-6
    )


def test_augmentations_support_local_generators_and_torchscript():
    """The existing ROInet pipeline can still script the updated augmentations."""
    transform_warp = augmentation.WarpPoints(
        prob=1.0, img_size_in=[9, 15], img_size_out=[9, 15], mode="bilinear"
    )
    transform_noise = augmentation.AddGaussianNoise(std=0.2, prob=1.0)
    pipeline_scripted = torch.jit.script(
        obj=torch.nn.Sequential(transform_warp, transform_noise)
    )
    image = torch.ones(size=(1, 9, 15))
    assert pipeline_scripted(image).shape == image.shape

    ## Local generators must reproduce both transforms without changing global state.
    state_before = torch.get_rng_state()
    outputs = []
    for _ in range(2):
        generator = torch.Generator().manual_seed(7)
        image_warped = transform_warp(tensor=image, generator=generator)
        outputs.append(transform_noise(tensor=image_warped, generator=generator))
    torch.testing.assert_close(actual=outputs[0], expected=outputs[1], rtol=0, atol=0)
    assert torch.equal(input=torch.get_rng_state(), other=state_before)


def test_independent_roi_warps_change_shapes_and_preserve_fov(
    dir_library, config_identity
):
    """ROI-specific deformation changes footprints while their FOV signal stays aligned."""
    config_warped = dataclasses.replace(
        config_identity, roi_warp_std_px=0.6, min_visible_fraction=0.001
    )
    seed = 31
    scene_unwarped = synthetic_sessions.SyntheticSessionDataset(
        dir_roi_library=dir_library, config=config_identity, seed=seed
    )[0]
    dataset_warped = synthetic_sessions.SyntheticSessionDataset(
        dir_roi_library=dir_library, config=config_warped, seed=seed
    )
    scene_warped = dataset_warped[0]
    assert scene_warped["is_observed"].all()
    seed_base = np.random.SeedSequence(entropy=[seed, 0]).spawn(n_children=3)[0]
    scene_base = dataset_warped._render_base_scene(
        index_scene=0, rng=np.random.default_rng(seed=seed_base)
    )
    for index_session, session in enumerate(scene_warped["sessions"]):
        indices_roi = session["roi_ids"]
        fov_expected = (
            session["roi_images_clean"]
            * scene_base["intensities_roi"][indices_roi, None, None]
        ).sum(dim=0)[None, ...] / scene_base["fov_peak"]
        torch.testing.assert_close(
            actual=session["fov_image"], expected=fov_expected, rtol=1e-5, atol=1e-6
        )
        session_unwarped = scene_unwarped["sessions"][index_session]
        for index_roi, roi_id in enumerate(indices_roi):
            index_unwarped = int(
                torch.where(condition=session_unwarped["roi_ids"] == roi_id)[0][0]
            )
            assert not torch.allclose(
                input=session["roi_images"][index_roi],
                other=session_unwarped["roi_images"][index_unwarped],
            )
        torch.testing.assert_close(
            actual=session["grid_session_to_base_yx"],
            expected=session_unwarped["grid_session_to_base_yx"],
            rtol=0,
            atol=0,
        )
    session_first, session_second = scene_warped["sessions"]
    indices_first = torch.argsort(input=session_first["roi_ids"])
    indices_second = torch.argsort(input=session_second["roi_ids"])
    assert not torch.allclose(
        input=session_first["roi_images"][indices_first],
        other=session_second["roi_images"][indices_second],
    )


def test_random_colormap_accepts_local_generator():
    """Seeded colors reproduce without advancing NumPy's global generator."""
    state_before = np.random.get_state()
    colormaps = [
        helpers.rand_cmap(nlabels=128, verbose=False, rng=np.random.default_rng(seed=4))
        for _ in range(2)
    ]
    state_after = np.random.get_state()
    np.testing.assert_array_equal(actual=state_after[1], desired=state_before[1])
    assert state_after[0] == state_before[0]
    assert state_after[2:] == state_before[2:]
    np.testing.assert_array_equal(
        actual=colormaps[0](np.linspace(start=0, stop=1, num=128)),
        desired=colormaps[1](np.linspace(start=0, stop=1, num=128)),
    )
