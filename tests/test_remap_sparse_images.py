"""
Unit tests for the sparse ROI warp added for ROICaT issue #686:
``helpers.remap_sparse_images`` and
``Aligner.transform_ROIs(method_warp=...)``.

Two references are written here from the documented contract, independently of
the implementation: \n
    * R1: ``_reference_loop``, a float64 python loop over output pixels.
    * R2: ``_reference_map_coordinates``, a wrapper around
      ``scipy.ndimage.map_coordinates(mode='grid-constant', cval=0)``. \n
``'grid-constant'`` is the scipy padding mode that interpolates across the frame
edge the way the function does; plain ``'constant'`` is not.

Exact equality is demanded on dyadic fixtures: integer pixel values below 1000
and field coordinates on multiples of 1/8. Every weight, product and partial sum
is then exactly representable in float32 and float64, so the order the products
are accumulated in cannot change the answer.

These tests need no downloaded test data and run on CPU.

To run the tests, use the command (in a terminal):
    pytest -v test_remap_sparse_images.py
"""

from typing import Tuple

import copy

import numpy as np
import pytest
import scipy.ndimage
import scipy.sparse

from roicat import helpers, util
from roicat.tracking import alignment


######################################################################################################################################
########################################################## REFERENCES ################################################################
######################################################################################################################################


def _reference_loop(im: np.ndarray, remappingIdx: np.ndarray, method: str) -> np.ndarray:
    """
    R1. Warps one image with a plain float64 loop over output pixels. Each tap is
    bounds-checked on its own: a tap outside the frame contributes 0 and the
    survivors are not renormalized. A non-finite coordinate fails every bounds
    check, so that output pixel comes out 0.

    Args:
        im (np.ndarray):
            The image to warp. Shape: *(H, W)*.
        remappingIdx (np.ndarray):
            The remapping field. Shape: *(H, W, 2)*, last dim is *(x, y)*.
        method (str):
            ``'nearest'`` or ``'linear'``.

    Returns:
        (np.ndarray):
            im_warped (np.ndarray):
                The warped image, float64. Shape: *(H, W)*.
    """
    im = np.asarray(im, dtype=np.float64)
    H, W = im.shape
    out = np.zeros((H, W), dtype=np.float64)
    ## An infinite coordinate makes inf - inf = NaN; those taps fail the bounds check.
    with np.errstate(invalid='ignore'):
        for r_out in range(H):
            for c_out in range(W):
                x, y = np.float64(remappingIdx[r_out, c_out, 0]), np.float64(remappingIdx[r_out, c_out, 1])
                if method == 'nearest':
                    taps = [(np.floor(y + 0.5), np.floor(x + 0.5), 1.0)]
                elif method == 'linear':
                    x0, y0 = np.floor(x), np.floor(y)
                    fx, fy = x - x0, y - y0
                    taps = [
                        (y0,     x0,     (1.0 - fy) * (1.0 - fx)),
                        (y0,     x0 + 1, (1.0 - fy) * fx),
                        (y0 + 1, x0,     fy * (1.0 - fx)),
                        (y0 + 1, x0 + 1, fy * fx),
                    ]
                for (row, col, weight) in taps:
                    if (0 <= row) and (row < H) and (0 <= col) and (col < W):
                        out[r_out, c_out] += im[int(row), int(col)] * weight
    return out


def _reference_map_coordinates(im: np.ndarray, remappingIdx: np.ndarray, method: str) -> np.ndarray:
    """
    R2. Warps one image with ``scipy.ndimage.map_coordinates``. It returns NaN
    where a coordinate is non-finite, while the function under test returns 0
    there by design, so those output pixels are masked to 0.
    Same arguments and return as ``_reference_loop``.
    """
    ## map_coordinates indexes (row, col); remappingIdx stores (x, y) = (col, row)
    coords = np.stack([remappingIdx[..., 1], remappingIdx[..., 0]], axis=0).astype(np.float64)  ## (2, H, W)
    out = scipy.ndimage.map_coordinates(
        input=np.asarray(im, dtype=np.float64),
        coordinates=coords,
        order={'nearest': 0, 'linear': 1}[method],
        mode='grid-constant',
        cval=0.0,
    )
    out[~np.isfinite(coords).all(axis=0)] = 0.0
    return out


def _warp_reference(ims: np.ndarray, remappingIdx: np.ndarray, method: str, reference: str = 'map_coordinates') -> np.ndarray:
    """
    Applies a reference (``'loop'`` for R1, ``'map_coordinates'`` for R2) to a
    stack of flattened images. Shape in and out: *(n_images, H*W)*.
    """
    shape_frame = remappingIdx.shape[:2]
    fn = {'loop': _reference_loop, 'map_coordinates': _reference_map_coordinates}[reference]
    out = [fn(im=im.reshape(shape_frame), remappingIdx=remappingIdx, method=method).reshape(-1) for im in np.asarray(ims, dtype=np.float64)]
    return np.stack(out, axis=0)


def _warp_matmul(ims: np.ndarray, remappingIdx: np.ndarray, method: str, dtype: np.dtype = np.float64, **kwargs) -> np.ndarray:
    """Runs the function under test on dense flattened images and densifies the output to float64."""
    out = helpers.remap_sparse_images(
        ims_sparse_flat=scipy.sparse.csr_array(np.asarray(ims)),
        remappingIdx=remappingIdx,
        method=method,
        dtype=dtype,
        **kwargs,
    )
    _assert_canonical(out=out, shape=np.asarray(ims).shape, dtype=dtype)
    return out.toarray().astype(np.float64)


def _assert_canonical(out, shape: Tuple[int, int], dtype: np.dtype):
    """Asserts the output contract: ``csr_array``, right shape and dtype, strictly ascending indices per row, no stored zeros."""
    assert isinstance(out, scipy.sparse.csr_array), f"expected csr_array, got {type(out)}"
    assert out.shape == tuple(shape)
    assert out.dtype == np.dtype(dtype)
    assert np.all(out.data != 0), 'explicit zeros are stored'
    for ii in range(out.shape[0]):
        ## strictly ascending means sorted and free of duplicates
        assert np.all(np.diff(out.indices[out.indptr[ii]:out.indptr[ii + 1]]) > 0), f"row {ii} has unsorted or duplicated indices"


def _assert_csr_identical(a, b, msg: str = ''):
    """Asserts that two CSR arrays are identical down to their stored arrays."""
    assert (a.shape == b.shape) and (a.dtype == b.dtype), msg
    for name in ['indptr', 'indices', 'data']:
        assert np.array_equal(getattr(a, name), getattr(b, name)), f"{name} differ. {msg}"


######################################################################################################################################
########################################################## FIXTURES ##################################################################
######################################################################################################################################


METHODS = ['nearest', 'linear']
SHAPES_FRAME = [(12, 9), (9, 12), (1, 7), (7, 1), (5, 5), (2, 2)]


def _dyadic(a: np.ndarray) -> np.ndarray:
    """Rounds coordinates onto multiples of 1/8."""
    return np.round(np.asarray(a, dtype=np.float64) * 8) / 8


def _field_translate(shape_frame: Tuple[int, int], shift_x: float = 0.0, shift_y: float = 0.0) -> np.ndarray:
    """Moves the image content by ``(+shift_x, +shift_y)``. Zero shifts give the identity field."""
    yy, xx = np.meshgrid(np.arange(shape_frame[0], dtype=np.float64), np.arange(shape_frame[1], dtype=np.float64), indexing='ij')
    return np.stack([xx - shift_x, yy - shift_y], axis=-1)


def _field_smooth(shape_frame: Tuple[int, int], rng: np.random.Generator, amplitude: float = 3.0) -> np.ndarray:
    """Identity plus a dyadic random displacement."""
    return _dyadic(_field_translate(shape_frame) + rng.uniform(-amplitude, amplitude, size=(*shape_frame, 2)))


def _field_folded(shape_frame: Tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    """A non-monotone field: several output pixels sample the same source pixel."""
    H, W = shape_frame
    field = _field_translate(shape_frame)
    field[..., 0] += 15.0 * np.sin(2 * np.pi * field[..., 1] / H)
    field[..., 1] += 10.0 * np.cos(2 * np.pi * field[..., 0] / W)
    return _dyadic(field)


def _field_constant(shape_frame: Tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    """Every output pixel samples the same source coordinate, so the output rows are dense."""
    return np.broadcast_to(np.array([0.375, 0.25]), (*shape_frame, 2)).copy()


def _field_nonfinite(shape_frame: Tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    """A smooth field with a rectangular patch of NaN and two infinite coordinates."""
    H, W = shape_frame
    field = _field_smooth(shape_frame, rng)
    field[: max(H // 3, 1), : max(W // 3, 1), :] = np.nan
    field[H - 1, W - 1, 0] = np.inf
    field[0, W - 1, 1] = -np.inf
    return field


def _field_offframe(shape_frame: Tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    """A field that carries most of the frame off the edge."""
    H, W = shape_frame
    return _dyadic(_field_smooth(shape_frame, rng) + np.array([W * 0.6, H * 0.6]))


FIELDS = {
    'identity': lambda shape_frame, rng: _field_translate(shape_frame),
    'smooth': _field_smooth,
    'folded': _field_folded,
    'constant': _field_constant,
    'nonfinite': _field_nonfinite,
    'offframe': _field_offframe,
}


def _dyadic_images(shape_frame: Tuple[int, int], rng: np.random.Generator, n_images: int = 4, density: float = 0.25) -> np.ndarray:
    """Flattened images with sparse integer values in [1, 1000). Shape: *(n_images, H*W)*, float64."""
    n_pixels = shape_frame[0] * shape_frame[1]
    return rng.integers(1, 1000, size=(n_images, n_pixels)).astype(np.float64) * (rng.random((n_images, n_pixels)) < density)


def _roi_ring(shape_frame: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """An annulus centred in the frame and the mask of its hole. Both shape *(H, W)*."""
    H, W = shape_frame
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    rr = np.sqrt((yy - (H - 1) / 2) ** 2 + (xx - (W - 1) / 2) ** 2)
    return ((rr >= 5.0) & (rr <= 9.0)).astype(np.float64), rr <= 3.0


######################################################################################################################################
###################################################### PARITY WITH REFERENCES ########################################################
######################################################################################################################################


@pytest.mark.parametrize('method', METHODS)
@pytest.mark.parametrize('shape_frame', SHAPES_FRAME)
@pytest.mark.parametrize('kind_field', list(FIELDS.keys()))
def test_matches_both_references_exactly_on_dyadic_fixtures(shape_frame, kind_field, method):
    """R1, R2 and the function agree to the last bit in float64 and float32, on every field kind and frame shape."""
    rng = np.random.default_rng(0)
    field = FIELDS[kind_field](shape_frame, rng)
    ims = _dyadic_images(shape_frame=shape_frame, rng=rng)
    out_loop = _warp_reference(ims=ims, remappingIdx=field, method=method, reference='loop')
    assert np.array_equal(out_loop, _warp_reference(ims=ims, remappingIdx=field, method=method, reference='map_coordinates'))
    for dtype in [np.float64, np.float32]:
        assert np.array_equal(out_loop, _warp_matmul(ims=ims, remappingIdx=field, method=method, dtype=dtype)), f"dtype={dtype}"


@pytest.mark.parametrize('method', METHODS)
@pytest.mark.parametrize('dtype,rtol', [(np.float64, 1e-12), (np.float32, 1e-6)])
def test_matches_reference_on_real_valued_fixture(dtype, rtol, method):
    """Generic coordinates and pixel values on a larger non-square frame, to a tolerance set by ``dtype``."""
    rng = np.random.default_rng(1)
    shape_frame = (70, 96)
    field = _field_translate(shape_frame) + rng.normal(0, 4, size=(*shape_frame, 2))
    ## Cast first: the function sees the images in ``dtype``, so the reference should too
    ims = (rng.normal(0, 1, size=(5, 70 * 96)) * (rng.random((5, 70 * 96)) < 0.1)).astype(dtype)
    out = _warp_matmul(ims=ims, remappingIdx=field, method=method, dtype=dtype)
    out_reference = _warp_reference(ims=ims, remappingIdx=field, method=method)
    assert np.allclose(out, out_reference, rtol=rtol, atol=rtol)


def test_random_draws():
    """50 seeded draws over frame shape, field kind, method, dtype, density, batch size and sparse format."""
    for seed in range(50):
        rng = np.random.default_rng(1000 + seed)
        shape_frame = (int(rng.integers(1, 40)), int(rng.integers(1, 40)))
        kind_field = str(rng.choice(list(FIELDS.keys())))
        method = str(rng.choice(METHODS))
        dtype = [np.float32, np.float64][int(rng.integers(2))]
        field = FIELDS[kind_field](shape_frame, rng)
        ims = _dyadic_images(shape_frame=shape_frame, rng=rng, n_images=int(rng.integers(0, 6)) + 1, density=float(rng.uniform(0, 0.6)))
        out = _warp_matmul(ims=ims, remappingIdx=field, method=method, dtype=dtype, n_pixels_per_batch=int(rng.integers(1, 200)))
        assert np.array_equal(out, _warp_reference(ims=ims, remappingIdx=field, method=method)), f"seed={seed} {shape_frame} {kind_field} {method} {dtype}"


######################################################################################################################################
######################################################## GEOMETRY AND EDGES ##########################################################
######################################################################################################################################


@pytest.mark.parametrize('method', METHODS)
@pytest.mark.parametrize('shift_x,shift_y', [(0, 0), (3, 0), (0, 2), (-2, 3)])
def test_integer_translation_pins_axis_and_flatten_order(shift_x, shift_y, method):
    """An integer shift moves pixels exactly. Fails if x/y or the C flatten order are mixed up. No reference involved."""
    shape_frame = (9, 12)
    im = np.arange(1, 9 * 12 + 1, dtype=np.float64).reshape(shape_frame)
    out = _warp_matmul(ims=im.reshape(1, -1), remappingIdx=_field_translate(shape_frame, shift_x=shift_x, shift_y=shift_y), method=method).reshape(shape_frame)
    expected = np.zeros(shape_frame)
    rows_out, cols_out = slice(max(shift_y, 0), 9 + min(shift_y, 0)), slice(max(shift_x, 0), 12 + min(shift_x, 0))
    rows_in, cols_in = slice(max(-shift_y, 0), 9 + min(-shift_y, 0)), slice(max(-shift_x, 0), 12 + min(-shift_x, 0))
    expected[rows_out, cols_out] = im[rows_in, cols_in]
    assert np.array_equal(out, expected)


@pytest.mark.parametrize('method', METHODS)
def test_single_pixel_rois_on_every_edge_and_corner(method):
    """One single-pixel ROI per corner and edge midpoint, under a half-pixel shift that pushes taps over each edge."""
    shape_frame = (5, 7)
    rcs_source = [(0, 0), (0, 6), (4, 0), (4, 6), (0, 3), (4, 3), (2, 0), (2, 6)]
    ims = np.zeros((len(rcs_source), 5, 7))
    for ii, (row, col) in enumerate(rcs_source):
        ims[ii, row, col] = 8.0
    ims = ims.reshape(len(rcs_source), -1)
    for (shift_x, shift_y) in [(0.5, -0.5), (-0.5, 0.5)]:
        field = _field_translate(shape_frame, shift_x=shift_x, shift_y=shift_y)
        assert np.array_equal(_warp_matmul(ims=ims, remappingIdx=field, method=method), _warp_reference(ims=ims, remappingIdx=field, method=method, reference='loop'))


def test_edge_taps_are_not_renormalized():
    """Output (0, 0) reads x=-0.5: half its weight falls off the frame, so it gets half the corner pixel, not all of it."""
    shape_frame = (3, 4)
    im = np.zeros(shape_frame)
    im[0, 0] = 8.0
    out = _warp_matmul(ims=im.reshape(1, -1), remappingIdx=_field_translate(shape_frame, shift_x=0.5), method='linear').reshape(shape_frame)
    assert out[0, 0] == 4.0
    assert out[0, 1] == 4.0
    assert out.sum() == 8.0


@pytest.mark.parametrize('method', METHODS)
def test_displacement_partly_and_fully_off_frame(method):
    """A ROI pushed half over the edge loses exactly the part that left; pushed all the way, it comes back empty."""
    shape_frame = (10, 14)
    im = np.zeros(shape_frame)
    im[3:7, 5:9] = np.arange(1, 17).reshape(4, 4)
    for (shift_x, shift_y) in [(7, 0), (-7, 0), (0, 5), (0, -5)]:
        field = _field_translate(shape_frame, shift_x=shift_x, shift_y=shift_y)
        out = _warp_matmul(ims=im.reshape(1, -1), remappingIdx=field, method=method)
        assert 0 < out.sum() < im.sum()
        assert np.array_equal(out, _warp_reference(ims=im.reshape(1, -1), remappingIdx=field, method=method))
    for (shift_x, shift_y) in [(20.25, 0), (-20.25, 0), (0, 15.5), (3, -15.5), (1e12, -1e12)]:
        out = _warp_matmul(ims=im.reshape(1, -1), remappingIdx=_field_translate(shape_frame, shift_x=shift_x, shift_y=shift_y), method=method)
        assert np.count_nonzero(out) == 0


def test_nearest_rounds_half_up():
    """
    ``'nearest'`` samples ``floor(coordinate + 0.5)``, so a coordinate exactly on
    a half picks the higher pixel. Libraries disagree here (torch and numpy round
    half to even), which is why the rule is pinned rather than inherited.
    """
    im = np.arange(1, 7, dtype=np.float64).reshape(1, 6)
    field = np.zeros((1, 6, 2))
    field[0, :, 0] = [-0.5, 0.5, 1.5, 2.5, 4.5, 5.5]
    out = _warp_matmul(ims=im, remappingIdx=field, method='nearest')
    ## -0.5 -> pixel 0, 0.5 -> 1, 1.5 -> 2, 2.5 -> 3, 4.5 -> 5, 5.5 -> 6 which is off the frame
    assert np.array_equal(out, np.array([[1, 2, 3, 4, 6, 0]], dtype=np.float64))
    field[0, :, 0] = -0.5000001
    assert np.count_nonzero(_warp_matmul(ims=im, remappingIdx=field, method='nearest')) == 0


######################################################################################################################################
###################################################### DEGENERATE INPUTS #############################################################
######################################################################################################################################


@pytest.mark.parametrize('method', METHODS)
def test_empty_batch_and_all_zero_image(method):
    """Zero images give a *(0, H*W)* output. An all-zero image gives an all-zero row between two intact ones."""
    rng = np.random.default_rng(2)
    shape_frame = (8, 6)
    field = _field_smooth(shape_frame, rng)
    out_empty = helpers.remap_sparse_images(ims_sparse_flat=scipy.sparse.csr_array((0, 48), dtype=np.float32), remappingIdx=field, method=method)
    _assert_canonical(out=out_empty, shape=(0, 48), dtype=np.float32)

    ims = _dyadic_images(shape_frame=shape_frame, rng=rng, n_images=3)
    ims[1] = 0
    out = _warp_matmul(ims=ims, remappingIdx=field, method=method)
    assert np.count_nonzero(out[1]) == 0
    assert np.array_equal(out, _warp_reference(ims=ims, remappingIdx=field, method=method))


@pytest.mark.parametrize('method', METHODS)
def test_nonfinite_coordinates_give_zeros(method):
    """NaN and infinite coordinates give output pixels of exactly 0 and never a NaN; a field of nothing else gives an empty output."""
    rng = np.random.default_rng(3)
    shape_frame = (9, 12)
    ims = np.ones((2, 9 * 12))
    field = _field_nonfinite(shape_frame, rng)
    out = _warp_matmul(ims=ims, remappingIdx=field, method=method)
    assert np.isfinite(out).all()
    assert np.count_nonzero(out.reshape(2, 9, 12)[:, ~np.isfinite(field).all(axis=-1)]) == 0
    assert np.count_nonzero(out) > 0, 'the finite part of the field should still sample the image'
    for value_bad in [np.nan, np.inf, -np.inf]:
        out_bad = _warp_matmul(ims=ims, remappingIdx=np.full((9, 12, 2), value_bad), method=method)
        assert np.count_nonzero(out_bad) == 0


@pytest.mark.parametrize('method', METHODS)
def test_ring_roi_keeps_its_hole_empty(method):
    """
    The #686 assertion. Each output pixel is a weighted sum of the pixels under
    it, so the middle of an annulus stays empty under a sub-pixel warp.
    """
    shape_frame = (24, 24)
    im, mask_hole = _roi_ring(shape_frame)
    field = _field_translate(shape_frame, shift_x=0.5, shift_y=-0.5)
    out = _warp_matmul(ims=im.reshape(1, -1), remappingIdx=field, method=method).reshape(shape_frame)
    assert im[mask_hole].sum() == 0, 'the fixture must have an empty hole to begin with'
    assert out[mask_hole].sum() == 0, 'the hole of a ring ROI was filled in'
    assert out.sum() > 0, 'the ring itself should survive the warp'


######################################################################################################################################
######################################################## CALL SEMANTICS ##############################################################
######################################################################################################################################


@pytest.mark.parametrize('constructor', [
    scipy.sparse.csr_array, scipy.sparse.csr_matrix,
    scipy.sparse.csc_array, scipy.sparse.csc_matrix,
    scipy.sparse.coo_array, scipy.sparse.coo_matrix,
    scipy.sparse.lil_array, scipy.sparse.dok_array, scipy.sparse.bsr_array,
])
def test_every_sparse_format_gives_the_same_csr_array(constructor):
    """The input format changes nothing, down to the stored arrays of the output."""
    rng = np.random.default_rng(4)
    shape_frame = (12, 9)
    field = _field_smooth(shape_frame, rng)
    ims = _dyadic_images(shape_frame=shape_frame, rng=rng)
    out_reference = helpers.remap_sparse_images(ims_sparse_flat=scipy.sparse.csr_array(ims), remappingIdx=field)
    out = helpers.remap_sparse_images(ims_sparse_flat=constructor(ims), remappingIdx=field)
    _assert_canonical(out=out, shape=ims.shape, dtype=np.float32)
    _assert_csr_identical(out, out_reference, msg=constructor.__name__)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_noncanonical_input_is_summed_and_not_modified(dtype):
    """
    Unsorted indices, duplicated entries and explicit zeros in the input are
    handled (duplicates add, as scipy defines them), and the caller's arrays are
    left exactly as they were. ``dtype`` equal to the input's dtype is the case
    where scipy's conversions would share memory if allowed to.
    """
    rng = np.random.default_rng(5)
    shape_frame = (12, 9)
    field = _field_smooth(shape_frame, rng)
    ## Built from raw CSR triplets, because every scipy constructor from dense or COO canonicalizes on the way in
    indices = np.concatenate([rng.permutation(108)[:20], rng.permutation(108)[:20]] * 2).astype(np.int32)  ## each row: 20 columns stored twice, unsorted
    data = rng.integers(1, 1000, size=80).astype(dtype)
    data[[3, 50]] = 0
    ims = scipy.sparse.csr_array((data, indices, np.array([0, 40, 80], dtype=np.int32)), shape=(2, 108))
    data_before, indices_before, indptr_before = ims.data.copy(), ims.indices.copy(), ims.indptr.copy()

    out = helpers.remap_sparse_images(ims_sparse_flat=ims, remappingIdx=field, dtype=dtype)

    assert np.array_equal(ims.data, data_before) and np.array_equal(ims.indices, indices_before) and np.array_equal(ims.indptr, indptr_before)
    assert not np.shares_memory(out.data, ims.data)
    _assert_canonical(out=out, shape=(2, 108), dtype=dtype)
    ims_dense = np.zeros((2, 108))
    np.add.at(ims_dense, (np.repeat([0, 1], 40), indices_before), data_before)
    assert np.array_equal(out.toarray(), _warp_reference(ims=ims_dense, remappingIdx=field, method='linear'))


@pytest.mark.parametrize('dtype_in', [bool, np.uint8, np.int32, np.float32, np.float64])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_input_dtypes_and_output_dtype(dtype_in, dtype):
    """Any input dtype is cast to ``dtype``, which is also the dtype of the output."""
    rng = np.random.default_rng(6)
    shape_frame = (9, 12)
    field = _field_smooth(shape_frame, rng)
    ims = (_dyadic_images(shape_frame=shape_frame, rng=rng) % 200).astype(dtype_in)
    out = _warp_matmul(ims=ims, remappingIdx=field, method='linear', dtype=dtype)  ## asserts out.dtype == dtype
    assert np.array_equal(out, _warp_reference(ims=ims, remappingIdx=field, method='linear'))


@pytest.mark.parametrize('method', METHODS)
def test_remappingIdx_memory_layout_and_dtype_do_not_matter(method):
    """A float32, a Fortran-ordered and a strided view of the same (dyadic) field all give the same output."""
    rng = np.random.default_rng(7)
    shape_frame = (12, 9)
    field = _field_smooth(shape_frame, rng)
    ims = scipy.sparse.csr_array(_dyadic_images(shape_frame=shape_frame, rng=rng))
    field_strided = np.repeat(field, 2, axis=1)[:, ::2]
    assert not field_strided.flags['C_CONTIGUOUS']
    out_reference = helpers.remap_sparse_images(ims_sparse_flat=ims, remappingIdx=field, method=method)
    for field_variant in [field.astype(np.float32), np.asfortranarray(field), field_strided]:
        _assert_csr_identical(helpers.remap_sparse_images(ims_sparse_flat=ims, remappingIdx=field_variant, method=method), out_reference)


@pytest.mark.parametrize('method', METHODS)
def test_linearity_and_row_independence(method):
    """warp(2a + b) == 2 warp(a) + warp(b) exactly on dyadic fixtures, and each row of a batch equals its own single call."""
    rng = np.random.default_rng(8)
    shape_frame = (12, 9)
    field = _field_folded(shape_frame, rng)
    ims = _dyadic_images(shape_frame=shape_frame, rng=rng, n_images=3)
    out = _warp_matmul(ims=ims, remappingIdx=field, method=method)
    assert np.array_equal(_warp_matmul(ims=2 * ims[[0]] + ims[[1]], remappingIdx=field, method=method), 2 * out[[0]] + out[[1]])
    for ii in range(3):
        assert np.array_equal(_warp_matmul(ims=ims[[ii]], remappingIdx=field, method=method), out[[ii]])


@pytest.mark.parametrize('method', METHODS)
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_output_is_bit_identical_for_any_batch_size_and_on_repeat(dtype, method):
    """
    ``n_pixels_per_batch`` only sets how many rows of the warp matrix exist at
    once. Generic real values are used on purpose: any change in accumulation
    order would show up in the last bits.
    """
    rng = np.random.default_rng(9)
    shape_frame = (23, 17)
    field = _field_translate(shape_frame) + rng.normal(0, 3, size=(23, 17, 2))
    ims = scipy.sparse.csr_array(rng.normal(0, 1, size=(6, 23 * 17)) * (rng.random((6, 23 * 17)) < 0.3))
    kwargs = dict(ims_sparse_flat=ims, remappingIdx=field, method=method, dtype=dtype)
    out_reference = helpers.remap_sparse_images(**kwargs)
    ## 1 -> one frame row per batch; 17 * 3 + 5 -> three rows; 23 * 17 -> whole frame; then a repeat of the default
    for n_pixels_per_batch in [1, 17, 17 * 3 + 5, 23 * 17 - 1, 23 * 17, 2**22]:
        out = helpers.remap_sparse_images(n_pixels_per_batch=n_pixels_per_batch, **kwargs)
        _assert_canonical(out=out, shape=ims.shape, dtype=dtype)
        _assert_csr_identical(out, out_reference, msg=f"n_pixels_per_batch={n_pixels_per_batch}")


@pytest.mark.parametrize('kwargs_bad', [
    {'remappingIdx': np.zeros((12, 9, 3))},
    {'remappingIdx': np.zeros((12, 9))},
    {'remappingIdx': np.zeros((9, 9, 2))},  ## H*W does not match the images
    {'remappingIdx': np.zeros((12, 9, 2)).tolist()},
    {'method': 'cubic'},
    {'dtype': np.int32},
    {'ims_sparse_flat': np.zeros((2, 108))},
], ids=lambda kwargs_bad: list(kwargs_bad.keys())[0])
def test_invalid_arguments_raise(kwargs_bad):
    """Arguments the function cannot honor fail loudly, before any work is done."""
    kwargs = {'ims_sparse_flat': scipy.sparse.csr_array((2, 108), dtype=np.float32), 'remappingIdx': _field_translate((12, 9)), 'method': 'linear', 'dtype': np.float32}
    helpers.remap_sparse_images(**kwargs)  ## the baseline is valid, so each failure below is due to the one bad argument
    with pytest.raises(AssertionError):
        helpers.remap_sparse_images(**{**kwargs, **kwargs_bad})


######################################################################################################################################
####################################################### ALIGNER-LEVEL ################################################################
######################################################################################################################################


def _rois_for_aligner(shape_frame: Tuple[int, int], rng: np.random.Generator, n_roi: int, include_empty: bool) -> scipy.sparse.csr_matrix:
    """ROIs in the *(n_roi, H*W)* float32 ``csr_matrix`` layout that ``data_importing`` produces. The last one is optionally all-zero."""
    H, W = shape_frame
    rois = np.zeros((n_roi, H, W), dtype=np.float32)
    for ii in range(n_roi - int(include_empty)):
        r0, c0 = int(rng.integers(2, H - 5)), int(rng.integers(2, W - 5))
        rois[ii, r0:r0 + 4, c0:c0 + 4] = rng.random((4, 4)).astype(np.float32) + 0.1
    return scipy.sparse.csr_matrix(rois.reshape(n_roi, H * W))


@pytest.fixture(scope='module')
def aligner_template():
    """One ``Aligner``, built once: ``ROICaT_Module.__init__`` calls ``util.system_info()``, which takes over a second."""
    return alignment.Aligner(verbose=False)


@pytest.fixture
def aligner(aligner_template):
    """A fresh ``Aligner`` per test, so no test sees another's ``params``."""
    return copy.deepcopy(aligner_template)


@pytest.mark.parametrize('method_warp', METHODS)
def test_transform_ROIs_contract(method_warp, aligner):
    """
    One float32 ``csr_array`` of shape *(n_roi, H*W)* per session, nonzero rows
    summing to 1, an all-zero ROI coming back as an all-zero row, and
    ``method_warp`` recorded in ``params``. Without normalization the output is
    the function's output.
    """
    rng = np.random.default_rng(10)
    shape_frame = (16, 13)
    rois = _rois_for_aligner(shape_frame=shape_frame, rng=rng, n_roi=6, include_empty=True)
    fields = [_field_smooth(shape_frame, rng) for _ in range(2)]

    out = aligner.transform_ROIs(ROIs=[rois, rois], remappingIdx=fields, normalize=True, method_warp=method_warp)
    assert isinstance(out, list) and (len(out) == 2)
    for rois_aligned in out:
        _assert_canonical(out=rois_aligned, shape=(6, 16 * 13), dtype=np.float32)
        assert np.isfinite(rois_aligned.data).all()
        sums = np.asarray(rois_aligned.sum(axis=1)).reshape(-1)
        assert np.allclose(sums[:-1], 1.0, atol=1e-6)
        assert sums[-1] == 0.0, 'the all-zero ROI should come back as an all-zero row'
    assert aligner.params['transform_ROIs']['method_warp'] == method_warp

    out_raw = aligner.transform_ROIs(ROIs=[rois], remappingIdx=fields[:1], normalize=False, method_warp=method_warp)[0]
    _assert_csr_identical(out_raw, helpers.remap_sparse_images(ims_sparse_flat=rois, remappingIdx=fields[0], method=method_warp, dtype=np.float32))


def test_transform_ROIs_method_warp_options(aligner):
    """``'linear'`` is the default everywhere and unknown options raise."""
    assert util.get_default_parameters()['alignment']['transform_ROIs']['method_warp'] == 'linear'
    rng = np.random.default_rng(11)
    shape_frame = (14, 12)
    rois = _rois_for_aligner(shape_frame=shape_frame, rng=rng, n_roi=3, include_empty=False)
    kwargs = dict(ROIs=[rois], remappingIdx=[_field_translate(shape_frame, shift_x=0.5, shift_y=-0.5)], normalize=True)

    out_default = aligner.transform_ROIs(**kwargs)
    assert aligner.params['transform_ROIs']['method_warp'] == 'linear'
    _assert_csr_identical(out_default[0], aligner.transform_ROIs(method_warp='linear', **kwargs)[0])

    for method_warp in ['cubic', 'bilinear', 'legacy_griddata_cubic', '', None]:
        with pytest.raises(AssertionError):
            aligner.transform_ROIs(method_warp=method_warp, **kwargs)
