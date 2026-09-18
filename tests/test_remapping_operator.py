"""
Unit tests for the sparse warp operator added for ROICaT issue #686:
``helpers.Remapping_operator2d``, ``helpers.support_for_remapping_operator2d``,
the ``backend='operator'`` path of ``helpers.remap_sparse_images``, and
``Aligner.transform_ROIs(method_warp=...)``.

Two references are written here from the documented contract, independently of
the implementation: \n
    * R1: ``_reference_loop``, a float64 python loop over destination pixels.
    * R2: ``_reference_map_coordinates``, a wrapper around
      ``scipy.ndimage.map_coordinates(mode='grid-constant', cval=0)``. \n
``'grid-constant'`` is the scipy padding mode that interpolates across the frame
edge the way the operator does; plain ``'constant'`` is not. ``map_coordinates``
propagates a NaN where a coordinate is non-finite, while the operator returns 0
there by design, so the wrapper masks those destination pixels to 0.

Exact equality is only demanded on dyadic fixtures: integer pixel values well
below 2**18 and field coordinates on multiples of 1/8. Every weight, product and
partial sum is then exactly representable in float32 and float64, so the order
the products are accumulated in cannot change the answer.

These tests need no downloaded test data and run on CPU.

To run the tests, use the command (in a terminal):
    pytest -v test_remapping_operator.py
"""

from typing import List, Tuple, Union

import copy
import inspect

import numpy as np
import pytest
import scipy.ndimage
import scipy.sparse
import torch

from roicat import helpers, util
from roicat.tracking import alignment


######################################################################################################################################
########################################################## REFERENCES ################################################################
######################################################################################################################################


def _taps_reference(
    x: float,
    y: float,
    interpolation_method: str,
) -> List[Tuple[float, float, float]]:
    """
    The interpolation kernel as a list of ``(row, col, weight)`` taps for one
    destination pixel, from the documented rules: ``'nearest'`` samples
    ``floor(coordinate + 0.5)``, ``'linear'`` is bilinear over the 4 pixels
    around ``floor(coordinate)``. Nothing is bounds-checked here.

    Args:
        x (float):
            Source column sampled by the destination pixel.
        y (float):
            Source row sampled by the destination pixel.
        interpolation_method (str):
            ``'nearest'`` or ``'linear'``.

    Returns:
        (List[Tuple[float, float, float]]):
            taps (List[Tuple[float, float, float]]):
                One ``(row, col, weight)`` per tap.
    """
    if interpolation_method == 'nearest':
        return [(np.floor(y + 0.5), np.floor(x + 0.5), 1.0)]
    elif interpolation_method == 'linear':
        x0, y0 = np.floor(x), np.floor(y)
        fx, fy = x - x0, y - y0
        return [
            (y0,     x0,     (1.0 - fy) * (1.0 - fx)),
            (y0,     x0 + 1, (1.0 - fy) * fx),
            (y0 + 1, x0,     fy * (1.0 - fx)),
            (y0 + 1, x0 + 1, fy * fx),
        ]
    raise ValueError(f"Unknown interpolation_method {interpolation_method}")


def _reference_loop(
    im: np.ndarray,
    remappingIdx: np.ndarray,
    interpolation_method: str,
) -> np.ndarray:
    """
    R1. Warps one image with a plain float64 loop over destination pixels.

    Each tap is bounds-checked on its own; a tap outside the frame contributes 0
    and the survivors are not renormalized. A tap whose weight is exactly 0 is
    skipped, because a pixel with zero weight is not sampled: that is what makes
    a non-finite *image* pixel contaminate only the destinations that actually
    read it. A non-finite *coordinate* fails every bounds check, so the
    destination pixel comes out 0.

    Args:
        im (np.ndarray):
            The image to warp. Shape: *(H, W)*.
        remappingIdx (np.ndarray):
            The remapping field. Shape: *(H, W, 2)*, last dim is *(x, y)*.
        interpolation_method (str):
            ``'nearest'`` or ``'linear'``.

    Returns:
        (np.ndarray):
            im_warped (np.ndarray):
                The warped image, float64. Shape: *(H, W)*.
    """
    im = np.asarray(im, dtype=np.float64)
    H, W = im.shape
    out = np.zeros((H, W), dtype=np.float64)
    ## An infinite coordinate makes inf-inf and inf*0, both NaN. Those taps are
    ## dropped by the bounds check below, so the warnings would be noise.
    with np.errstate(invalid='ignore'):
        for r_dest in range(H):
            for c_dest in range(W):
                taps = _taps_reference(
                    x=np.float64(remappingIdx[r_dest, c_dest, 0]),
                    y=np.float64(remappingIdx[r_dest, c_dest, 1]),
                    interpolation_method=interpolation_method,
                )
                acc = np.float64(0.0)
                for (row, col, weight) in taps:
                    if weight == 0.0:
                        continue
                    if (0 <= row) and (row < H) and (0 <= col) and (col < W):
                        acc = acc + im[int(row), int(col)] * weight
                out[r_dest, c_dest] = acc
    return out


def _reference_map_coordinates(
    im: np.ndarray,
    remappingIdx: np.ndarray,
    interpolation_method: str,
) -> np.ndarray:
    """
    R2. Warps one image with ``scipy.ndimage.map_coordinates``.

    ``map_coordinates`` returns NaN where a coordinate is non-finite; the
    operator under test returns 0 there. Those destination pixels are masked to 0
    so that the two can be compared.

    Args:
        im (np.ndarray):
            The image to warp. Shape: *(H, W)*.
        remappingIdx (np.ndarray):
            The remapping field. Shape: *(H, W, 2)*, last dim is *(x, y)*.
        interpolation_method (str):
            ``'nearest'`` (spline order 0) or ``'linear'`` (order 1).

    Returns:
        (np.ndarray):
            im_warped (np.ndarray):
                The warped image, float64. Shape: *(H, W)*.
    """
    order = {'nearest': 0, 'linear': 1}[interpolation_method]
    ## map_coordinates indexes (row, col); remappingIdx stores (x, y) = (col, row)
    coords = np.stack([
        np.asarray(remappingIdx[..., 1], dtype=np.float64),
        np.asarray(remappingIdx[..., 0], dtype=np.float64),
    ], axis=0)
    out = np.asarray(scipy.ndimage.map_coordinates(
        input=np.asarray(im, dtype=np.float64),
        coordinates=coords,
        order=order,
        mode='grid-constant',
        cval=0.0,
    ), dtype=np.float64).copy()
    out[~(np.isfinite(coords[0]) & np.isfinite(coords[1]))] = 0.0
    return out


def _reference_batch(
    ims: np.ndarray,
    remappingIdx: np.ndarray,
    interpolation_method: str,
    reference: str = 'map_coordinates',
) -> np.ndarray:
    """
    Applies a reference to a stack of flattened images.

    Args:
        ims (np.ndarray):
            Flattened images. Shape: *(n_images, H*W)*.
        remappingIdx (np.ndarray):
            The remapping field. Shape: *(H, W, 2)*.
        interpolation_method (str):
            ``'nearest'`` or ``'linear'``.
        reference (str):
            ``'loop'`` for R1 or ``'map_coordinates'`` for R2.

    Returns:
        (np.ndarray):
            ims_warped (np.ndarray):
                Shape: *(n_images, H*W)*, float64.
    """
    shape_frame = remappingIdx.shape[:2]
    fn = _reference_loop if reference == 'loop' else _reference_map_coordinates
    ims = np.asarray(ims, dtype=np.float64)
    out = [fn(im.reshape(shape_frame), remappingIdx, interpolation_method).reshape(-1) for im in ims]
    return np.stack(out, axis=0) if len(out) > 0 else np.zeros((0, ims.shape[1]), dtype=np.float64)


def _mask_destinations_sampling(
    remappingIdx: np.ndarray,
    rc_source: Tuple[int, int],
    interpolation_method: str,
) -> np.ndarray:
    """
    Which destination pixels read a given source pixel with a nonzero weight.

    Args:
        remappingIdx (np.ndarray):
            The remapping field. Shape: *(H, W, 2)*.
        rc_source (Tuple[int, int]):
            The *(row, col)* of the source pixel.
        interpolation_method (str):
            ``'nearest'`` or ``'linear'``.

    Returns:
        (np.ndarray):
            mask (np.ndarray):
                Boolean. Shape: *(H, W)*.
    """
    H, W = remappingIdx.shape[:2]
    mask = np.zeros((H, W), dtype=bool)
    with np.errstate(invalid='ignore'):
        for r_dest in range(H):
            for c_dest in range(W):
                taps = _taps_reference(
                    x=np.float64(remappingIdx[r_dest, c_dest, 0]),
                    y=np.float64(remappingIdx[r_dest, c_dest, 1]),
                    interpolation_method=interpolation_method,
                )
                for (row, col, weight) in taps:
                    if (weight != 0.0) and (row == rc_source[0]) and (col == rc_source[1]):
                        mask[r_dest, c_dest] = True
    return mask


######################################################################################################################################
########################################################## FIXTURES ##################################################################
######################################################################################################################################


KERNELS = ['nearest', 'linear']

## Every shape the two references are cross-checked on is also a shape the
## operator is judged at, and the other way round: an operator result must never
## be compared against a reference pair that was not shown to agree at that shape.
SHAPES_SMALL = [(12, 9), (9, 12), (1, 7), (7, 1), (5, 5), (2, 2)]


def _dyadic(a: np.ndarray, denominator: int = 8) -> np.ndarray:
    """Rounds coordinates onto multiples of ``1/denominator``."""
    return np.round(np.asarray(a, dtype=np.float64) * denominator) / denominator


def _grid(shape_frame: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """Returns the *(x, y)* coordinate grids of a frame, both shape *(H, W)*."""
    H, W = shape_frame
    yy, xx = np.meshgrid(np.arange(H, dtype=np.float64), np.arange(W, dtype=np.float64), indexing='ij')
    return xx, yy


def _field_identity(shape_frame: Tuple[int, int]) -> np.ndarray:
    """Every destination pixel samples itself."""
    xx, yy = _grid(shape_frame)
    return np.stack([xx, yy], axis=-1)


def _field_translate(shape_frame: Tuple[int, int], shift_x: float, shift_y: float) -> np.ndarray:
    """Moves the image content by ``(+shift_x, +shift_y)``."""
    xx, yy = _grid(shape_frame)
    return np.stack([xx - shift_x, yy - shift_y], axis=-1)


def _field_smooth(shape_frame: Tuple[int, int], rng: np.random.Generator, amplitude: float = 3.0) -> np.ndarray:
    """Identity plus a dyadic random displacement."""
    return _dyadic(_field_identity(shape_frame) + rng.uniform(-amplitude, amplitude, size=(*shape_frame, 2)))


def _field_affine(shape_frame: Tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    """A random rotation-scale-shear plus a translation, rounded to dyadic."""
    xx, yy = _grid(shape_frame)
    m = np.eye(2) + rng.uniform(-0.2, 0.2, size=(2, 2))
    t = rng.uniform(-3, 3, size=2)
    return _dyadic(np.stack([
        m[0, 0] * xx + m[0, 1] * yy + t[0],
        m[1, 0] * xx + m[1, 1] * yy + t[1],
    ], axis=-1))


def _field_folded(shape_frame: Tuple[int, int], amp_x: float = 15.0, amp_y: float = 10.0) -> np.ndarray:
    """
    A non-monotone field: the sinusoid fixture from the earlier prototypes, so
    several destination pixels sample the same source pixel and the field folds
    back on itself.
    """
    H, W = shape_frame
    xx, yy = _grid(shape_frame)
    return _dyadic(np.stack([
        xx + amp_x * np.sin(2 * np.pi * yy / max(H, 1)),
        yy + amp_y * np.cos(2 * np.pi * xx / max(W, 1)),
    ], axis=-1))


def _field_constant(shape_frame: Tuple[int, int], x: float = 2.0, y: float = 3.0) -> np.ndarray:
    """Every destination pixel samples the same source coordinate."""
    return np.stack([np.full(shape_frame, x), np.full(shape_frame, y)], axis=-1)


def _field_clamped(shape_frame: Tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    """A field whose coordinates are clipped to the frame edge, as RoMa's does."""
    H, W = shape_frame
    field = _field_smooth(shape_frame=shape_frame, rng=rng, amplitude=6.0)
    field[..., 0] = np.clip(field[..., 0], 0, W - 1)
    field[..., 1] = np.clip(field[..., 1], 0, H - 1)
    return field


def _field_nan_patch(shape_frame: Tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    """A smooth field with a rectangular patch of NaN and one infinite coordinate."""
    H, W = shape_frame
    field = _field_smooth(shape_frame=shape_frame, rng=rng)
    field[: max(H // 3, 1), : max(W // 3, 1), :] = np.nan
    field[H - 1, W - 1, 0] = np.inf
    field[0, W - 1, 1] = -np.inf
    return field


def _field_offframe(shape_frame: Tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    """A field that carries most of the frame off the edge."""
    H, W = shape_frame
    return _dyadic(_field_smooth(shape_frame=shape_frame, rng=rng) + np.array([W * 0.6, H * 0.6]))


def _make_field(kind: str, shape_frame: Tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    """Dispatches the field builders by name, for the parametrized and volume tests."""
    if kind == 'identity':
        return _field_identity(shape_frame)
    if kind == 'smooth':
        return _field_smooth(shape_frame, rng)
    if kind == 'affine':
        return _field_affine(shape_frame, rng)
    if kind == 'folded':
        return _field_folded(shape_frame)
    if kind == 'constant':
        return _field_constant(shape_frame)
    if kind == 'clamped':
        return _field_clamped(shape_frame, rng)
    if kind == 'nan_patch':
        return _field_nan_patch(shape_frame, rng)
    if kind == 'offframe':
        return _field_offframe(shape_frame, rng)
    raise ValueError(f"Unknown field kind {kind}")


FIELD_KINDS = ['identity', 'smooth', 'affine', 'folded', 'constant', 'clamped', 'nan_patch', 'offframe']


def _dyadic_images(
    shape_frame: Tuple[int, int],
    rng: np.random.Generator,
    n_images: int = 4,
    density: float = 0.25,
) -> np.ndarray:
    """
    A batch of flattened sparse images with small non-negative integer values, so
    that every product and sum of a dyadic warp is exactly representable.

    Returns:
        (np.ndarray):
            ims (np.ndarray):
                Shape: *(n_images, H*W)*, float64.
    """
    n_pixels = shape_frame[0] * shape_frame[1]
    values = rng.integers(1, 1000, size=(n_images, n_pixels)).astype(np.float64)
    return values * (rng.random((n_images, n_pixels)) < density)


def _to_dense(x: Union[np.ndarray, scipy.sparse.sparray, scipy.sparse.spmatrix]) -> np.ndarray:
    """Densifies sparse output so it can be compared with a reference."""
    return np.asarray(x.todense(), dtype=np.float64) if scipy.sparse.issparse(x) else np.asarray(x, dtype=np.float64)


def _warp_dense(
    ims: np.ndarray,
    remappingIdx: np.ndarray,
    interpolation_method: str,
    dtype: np.dtype = np.float64,
    support=None,
) -> np.ndarray:
    """Builds an operator and warps a batch of flattened dense images with it."""
    remapper = helpers.Remapping_operator2d(
        remappingIdx=remappingIdx,
        interpolation_method=interpolation_method,
        dtype=dtype,
        support=support,
    )
    return np.asarray(remapper(x=np.asarray(ims, dtype=dtype), batching=True))


def _assert_csr_identical(a, b, msg: str = ''):
    """Asserts that two CSR arrays are identical down to their stored arrays."""
    assert a.shape == b.shape, f"shape {a.shape} != {b.shape}. {msg}"
    assert a.dtype == b.dtype, f"dtype {a.dtype} != {b.dtype}. {msg}"
    assert np.array_equal(a.indptr, b.indptr), f"indptr differ. {msg}"
    assert np.array_equal(a.indices, b.indices), f"indices differ. {msg}"
    assert np.array_equal(a.data, b.data), f"data differ. {msg}"


class _Operator_narrowIndices(helpers.Remapping_operator2d):
    """
    The operator with an index dtype far too small for the matrix it builds. The
    real ``> 2**31`` branch would need a frame of billions of pixels to reach, so
    this subclass exercises the same overflow check at 200 x 200. Subclassing
    rather than monkeypatching because ``_dtype_index`` is a ``staticmethod`` and
    restoring it through ``getattr`` rebinds it as an instance method.
    """
    @staticmethod
    def _dtype_index(n_nonzero: int, n_index_max: int) -> np.dtype:
        return np.dtype(np.int16)


######################################################################################################################################
################################################ THE REFERENCES AGREE WITH EACH OTHER ################################################
######################################################################################################################################


@pytest.mark.parametrize('interpolation_method', KERNELS)
@pytest.mark.parametrize('shape_frame', SHAPES_SMALL)
@pytest.mark.parametrize('kind_field', FIELD_KINDS)
def test_references_agree(shape_frame, kind_field, interpolation_method):
    """
    R1 and R2 are independent of each other, so they must agree before either is
    used to judge the operator. On dyadic fixtures the agreement is exact.
    """
    rng = np.random.default_rng(0)
    field = _make_field(kind=kind_field, shape_frame=shape_frame, rng=rng)
    im = _dyadic_images(shape_frame, rng, n_images=1)[0].reshape(shape_frame)
    r1 = _reference_loop(im, field, interpolation_method)
    r2 = _reference_map_coordinates(im, field, interpolation_method)
    assert np.array_equal(r1, r2), f"R1 and R2 disagree, max diff {np.abs(r1 - r2).max()}"


######################################################################################################################################
######################################################## KERNEL PARITY ###############################################################
######################################################################################################################################


@pytest.mark.parametrize('interpolation_method', KERNELS)
@pytest.mark.parametrize('shape_frame', SHAPES_SMALL)
@pytest.mark.parametrize('kind_field', FIELD_KINDS)
def test_matches_references_exactly_on_dyadic_fixtures(shape_frame, kind_field, interpolation_method):
    """The float64 operator equals both references bit for bit on dyadic fixtures."""
    rng = np.random.default_rng(1)
    field = _make_field(kind=kind_field, shape_frame=shape_frame, rng=rng)
    ims = _dyadic_images(shape_frame, rng, n_images=3)
    out = _warp_dense(ims, field, interpolation_method, dtype=np.float64)
    assert np.array_equal(out, _reference_batch(ims, field, interpolation_method, reference='loop'))
    assert np.array_equal(out, _reference_batch(ims, field, interpolation_method, reference='map_coordinates'))


@pytest.mark.parametrize('interpolation_method', KERNELS)
@pytest.mark.parametrize('shape_frame', [(256, 206), (512, 705), (1, 64), (64, 1)])
def test_matches_reference_on_larger_nonsquare_frames(shape_frame, interpolation_method):
    """
    The same parity at the sizes of real ROICaT datasets: 256 x 206 is the
    smallest (scout) and 512 x 705 is the Harnett dendrite frame. R1's python loop
    is too slow here, so only R2 is used.
    """
    rng = np.random.default_rng(2)
    field = _field_smooth(shape_frame=shape_frame, rng=rng, amplitude=5.0)
    ims = _dyadic_images(shape_frame, rng, n_images=2, density=0.05)
    out = _warp_dense(ims, field, interpolation_method, dtype=np.float64)
    assert np.array_equal(out, _reference_batch(ims, field, interpolation_method))


@pytest.mark.parametrize('interpolation_method', KERNELS)
def test_random_real_valued_field_within_tolerance(interpolation_method):
    """
    Off the dyadic grid the summation order can matter, so the criterion is a
    relative tolerance and the nonzero pattern rather than bit equality.
    """
    rng = np.random.default_rng(3)
    shape_frame = (23, 17)
    field = _field_identity(shape_frame) + rng.normal(0, 2.5, size=(*shape_frame, 2))
    ims = rng.random((3, shape_frame[0] * shape_frame[1])) * (rng.random((3, shape_frame[0] * shape_frame[1])) < 0.3)
    ref = _reference_batch(ims, field, interpolation_method)
    out64 = _warp_dense(ims, field, interpolation_method, dtype=np.float64)
    assert np.abs(out64 - ref).max() <= 1e-12 * np.abs(ref).max()
    assert np.array_equal(out64 != 0, ref != 0)
    out32 = _warp_dense(ims, field, interpolation_method, dtype=np.float32)
    assert np.abs(out32.astype(np.float64) - ref).max() <= 1e-6 * np.abs(ref).max()


@pytest.mark.parametrize('interpolation_method', KERNELS)
@pytest.mark.parametrize('shape_frame', [(12, 9), (9, 12)])
def test_identity_field_is_the_identity(shape_frame, interpolation_method):
    """An identity field returns the input bit for bit, and its matrix is one tap per pixel."""
    rng = np.random.default_rng(4)
    ims = _dyadic_images(shape_frame, rng, n_images=3)
    field = _field_identity(shape_frame)
    assert np.array_equal(_warp_dense(ims, field, interpolation_method, dtype=np.float64), ims)
    remapper = helpers.Remapping_operator2d(remappingIdx=field, interpolation_method=interpolation_method, dtype=np.float64)
    assert remapper.Wt.nnz == shape_frame[0] * shape_frame[1]


@pytest.mark.parametrize('interpolation_method', KERNELS)
@pytest.mark.parametrize('shape_frame', [(12, 9), (9, 12)])
@pytest.mark.parametrize('shift_x,shift_y', [(3, 0), (0, 2), (-2, 3)])
def test_integer_translation_pins_axis_and_flatten_order(shape_frame, shift_x, shift_y, interpolation_method):
    """
    An integer translation must equal a zero-filled shift of the image. This pins
    that ``remappingIdx[..., 0]`` is the column and ``[..., 1]`` the row, and that
    images are flattened in C order.
    """
    rng = np.random.default_rng(5)
    H, W = shape_frame
    im = _dyadic_images(shape_frame, rng, n_images=1, density=1.0)[0].reshape(shape_frame)
    field = _field_translate(shape_frame=shape_frame, shift_x=shift_x, shift_y=shift_y)

    expected = np.zeros_like(im)
    rows_dest = slice(max(shift_y, 0), H + min(shift_y, 0))
    cols_dest = slice(max(shift_x, 0), W + min(shift_x, 0))
    rows_src = slice(max(-shift_y, 0), H + min(-shift_y, 0))
    cols_src = slice(max(-shift_x, 0), W + min(-shift_x, 0))
    expected[rows_dest, cols_dest] = im[rows_src, cols_src]

    out = _warp_dense(im.reshape(1, -1), field, interpolation_method, dtype=np.float64).reshape(shape_frame)
    assert np.array_equal(out, expected)


@pytest.mark.parametrize('rc_source', [(0, 0), (0, 8), (11, 0), (11, 8), (0, 4), (11, 4), (6, 0), (6, 8)])
@pytest.mark.parametrize('interpolation_method', KERNELS)
def test_rois_on_every_edge_and_corner(rc_source, interpolation_method):
    """A single-pixel ROI at each frame edge and corner still matches the references."""
    shape_frame = (12, 9)
    im = np.zeros(shape_frame, dtype=np.float64)
    im[rc_source] = 511.0
    field = _field_translate(shape_frame=shape_frame, shift_x=0.5, shift_y=0.5)
    out = _warp_dense(im.reshape(1, -1), field, interpolation_method, dtype=np.float64).reshape(shape_frame)
    assert np.array_equal(out, _reference_loop(im, field, interpolation_method))
    assert np.array_equal(out, _reference_map_coordinates(im, field, interpolation_method))


def test_edge_taps_are_not_renormalized():
    """
    A destination pixel whose bilinear footprint hangs off the corner keeps only
    the weight of the taps that landed inside. With renormalization the corner
    would read 1.0 instead of 0.25.
    """
    shape_frame = (6, 5)
    im = np.zeros(shape_frame, dtype=np.float64)
    im[0, 0] = 1.0
    field = _field_translate(shape_frame=shape_frame, shift_x=0.5, shift_y=0.5)
    out = _warp_dense(im.reshape(1, -1), field, 'linear', dtype=np.float64).reshape(shape_frame)
    assert out[0, 0] == 0.25
    assert out[0, 1] == 0.25
    assert out[1, 0] == 0.25
    assert out[1, 1] == 0.25


@pytest.mark.parametrize('interpolation_method', KERNELS)
def test_displacement_partly_and_fully_off_frame(interpolation_method):
    """A partly off-frame warp matches the references; a fully off-frame one empties every row."""
    rng = np.random.default_rng(6)
    shape_frame = (12, 9)
    ims = _dyadic_images(shape_frame, rng, n_images=3)

    field_partial = _field_translate(shape_frame=shape_frame, shift_x=6, shift_y=4)
    out_partial = _warp_dense(ims, field_partial, interpolation_method, dtype=np.float64)
    assert np.array_equal(out_partial, _reference_batch(ims, field_partial, interpolation_method))
    assert out_partial.any()

    field_full = _field_translate(shape_frame=shape_frame, shift_x=1000, shift_y=1000)
    remapper = helpers.Remapping_operator2d(remappingIdx=field_full, interpolation_method=interpolation_method, dtype=np.float64)
    out_full = remapper(x=scipy.sparse.csr_array(ims), batching=True)
    assert out_full.shape == ims.shape
    assert out_full.nnz == 0
    assert np.array_equal(np.diff(out_full.indptr), np.zeros(ims.shape[0], dtype=np.int64))


######################################################################################################################################
###################################################### DEGENERATE INPUTS #############################################################
######################################################################################################################################


@pytest.mark.parametrize('interpolation_method', KERNELS)
def test_degenerate_rois(interpolation_method):
    """
    An empty ROI, a zero-row batch, a single pixel, a 1-D line and a diagonal all
    warp without a crash and match the references. These are the shapes the old
    ``safe`` branch of ``remap_sparse_images`` had to reroute.
    """
    shape_frame = (12, 9)
    field = _field_translate(shape_frame=shape_frame, shift_x=1.25, shift_y=-0.75)

    rois = np.zeros((5, *shape_frame), dtype=np.float64)
    ## rois[0] stays empty
    rois[1][5, 4] = 7.0                                    ## single pixel
    rois[2][3, :] = 3.0                                    ## horizontal line
    rois[3][np.arange(9), np.arange(9)] = 5.0              ## diagonal
    rois[4][:, 2] = 2.0                                    ## vertical line
    ims = rois.reshape(5, -1)

    remapper = helpers.Remapping_operator2d(remappingIdx=field, interpolation_method=interpolation_method, dtype=np.float64)
    out = remapper(x=scipy.sparse.csr_array(ims), batching=True)
    assert out.shape == ims.shape
    assert (out.indptr[1] - out.indptr[0]) == 0, 'the empty ROI should give an empty row'
    assert np.array_equal(_to_dense(out), _reference_batch(ims, field, interpolation_method))

    out_empty = remapper(x=scipy.sparse.csr_array(np.zeros((0, ims.shape[1]))), batching=True)
    assert out_empty.shape == (0, ims.shape[1])
    assert out_empty.nnz == 0


@pytest.mark.parametrize('interpolation_method', KERNELS)
def test_nan_patch_and_all_nan_field_give_zeros(interpolation_method):
    """A non-finite coordinate empties its destination pixel; it never yields NaN."""
    rng = np.random.default_rng(7)
    shape_frame = (12, 9)
    ims = _dyadic_images(shape_frame, rng, n_images=3, density=1.0)

    field_patch = _field_nan_patch(shape_frame=shape_frame, rng=rng)
    out_patch = _warp_dense(ims, field_patch, interpolation_method, dtype=np.float64)
    assert np.isfinite(out_patch).all()
    mask_bad = ~np.isfinite(field_patch).all(axis=-1)
    assert np.array_equal(out_patch.reshape(-1, *shape_frame)[:, mask_bad], np.zeros((3, int(mask_bad.sum()))))
    assert np.array_equal(out_patch, _reference_batch(ims, field_patch, interpolation_method))

    field_nan = np.full((*shape_frame, 2), np.nan)
    remapper = helpers.Remapping_operator2d(remappingIdx=field_nan, interpolation_method=interpolation_method, dtype=np.float64)
    assert remapper.Wt.nnz == 0
    out_nan = remapper(x=scipy.sparse.csr_array(ims), batching=True)
    assert out_nan.nnz == 0
    assert np.array_equal(_warp_dense(ims, field_nan, interpolation_method, dtype=np.float64), np.zeros_like(ims))


@pytest.mark.parametrize('value_bad', [np.nan, np.inf])
@pytest.mark.parametrize('interpolation_method', KERNELS)
def test_nonfinite_image_pixel_contaminates_only_its_samplers(value_bad, interpolation_method):
    """
    A non-finite value in the *image* spreads to exactly the destination pixels
    that read that source pixel with a nonzero weight. A pixel whose tap weight is
    0 is not sampled at all, so it is not contaminated: the operator drops
    zero-weight taps from the matrix, which is the sparse-matmul reading of
    ``0 * NaN``, not the dense IEEE one.
    """
    rng = np.random.default_rng(8)
    shape_frame = (12, 9)
    rc_source = (5, 4)
    im = _dyadic_images(shape_frame, rng, n_images=1, density=1.0)[0].reshape(shape_frame)
    im[rc_source] = value_bad
    field = _field_translate(shape_frame=shape_frame, shift_x=1.5, shift_y=0.5)

    mask_expected = _mask_destinations_sampling(remappingIdx=field, rc_source=rc_source, interpolation_method=interpolation_method)
    assert mask_expected.any() and (not mask_expected.all()), 'the fixture must contaminate some but not all pixels'

    out_dense = _warp_dense(im.reshape(1, -1), field, interpolation_method, dtype=np.float64).reshape(shape_frame)
    assert np.array_equal(~np.isfinite(out_dense), mask_expected)

    remapper = helpers.Remapping_operator2d(remappingIdx=field, interpolation_method=interpolation_method, dtype=np.float64)
    out_sparse = _to_dense(remapper(x=scipy.sparse.csr_array(im.reshape(1, -1)), batching=True)).reshape(shape_frame)
    assert np.array_equal(~np.isfinite(out_sparse), mask_expected)


@pytest.mark.parametrize('interpolation_method', KERNELS)
def test_constant_field_makes_the_output_dense(interpolation_method):
    """
    When every destination samples one source pixel, a one-pixel input comes back
    filling the whole frame. Worth pinning: it is the case where a sparse warp
    stops being sparse.
    """
    shape_frame = (12, 9)
    im = np.zeros(shape_frame, dtype=np.float64)
    im[3, 2] = 16.0
    field = _field_constant(shape_frame=shape_frame, x=2.0, y=3.0)
    remapper = helpers.Remapping_operator2d(remappingIdx=field, interpolation_method=interpolation_method, dtype=np.float64)
    out = remapper(x=scipy.sparse.csr_array(im.reshape(1, -1)), batching=True)
    assert out.nnz == shape_frame[0] * shape_frame[1]
    assert np.array_equal(_to_dense(out).reshape(shape_frame), np.full(shape_frame, 16.0))


######################################################################################################################################
################################################## ISSUE #686: HOLES STAY EMPTY ######################################################
######################################################################################################################################


def _roi_ring(shape_frame: Tuple[int, int], radius_inner: float = 5.0, radius_outer: float = 9.0) -> np.ndarray:
    """An annulus centred in the frame."""
    H, W = shape_frame
    yy, xx = np.meshgrid(np.arange(H, dtype=np.float64), np.arange(W, dtype=np.float64), indexing='ij')
    rr = np.sqrt((yy - (H - 1) / 2) ** 2 + (xx - (W - 1) / 2) ** 2)
    return ((rr >= radius_inner) & (rr <= radius_outer)).astype(np.float64)


def _roi_branching(shape_frame: Tuple[int, int]) -> np.ndarray:
    """Three thin branches leaving a common stem, with empty wedges between them."""
    H, W = shape_frame
    im = np.zeros(shape_frame, dtype=np.float64)
    r_c, c_c = H // 2, W // 2
    im[r_c, 2:c_c + 1] = 1.0
    for ii in range(c_c, W - 2):
        im[min(r_c + (ii - c_c), H - 1), ii] = 1.0
        im[max(r_c - (ii - c_c), 0), ii] = 1.0
    return im


@pytest.mark.parametrize('interpolation_method', KERNELS)
def test_ring_roi_keeps_its_hole_empty(interpolation_method):
    """
    The #686 assertion. Each output pixel is a weighted sum of the pixels under
    it, so the middle of an annulus stays empty under a sub-pixel warp.
    """
    shape_frame = (24, 24)
    im = _roi_ring(shape_frame=shape_frame)
    field = _field_translate(shape_frame=shape_frame, shift_x=0.5, shift_y=-0.5)
    out = _warp_dense(im.reshape(1, -1), field, interpolation_method, dtype=np.float64).reshape(shape_frame)

    H, W = shape_frame
    yy, xx = np.meshgrid(np.arange(H, dtype=np.float64), np.arange(W, dtype=np.float64), indexing='ij')
    rr = np.sqrt((yy - (H - 1) / 2) ** 2 + (xx - (W - 1) / 2) ** 2)
    mask_hole = rr <= 3.0
    assert im[mask_hole].sum() == 0, 'the fixture must have an empty hole to begin with'
    assert out[mask_hole].sum() == 0, 'the hole of a ring ROI was filled in'
    assert out.sum() > 0, 'the ring itself should survive the warp'


@pytest.mark.parametrize('interpolation_method', KERNELS)
def test_branching_roi_keeps_its_gaps_empty(interpolation_method):
    """The wedges between the branches of a non-convex ROI stay empty."""
    shape_frame = (24, 24)
    im = _roi_branching(shape_frame=shape_frame)
    field = _field_translate(shape_frame=shape_frame, shift_x=0.25, shift_y=0.25)
    out = _warp_dense(im.reshape(1, -1), field, interpolation_method, dtype=np.float64).reshape(shape_frame)

    ## Pixels at least 2 away from any nonzero of the input cannot be reached by a
    ## 0.25 px bilinear warp.
    mask_far = scipy.ndimage.binary_dilation(im > 0, iterations=2) == 0
    assert mask_far.sum() > 20, 'the fixture must have real gaps'
    assert out[mask_far].sum() == 0, 'gaps of a branching ROI were filled in'


def test_legacy_griddata_path_fills_the_hole():
    """
    The contrast that motivates issue #686: the old scattered-point path
    interpolates over the convex hull, so the middle of a ring gets filled.
    """
    shape_frame = (24, 24)
    im = _roi_ring(shape_frame=shape_frame)
    field = _field_translate(shape_frame=shape_frame, shift_x=0.5, shift_y=-0.5)
    out_legacy = helpers.remap_sparse_images(
        ims_sparse=[scipy.sparse.csr_array(im)],
        remappingIdx=field,
        method='cubic',
        fill_value=0,
        dtype=np.float32,
        safe=True,
        verbose=False,
    )[0]

    H, W = shape_frame
    yy, xx = np.meshgrid(np.arange(H, dtype=np.float64), np.arange(W, dtype=np.float64), indexing='ij')
    mask_hole = np.sqrt((yy - (H - 1) / 2) ** 2 + (xx - (W - 1) / 2) ** 2) <= 3.0
    assert _to_dense(out_legacy)[mask_hole].sum() > 0, 'the legacy path is expected to fill the hole'


######################################################################################################################################
######################################################## CALL SEMANTICS ##############################################################
######################################################################################################################################


@pytest.mark.parametrize('interpolation_method', KERNELS)
def test_batch_equals_single_calls_and_keeps_row_order(interpolation_method):
    """Warping n images at once equals n calls of one image, in the same order."""
    rng = np.random.default_rng(9)
    shape_frame = (12, 9)
    field = _field_smooth(shape_frame=shape_frame, rng=rng)
    ims = _dyadic_images(shape_frame, rng, n_images=6)
    remapper = helpers.Remapping_operator2d(remappingIdx=field, interpolation_method=interpolation_method, dtype=np.float64)

    out_batch = remapper(x=scipy.sparse.csr_array(ims), batching=True)
    out_singles = scipy.sparse.vstack(
        [remapper(x=scipy.sparse.csr_array(ims[ii:ii + 1]), batching=True) for ii in range(ims.shape[0])],
        format='csr',
    )
    assert np.array_equal(_to_dense(out_batch), _to_dense(out_singles))

    order = np.array([4, 0, 5, 1, 3, 2])
    out_permuted = remapper(x=scipy.sparse.csr_array(ims[order]), batching=True)
    assert np.array_equal(_to_dense(out_permuted), _to_dense(out_batch)[order])


@pytest.mark.parametrize('interpolation_method', KERNELS)
def test_batching_false_matches_batching_true(interpolation_method):
    """A single *(H, W)* image round-trips through ``batching=False``, sparse and dense."""
    rng = np.random.default_rng(10)
    shape_frame = (12, 9)
    field = _field_smooth(shape_frame=shape_frame, rng=rng)
    im = _dyadic_images(shape_frame, rng, n_images=1)[0].reshape(shape_frame)
    remapper = helpers.Remapping_operator2d(remappingIdx=field, interpolation_method=interpolation_method, dtype=np.float64)

    out_batched = _to_dense(remapper(x=scipy.sparse.csr_array(im.reshape(1, -1)), batching=True)).reshape(shape_frame)
    out_2d_sparse = remapper(x=scipy.sparse.csr_array(im), batching=False)
    out_2d_dense = remapper(x=im, batching=False)
    assert out_2d_sparse.shape == shape_frame
    assert out_2d_dense.shape == shape_frame
    assert np.array_equal(_to_dense(out_2d_sparse), out_batched)
    assert np.array_equal(out_2d_dense, out_batched)


@pytest.mark.parametrize('interpolation_method', KERNELS)
def test_sparse_input_equals_dense_input(interpolation_method):
    """Sparse in gives sparse out, dense in gives dense out, same numbers."""
    rng = np.random.default_rng(11)
    shape_frame = (12, 9)
    field = _field_smooth(shape_frame=shape_frame, rng=rng)
    ims = _dyadic_images(shape_frame, rng, n_images=4)
    remapper = helpers.Remapping_operator2d(remappingIdx=field, interpolation_method=interpolation_method, dtype=np.float64)

    out_sparse = remapper(x=scipy.sparse.csr_array(ims), batching=True)
    out_dense = remapper(x=ims, batching=True)
    assert scipy.sparse.issparse(out_sparse)
    assert isinstance(out_dense, np.ndarray)
    assert np.array_equal(_to_dense(out_sparse), out_dense)


@pytest.mark.parametrize('constructor', [
    scipy.sparse.csr_array,
    scipy.sparse.csc_array,
    scipy.sparse.coo_array,
    scipy.sparse.csr_matrix,
    scipy.sparse.csc_matrix,
    scipy.sparse.lil_matrix,
])
def test_every_sparse_format_gives_the_same_csr_array(constructor):
    """Any scipy sparse format in; a canonical ``csr_array`` out."""
    rng = np.random.default_rng(12)
    shape_frame = (12, 9)
    field = _field_smooth(shape_frame=shape_frame, rng=rng)
    ims = _dyadic_images(shape_frame, rng, n_images=3)
    remapper = helpers.Remapping_operator2d(remappingIdx=field, interpolation_method='linear', dtype=np.float64)

    out_reference = remapper(x=scipy.sparse.csr_array(ims), batching=True)
    out = remapper(x=constructor(ims), batching=True)
    assert isinstance(out, scipy.sparse.csr_array), f"expected csr_array, got {type(out)}"
    _assert_csr_identical(out, out_reference, msg=f"format {constructor.__name__}")


def _noncanonical_batch(
    seed: int,
    n_pixels: int,
    dtype: np.dtype,
    n_entries: int = 40,
    n_duplicated: int = 6,
) -> scipy.sparse.csr_array:
    """
    A CSR array built straight from triplets, with unsorted column indices,
    duplicated column ids and explicit zeros. Hand-built because every scipy
    conversion (``tocsr``, ``coo_array``, ``todense``) canonicalizes on the way
    through, so a fixture made the obvious way would arrive already clean and the
    test would assert nothing.

    The values are generic floats rather than the dyadic ones used elsewhere,
    precisely so that the order the products accumulate in changes the last bits.

    No column appears more than twice. With three or more copies the sum itself
    depends on the order they are added in, so ``a + b + c`` stored one way and
    the dense array it flattens to would already differ in their last bits before
    the operator sees either, and the comparison would be meaningless.
    """
    rng = np.random.default_rng(seed)
    n_rows = 2
    n_single = n_entries // n_rows - n_duplicated * 2
    indices = []
    for _ in range(n_rows):
        ## Distinct columns per row, then the last few are stored twice, so every
        ## multiplicity is exactly 2 and the within-row order is scrambled.
        columns = rng.permutation(n_pixels)[:n_single + n_duplicated]
        idx_row = np.concatenate([columns[:n_single], np.repeat(columns[n_single:], 2)])
        rng.shuffle(idx_row)
        indices.append(idx_row)
    indptr = np.concatenate([[0], np.cumsum([len(i) for i in indices])]).astype(np.int32)
    indices = np.concatenate(indices).astype(np.int32)
    data = rng.normal(0, 1, size=indices.size).astype(dtype)
    data[rng.integers(0, indices.size, size=2)] = 0.0
    return scipy.sparse.csr_array((data, indices, indptr), shape=(n_rows, n_pixels))


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('seed', [30, 31, 32, 33])
def test_noncanonical_input_gives_the_same_canonical_output(dtype, seed):
    """
    A matrix stored with unsorted indices, duplicate column ids and explicit zeros
    must give the same bits as the same matrix stored cleanly, and the output must
    be canonical. scipy multiplies a CSR matrix as it is stored, so without the
    operator's own ``sum_duplicates`` the two storages land on different last bits.
    """
    shape_frame = (12, 9)
    n_pixels = shape_frame[0] * shape_frame[1]
    field = _field_identity(shape_frame) + np.random.default_rng(13).normal(0, 2.0, size=(*shape_frame, 2))

    x_messy = _noncanonical_batch(seed=seed, n_pixels=n_pixels, dtype=dtype)
    assert not x_messy.has_canonical_format, 'the fixture must actually be non-canonical'
    data, indices, indptr = x_messy.data.copy(), x_messy.indices.copy(), x_messy.indptr.copy()
    ## The same matrix, stored cleanly and in the same dtype
    x_clean = scipy.sparse.csr_array(np.asarray(x_messy.todense()))
    assert x_clean.has_canonical_format and (x_clean.dtype == np.dtype(dtype))
    ## The two storages really are the same matrix, bit for bit, before the
    ## operator sees either. Without this the test could fail on a difference
    ## that scipy's own densification introduced.
    x_canonicalized = x_messy.copy()
    x_canonicalized.sum_duplicates()
    x_canonicalized.eliminate_zeros()
    assert np.array_equal(x_canonicalized.indices, x_clean.indices)
    assert np.array_equal(x_canonicalized.data, x_clean.data)
    assert x_messy.nnz > x_clean.nnz, 'the fixture must carry duplicates and explicit zeros'

    remapper = helpers.Remapping_operator2d(remappingIdx=field, interpolation_method='linear', dtype=dtype)
    out_messy = remapper(x=x_messy, batching=True)
    out_clean = remapper(x=x_clean, batching=True)
    _assert_csr_identical(out_messy, out_clean, msg=f"seed={seed} dtype={dtype}")
    assert out_messy.has_canonical_format
    assert np.all(out_messy.data != 0), 'canonical output should carry no explicit zeros'
    for i_row in range(out_messy.shape[0]):
        idx_row = out_messy.indices[out_messy.indptr[i_row]:out_messy.indptr[i_row + 1]]
        assert np.all(np.diff(idx_row) > 0), 'canonical output should have sorted, unique indices per row'
    ## The caller's arrays are untouched
    assert np.array_equal(x_messy.data, data)
    assert np.array_equal(x_messy.indices, indices)
    assert np.array_equal(x_messy.indptr, indptr)


def test_noncanonical_input_in_a_wider_dtype_agrees_only_to_a_tolerance():
    """
    Documents a corner the class docstring does not qualify. The input is cast to
    the operator's dtype *before* its duplicates are summed, so a float64 matrix
    with duplicate entries and a float32 operator computes
    ``f32(f32(a) + f32(b))`` where the same matrix stored cleanly computes
    ``f32(a + b)``. The two agree to about one float32 ulp, not bit for bit.

    Every ROICaT caller hands over an input already in the operator's dtype, which
    is the case the test above pins exactly.
    """
    shape_frame = (12, 9)
    n_pixels = shape_frame[0] * shape_frame[1]
    field = _field_identity(shape_frame) + np.random.default_rng(13).normal(0, 2.0, size=(*shape_frame, 2))

    x_messy = _noncanonical_batch(seed=30, n_pixels=n_pixels, dtype=np.float64)
    x_clean = scipy.sparse.csr_array(np.asarray(x_messy.todense()))
    ## Same precondition as the matched-dtype test: the two storages are the same
    ## float64 matrix, so anything the operator does differently is the operator's.
    x_canonicalized = x_messy.copy()
    x_canonicalized.sum_duplicates()
    x_canonicalized.eliminate_zeros()
    assert np.array_equal(x_canonicalized.indices, x_clean.indices)
    assert np.array_equal(x_canonicalized.data, x_clean.data)

    remapper = helpers.Remapping_operator2d(remappingIdx=field, interpolation_method='linear', dtype=np.float32)
    out_messy = _to_dense(remapper(x=x_messy, batching=True))
    out_clean = _to_dense(remapper(x=x_clean, batching=True))
    assert np.abs(out_messy - out_clean).max() <= 1e-6 * np.abs(out_clean).max()


def test_explicit_zero_outside_the_support_is_a_support_violation():
    """
    The input is canonicalized but never has its explicit zeros eliminated, so a
    stored zero at a pixel outside the support trips the support check. The
    documented way of building the support is to hand it the images themselves,
    and ``_flatIdx_from_support`` counts every stored entry including the zeros,
    so that path is consistent.
    """
    shape_frame = (12, 9)
    n_pixels = shape_frame[0] * shape_frame[1]
    field = _field_identity(shape_frame)
    x = scipy.sparse.csr_array((np.array([1.0, 0.0]), np.array([5, 60]), np.array([0, 2])), shape=(1, n_pixels))

    remapper_tight = helpers.Remapping_operator2d(remappingIdx=field, interpolation_method='linear', dtype=np.float64, support=np.array([5]))
    with pytest.raises(ValueError):
        remapper_tight(x=x, batching=True)

    remapper_from_x = helpers.Remapping_operator2d(remappingIdx=field, interpolation_method='linear', dtype=np.float64, support=x)
    assert np.array_equal(_to_dense(remapper_from_x(x=x, batching=True)), _to_dense(helpers.Remapping_operator2d(
        remappingIdx=field, interpolation_method='linear', dtype=np.float64)(x=x, batching=True)))


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('use_support', [False, True])
def test_input_is_not_modified_and_does_not_share_memory(dtype, use_support):
    """
    ``scipy.sparse.csr_array(A)`` shares index arrays with ``A``, so an in-place
    sort inside the operator would reorder the caller's matrix. Checked in the
    case that can actually go wrong: an input already in the operator's dtype.
    """
    rng = np.random.default_rng(14)
    shape_frame = (12, 9)
    field = _field_smooth(shape_frame=shape_frame, rng=rng)
    x = scipy.sparse.csr_array(_dyadic_images(shape_frame, rng, n_images=4).astype(dtype))
    data, indices, indptr = x.data.copy(), x.indices.copy(), x.indptr.copy()

    remapper = helpers.Remapping_operator2d(
        remappingIdx=field,
        interpolation_method='linear',
        dtype=dtype,
        support=x if use_support else None,
    )
    out = remapper(x=x, batching=True)
    assert np.array_equal(x.data, data)
    assert np.array_equal(x.indices, indices)
    assert np.array_equal(x.indptr, indptr)
    assert not np.shares_memory(out.data, x.data)
    assert not np.shares_memory(out.indices, x.indices)
    assert not np.shares_memory(out.indptr, x.indptr)


@pytest.mark.parametrize('dtype_in', [bool, np.uint8, np.int32, np.float32, np.float64])
@pytest.mark.parametrize('dtype_op', [np.float32, np.float64])
def test_input_dtypes_and_output_dtype(dtype_in, dtype_op):
    """Any input dtype is accepted; the output dtype is always the operator's."""
    rng = np.random.default_rng(15)
    shape_frame = (12, 9)
    field = _field_smooth(shape_frame=shape_frame, rng=rng)
    pattern = (rng.random((3, shape_frame[0] * shape_frame[1])) < 0.3)
    ims = pattern if dtype_in is bool else (pattern * rng.integers(1, 100, size=pattern.shape)).astype(dtype_in)
    remapper = helpers.Remapping_operator2d(remappingIdx=field, interpolation_method='linear', dtype=dtype_op)

    out_sparse = remapper(x=scipy.sparse.csr_array(ims), batching=True)
    out_dense = remapper(x=ims, batching=True)
    assert out_sparse.dtype == np.dtype(dtype_op)
    assert out_dense.dtype == np.dtype(dtype_op)
    ref = _reference_batch(np.asarray(ims, dtype=np.float64), field, 'linear')
    assert np.abs(_to_dense(out_sparse) - ref).max() <= 1e-6 * max(np.abs(ref).max(), 1.0)


@pytest.mark.parametrize('dtype_bad', [np.int32, np.int64, np.uint8, bool])
def test_non_floating_dtype_raises(dtype_bad):
    """An integer dtype would round every interpolation weight to 0, so it is rejected."""
    field = _field_identity((6, 5))
    with pytest.raises(AssertionError):
        helpers.Remapping_operator2d(remappingIdx=field, interpolation_method='linear', dtype=dtype_bad)


def test_unknown_interpolation_method_raises():
    """An unknown kernel name raises ``ValueError``, not a silent fallback."""
    field = _field_identity((6, 5))
    with pytest.raises(ValueError):
        helpers.Remapping_operator2d(remappingIdx=field, interpolation_method='cubic', dtype=np.float64)


@pytest.mark.parametrize('interpolation_method', KERNELS)
def test_remappingIdx_forms_agree(interpolation_method):
    """
    A float32 field, a torch tensor and a non-contiguous permuted view all give
    the same answer as a contiguous float64 numpy array. ``resize_remappingIdx``
    returns a permuted tensor, so the non-contiguous case is the realistic one.
    The field is dyadic, so float32 holds it exactly and equality can be demanded.
    """
    rng = np.random.default_rng(16)
    shape_frame = (12, 9)
    field = _field_smooth(shape_frame=shape_frame, rng=rng)
    ims = _dyadic_images(shape_frame, rng, n_images=3)
    out_reference = _warp_dense(ims, field, interpolation_method, dtype=np.float64)

    field_permuted = np.moveaxis(np.ascontiguousarray(np.moveaxis(field, -1, 0)), 0, -1)
    assert not field_permuted.flags['C_CONTIGUOUS']
    tensor_permuted = torch.as_tensor(np.ascontiguousarray(np.moveaxis(field, -1, 0))).permute(1, 2, 0)
    assert not tensor_permuted.is_contiguous()

    for name, f in [
        ('float32', field.astype(np.float32)),
        ('torch_float64', torch.as_tensor(field)),
        ('torch_float32', torch.as_tensor(field.astype(np.float32))),
        ('numpy_noncontiguous', field_permuted),
        ('torch_noncontiguous', tensor_permuted),
    ]:
        out = _warp_dense(ims, f, interpolation_method, dtype=np.float64)
        assert np.array_equal(out, out_reference), f"field form {name} disagrees"


def test_torch_input_returns_a_torch_tensor():
    """A ``torch.Tensor`` image goes in and a CPU tensor comes out."""
    rng = np.random.default_rng(17)
    shape_frame = (12, 9)
    field = _field_smooth(shape_frame=shape_frame, rng=rng)
    ims = _dyadic_images(shape_frame, rng, n_images=3)
    remapper = helpers.Remapping_operator2d(remappingIdx=field, interpolation_method='linear', dtype=np.float64)
    out = remapper(x=torch.as_tensor(ims), batching=True)
    assert isinstance(out, torch.Tensor)
    assert np.array_equal(out.numpy(), remapper(x=ims, batching=True))


@pytest.mark.parametrize('interpolation_method', KERNELS)
def test_linearity_is_exact_on_dyadic_fixtures(interpolation_method):
    """warp(a*u + b*v) == a*warp(u) + b*warp(v), bit for bit."""
    rng = np.random.default_rng(18)
    shape_frame = (12, 9)
    field = _field_smooth(shape_frame=shape_frame, rng=rng)
    u = _dyadic_images(shape_frame, rng, n_images=2)
    v = _dyadic_images(shape_frame, rng, n_images=2)
    a, b = 2.0, 4.0
    remapper = helpers.Remapping_operator2d(remappingIdx=field, interpolation_method=interpolation_method, dtype=np.float64)
    out_combined = remapper(x=(a * u + b * v), batching=True)
    out_separate = a * remapper(x=u, batching=True) + b * remapper(x=v, batching=True)
    assert np.array_equal(out_combined, out_separate)


def test_nearest_rounds_half_up():
    """
    ``'nearest'`` samples ``floor(coordinate + 0.5)``, so a coordinate exactly on
    a half picks the higher pixel. Libraries disagree here (torch and numpy round
    half to even), which is why the rule is pinned rather than inherited.
    """
    shape_frame = (4, 6)
    im = np.arange(shape_frame[0] * shape_frame[1], dtype=np.float64).reshape(shape_frame)
    field = np.zeros((*shape_frame, 2), dtype=np.float64)
    field[..., 1] = 0.0
    ## Destination row 0 reads a range of half-integer columns on source row 0
    field[0, :, 0] = np.array([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5])
    ## The rest read something harmless
    field[1:, :, 0] = 0.0
    out = _warp_dense(im.reshape(1, -1), field, 'nearest', dtype=np.float64).reshape(shape_frame)
    assert np.array_equal(out[0], im[0, [0, 1, 2, 3, 4, 5]])

    ## A half a pixel past the last column falls outside the frame and gives 0
    field_edge = np.zeros((*shape_frame, 2), dtype=np.float64)
    field_edge[..., 0] = shape_frame[1] - 1 + 0.5
    out_edge = _warp_dense(im.reshape(1, -1), field_edge, 'nearest', dtype=np.float64)
    assert np.array_equal(out_edge, np.zeros_like(out_edge))

    ## And just below -0.5 falls outside too
    field_low = np.zeros((*shape_frame, 2), dtype=np.float64)
    field_low[..., 0] = -0.5000001
    out_low = _warp_dense(im.reshape(1, -1), field_low, 'nearest', dtype=np.float64)
    assert np.array_equal(out_low, np.zeros_like(out_low))


@pytest.mark.parametrize('interpolation_method', KERNELS)
def test_two_calls_are_bit_equal(interpolation_method):
    """The pipeline's determinism gate depends on this."""
    rng = np.random.default_rng(19)
    shape_frame = (17, 23)
    field = _field_identity(shape_frame) + rng.normal(0, 2.0, size=(shape_frame[0], shape_frame[1], 2))
    ims = scipy.sparse.csr_array(rng.random((5, shape_frame[0] * shape_frame[1])) * (rng.random((5, shape_frame[0] * shape_frame[1])) < 0.3))
    remapper = helpers.Remapping_operator2d(remappingIdx=field, interpolation_method=interpolation_method, dtype=np.float32)
    _assert_csr_identical(remapper(x=ims, batching=True), remapper(x=ims, batching=True))


######################################################################################################################################
################################################### OCCUPIED-PIXEL BUILD #############################################################
######################################################################################################################################


@pytest.mark.parametrize('interpolation_method', KERNELS)
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('kind_field', FIELD_KINDS)
@pytest.mark.parametrize('form_support', ['batch', 'frame', 'flat'])
def test_occupied_build_equals_full_build(kind_field, dtype, interpolation_method, form_support):
    """
    The restricted build drops only matrix entries that would multiply a zero, so
    its output must be identical to the full build's down to the stored arrays.
    The support is given in all three documented forms; a *(12, 9)* frame is used
    so that the *(H, W)* and *(n_images, H*W)* branches of
    ``_flatIdx_from_support`` are actually distinguishable.
    """
    rng = np.random.default_rng(20)
    shape_frame = (12, 9)
    field = _make_field(kind=kind_field, shape_frame=shape_frame, rng=rng)
    x = scipy.sparse.csr_array(_dyadic_images(shape_frame, rng, n_images=5, density=0.2).astype(dtype))

    if form_support == 'batch':
        support = x
    elif form_support == 'frame':
        support = scipy.sparse.csr_array(np.asarray(x.todense()).sum(axis=0).reshape(shape_frame))
    else:
        support = np.unique(x.indices)

    remapper_full = helpers.Remapping_operator2d(remappingIdx=field, interpolation_method=interpolation_method, dtype=dtype)
    remapper_occupied = helpers.Remapping_operator2d(remappingIdx=field, interpolation_method=interpolation_method, dtype=dtype, support=support)
    assert remapper_full.Wt is not None
    assert (remapper_full.Wt_compact is None) and (remapper_full.src_ids is None) and (remapper_full.dest_ids is None)
    assert remapper_occupied.Wt is None
    assert remapper_occupied.Wt_compact is not None

    _assert_csr_identical(
        remapper_occupied(x=x, batching=True),
        remapper_full(x=x, batching=True),
        msg=f"field={kind_field} dtype={dtype} kernel={interpolation_method} support={form_support}",
    )
    assert np.array_equal(
        remapper_occupied(x=np.asarray(x.todense()), batching=True),
        remapper_full(x=np.asarray(x.todense()), batching=True),
    )


@pytest.mark.parametrize('rows_per_strip', [1, 3, 1000, None])
def test_rows_per_strip_does_not_change_the_result(rows_per_strip):
    """Striping the pass over the field is a memory knob only."""
    rng = np.random.default_rng(21)
    shape_frame = (12, 9)
    field = _field_smooth(shape_frame=shape_frame, rng=rng)
    x = scipy.sparse.csr_array(_dyadic_images(shape_frame, rng, n_images=4))
    reference = helpers.Remapping_operator2d(remappingIdx=field, interpolation_method='linear', dtype=np.float64)(x=x, batching=True)
    remapper = helpers.Remapping_operator2d(remappingIdx=field, interpolation_method='linear', dtype=np.float64, rows_per_strip=rows_per_strip)
    _assert_csr_identical(remapper(x=x, batching=True), reference)


def test_nonzero_outside_the_support_raises():
    """
    Dropping a nonzero silently would return a wrong answer with no signal, so
    both the sparse and the dense path raise instead.
    """
    rng = np.random.default_rng(22)
    shape_frame = (12, 9)
    n_pixels = shape_frame[0] * shape_frame[1]
    field = _field_smooth(shape_frame=shape_frame, rng=rng)
    x = scipy.sparse.csr_array(_dyadic_images(shape_frame, rng, n_images=3, density=0.2))
    remapper = helpers.Remapping_operator2d(remappingIdx=field, interpolation_method='linear', dtype=np.float64, support=x)

    x_extra = np.asarray(x.todense())
    idx_outside = int(np.setdiff1d(np.arange(n_pixels), np.unique(x.indices))[0])
    x_extra[0, idx_outside] = 9.0
    with pytest.raises(ValueError):
        remapper(x=scipy.sparse.csr_array(x_extra), batching=True)
    with pytest.raises(ValueError):
        remapper(x=x_extra, batching=True)


def test_empty_support_accepts_only_empty_images():
    """An operator built on nothing warps an all-zero batch and rejects anything else."""
    shape_frame = (12, 9)
    n_pixels = shape_frame[0] * shape_frame[1]
    field = _field_identity(shape_frame)
    remapper = helpers.Remapping_operator2d(
        remappingIdx=field,
        interpolation_method='linear',
        dtype=np.float64,
        support=np.zeros(0, dtype=np.int64),
    )
    out = remapper(x=scipy.sparse.csr_array((2, n_pixels), dtype=np.float64), batching=True)
    assert out.shape == (2, n_pixels)
    assert out.nnz == 0
    x_something = np.zeros((2, n_pixels)); x_something[0, 5] = 1.0
    with pytest.raises(ValueError):
        remapper(x=scipy.sparse.csr_array(x_something), batching=True)


def test_support_outside_the_frame_raises():
    """A flat index beyond the frame is a caller error, not something to clip."""
    field = _field_identity((6, 5))
    with pytest.raises(AssertionError):
        helpers.Remapping_operator2d(remappingIdx=field, interpolation_method='linear', dtype=np.float64, support=np.array([0, 30]))


######################################################################################################################################
##################################################### SUPPORT SELECTION HELPER #######################################################
######################################################################################################################################


def test_support_for_remapping_operator2d_threshold():
    """
    The helper is the single place the full-vs-restricted threshold is written
    down. It returns ``x`` above the threshold and ``None`` at or below it. Only
    the shape is inspected, so the large cases cost no memory.
    """
    x = scipy.sparse.csr_array(np.eye(4))
    assert helpers.support_for_remapping_operator2d(x=x, shape_frame=(4, 4), n_pixels_max_full=15) is x
    assert helpers.support_for_remapping_operator2d(x=x, shape_frame=(4, 4), n_pixels_max_full=16) is None
    assert helpers.support_for_remapping_operator2d(x=x, shape_frame=(4, 4), n_pixels_max_full=17) is None
    ## Default threshold: 4096 x 4096 builds the full matrix, one row more does not
    assert helpers.support_for_remapping_operator2d(x=x, shape_frame=(4096, 4096)) is None
    assert helpers.support_for_remapping_operator2d(x=x, shape_frame=(4097, 4096)) is x
    ## Every real ROICaT frame is below the threshold
    for shape_frame in [(512, 512), (512, 705), (4660, 512), (870, 1392), (256, 206)]:
        assert helpers.support_for_remapping_operator2d(x=x, shape_frame=shape_frame) is None


######################################################################################################################################
###################################################### INDEX DTYPE GUARD #############################################################
######################################################################################################################################


@pytest.mark.parametrize('n_nonzero,n_index_max,dtype_expected', [
    (0, 0, np.int32),
    (10, 10, np.int32),
    (2 ** 31 - 1, 2 ** 31 - 1, np.int32),
    (2 ** 31, 0, np.int64),          ## indptr has to count past int32
    (0, 2 ** 31, np.int64),          ## indices have to hold a column id past int32
    (2 ** 33, 2 ** 33, np.int64),
])
def test_dtype_index_rule(n_nonzero, n_index_max, dtype_expected):
    """
    The index dtype follows the assembled matrix, not the pixel count. A frame of
    23000 x 23000 has under 2**31 pixels but about 4x that many matrix nonzeros,
    so deciding from the pixel count wraps ``indptr`` silently. Exercised through
    the two-argument static method, which allocates nothing.
    """
    assert helpers.Remapping_operator2d._dtype_index(n_nonzero=n_nonzero, n_index_max=n_index_max) == np.dtype(dtype_expected)


def test_index_overflow_is_a_hard_error():
    """
    If the index dtype is ever too narrow for the matrix, the build must raise
    rather than return a wrapped, silently wrong operator. The real ``> 2**31``
    branch would need billions of pixels, so the same check is reached at
    200 x 200 with a deliberately undersized index dtype.
    """
    field = _field_identity((200, 200))
    ## The honest build is fine at this size
    helpers.Remapping_operator2d(remappingIdx=field, interpolation_method='nearest', dtype=np.float32)
    with pytest.raises(RuntimeError):
        _Operator_narrowIndices(remappingIdx=field, interpolation_method='nearest', dtype=np.float32)


######################################################################################################################################
########################################################## VOLUME ####################################################################
######################################################################################################################################


def test_volume_random_draws():
    """
    320 seeded draws over frame shape, field type, ROI count and density, and
    kernel. Each draw is checked against R2 and against the restricted build.
    Frames are kept small so the whole test runs in a few seconds.
    """
    shapes = [(1, 11), (11, 1), (4, 4), (12, 9), (9, 12), (16, 16), (7, 21), (21, 7), (32, 28), (28, 32)]
    n_draws = 320
    n_checked = 0
    for i_draw in range(n_draws):
        rng = np.random.default_rng(1000 + i_draw)
        shape_frame = shapes[int(rng.integers(len(shapes)))]
        kind_field = FIELD_KINDS[int(rng.integers(len(FIELD_KINDS)))]
        interpolation_method = KERNELS[int(rng.integers(len(KERNELS)))]
        field = _make_field(kind=kind_field, shape_frame=shape_frame, rng=rng)
        ims = _dyadic_images(
            shape_frame=shape_frame,
            rng=rng,
            n_images=int(rng.integers(0, 6)),
            density=float(rng.uniform(0.02, 0.6)),
        )
        msg = f"draw {i_draw}: shape={shape_frame} field={kind_field} kernel={interpolation_method} n={ims.shape[0]}"

        remapper_full = helpers.Remapping_operator2d(remappingIdx=field, interpolation_method=interpolation_method, dtype=np.float64)
        x = scipy.sparse.csr_array(ims)
        out_full = remapper_full(x=x, batching=True)
        assert np.array_equal(_to_dense(out_full), _reference_batch(ims, field, interpolation_method)), msg

        remapper_occupied = helpers.Remapping_operator2d(
            remappingIdx=field,
            interpolation_method=interpolation_method,
            dtype=np.float64,
            support=x,
        )
        _assert_csr_identical(remapper_occupied(x=x, batching=True), out_full, msg=msg)
        n_checked += 1
    assert n_checked == n_draws


######################################################################################################################################
################################################## remap_sparse_images NEW PATH ######################################################
######################################################################################################################################


def test_remap_sparse_images_signature_is_backwards_compatible():
    """
    The new ``backend`` argument is appended, so every existing positional call
    still means what it did.
    """
    params = list(inspect.signature(helpers.remap_sparse_images).parameters)
    assert params[:8] == ['ims_sparse', 'remappingIdx', 'method', 'fill_value', 'dtype', 'safe', 'n_workers', 'verbose']
    assert params[8] == 'backend'
    assert len(params) == 9
    assert inspect.signature(helpers.remap_sparse_images).parameters['backend'].default == 'griddata'


@pytest.mark.parametrize('interpolation_method', KERNELS)
def test_remap_sparse_images_operator_backend_equals_the_class(interpolation_method):
    """The public function's new path is the class, reshaped back into 2D images."""
    rng = np.random.default_rng(23)
    shape_frame = (12, 9)
    field = _field_smooth(shape_frame=shape_frame, rng=rng)
    ims = _dyadic_images(shape_frame, rng, n_images=4)
    ims_sparse = [scipy.sparse.csr_array(im.reshape(shape_frame)) for im in ims]

    out = helpers.remap_sparse_images(
        ims_sparse=ims_sparse,
        remappingIdx=field,
        method=interpolation_method,
        fill_value=0,
        dtype=np.float64,
        verbose=False,
        backend='operator',
    )
    assert isinstance(out, list)
    assert len(out) == len(ims_sparse)
    assert all(isinstance(im, scipy.sparse.csr_array) for im in out)
    assert all(tuple(im.shape) == shape_frame for im in out)

    remapper = helpers.Remapping_operator2d(remappingIdx=field, interpolation_method=interpolation_method, dtype=np.float64)
    expected = _to_dense(remapper(x=scipy.sparse.csr_array(ims), batching=True))
    assert np.array_equal(np.stack([_to_dense(im).reshape(-1) for im in out], axis=0), expected)

    ## A single sparse image in still gives a list of one out
    out_single = helpers.remap_sparse_images(
        ims_sparse=ims_sparse[0],
        remappingIdx=field,
        method=interpolation_method,
        dtype=np.float64,
        verbose=False,
        backend='operator',
    )
    assert isinstance(out_single, list) and (len(out_single) == 1)
    assert np.array_equal(_to_dense(out_single[0]), _to_dense(out[0]))


@pytest.mark.parametrize('kwargs_bad,type_error', [
    ({'method': 'cubic'}, ValueError),
    ({'method': 'quintic'}, ValueError),
    ({'fill_value': 1.0}, ValueError),
    ({'fill_value': np.nan}, ValueError),
])
def test_remap_sparse_images_operator_backend_rejects_what_it_cannot_honor(kwargs_bad, type_error):
    """``backend='operator'`` has no cubic kernel and no nonzero fill, so it says so."""
    shape_frame = (6, 5)
    field = _field_identity(shape_frame)
    kwargs = dict(method='linear', fill_value=0, dtype=np.float64, verbose=False, backend='operator')
    kwargs.update(kwargs_bad)
    with pytest.raises(type_error):
        helpers.remap_sparse_images(
            ims_sparse=[scipy.sparse.csr_array(np.eye(*shape_frame))],
            remappingIdx=field,
            **kwargs,
        )


def test_remap_sparse_images_unknown_backend_raises():
    """An unknown backend name raises rather than falling back to the default."""
    shape_frame = (6, 5)
    with pytest.raises(ValueError):
        helpers.remap_sparse_images(
            ims_sparse=[scipy.sparse.csr_array(np.eye(*shape_frame))],
            remappingIdx=_field_identity(shape_frame),
            verbose=False,
            backend='sparse',
        )


######################################################################################################################################
####################################################### ALIGNER-LEVEL ################################################################
######################################################################################################################################


def _rois_for_aligner(shape_frame: Tuple[int, int], rng: np.random.Generator, n_roi: int = 6, include_empty: bool = True):
    """
    A session's worth of ROIs in the *(n_roi, H*W)* layout ``transform_ROIs``
    expects, as float32 ``csr_matrix`` (what ``data_importing`` produces).
    """
    H, W = shape_frame
    rois = np.zeros((n_roi, H, W), dtype=np.float32)
    for ii in range(n_roi):
        r0 = int(rng.integers(2, H - 5))
        c0 = int(rng.integers(2, W - 5))
        rois[ii, r0:r0 + 4, c0:c0 + 4] = rng.random((4, 4)).astype(np.float32) + 0.1
    if include_empty:
        rois[-1] = 0.0
    return scipy.sparse.csr_matrix(rois.reshape(n_roi, H * W))


@pytest.fixture(scope='module')
def aligner_template():
    """
    One ``Aligner``, built once. ``ROICaT_Module.__init__`` calls
    ``util.system_info()``, which takes over a second, and nothing in these tests
    depends on it.
    """
    return alignment.Aligner(verbose=False)


@pytest.fixture
def aligner(aligner_template):
    """A fresh ``Aligner`` per test, so no test sees another's ``params``."""
    return copy.deepcopy(aligner_template)


def test_get_default_parameters_carries_method_warp():
    """The default parameter dict has to name the new argument, or pipelines cannot set it."""
    params = util.get_default_parameters()
    assert params['alignment']['transform_ROIs']['method_warp'] == 'linear'
    assert params['alignment']['transform_ROIs']['normalize'] is True


@pytest.mark.parametrize('method_warp', KERNELS)
def test_transform_ROIs_default_contract(method_warp, aligner):
    """
    One ``csr_array`` per session, float32, shape *(n_roi, H*W)*, rows in input
    order, nonzero rows summing to 1, and an all-zero ROI coming back as an
    all-zero row rather than a crash or a NaN.
    """
    rng = np.random.default_rng(24)
    shape_frame = (16, 13)
    rois = _rois_for_aligner(shape_frame=shape_frame, rng=rng, n_roi=6, include_empty=True)
    fields = [_field_smooth(shape_frame=shape_frame, rng=rng) for _ in range(2)]

    out = aligner.transform_ROIs(ROIs=[rois, rois], remappingIdx=fields, normalize=True, method_warp=method_warp)

    assert isinstance(out, list) and (len(out) == 2)
    for rois_aligned in out:
        assert isinstance(rois_aligned, scipy.sparse.csr_array)
        assert rois_aligned.dtype == np.float32
        assert rois_aligned.shape == (6, shape_frame[0] * shape_frame[1])
        assert np.isfinite(rois_aligned.data).all()
        assert (rois_aligned.data >= 0).all()
        nnz_rows = np.diff(rois_aligned.indptr)
        assert nnz_rows[-1] == 0, 'the all-zero ROI should come back as an all-zero row'
        sums = np.asarray(rois_aligned.sum(axis=1)).reshape(-1)
        assert np.allclose(sums[:-1], 1.0, atol=1e-6)
        assert sums[-1] == 0.0
    assert aligner.params['transform_ROIs']['method_warp'] == method_warp
    assert aligner.params['transform_ROIs']['normalize'] is True


def test_transform_ROIs_without_normalize_preserves_row_order(aligner):
    """Row order is the contract every downstream consumer relies on."""
    rng = np.random.default_rng(25)
    shape_frame = (16, 13)
    rois = _rois_for_aligner(shape_frame=shape_frame, rng=rng, n_roi=5, include_empty=False)
    field = _field_smooth(shape_frame=shape_frame, rng=rng)

    out = aligner.transform_ROIs(ROIs=[rois], remappingIdx=[field], normalize=False, method_warp='linear')[0]
    order = np.array([3, 0, 4, 1, 2])
    out_permuted = aligner.transform_ROIs(
        ROIs=[scipy.sparse.csr_matrix(np.asarray(rois.todense())[order])],
        remappingIdx=[field],
        normalize=False,
        method_warp='linear',
    )[0]
    assert np.array_equal(_to_dense(out_permuted), _to_dense(out)[order])


@pytest.mark.parametrize('method_warp', ['cubic', 'bilinear', 'legacy', '', None])
def test_transform_ROIs_invalid_method_warp_raises(method_warp, aligner):
    """An unknown ``method_warp`` raises ``ValueError`` before any work is done."""
    rng = np.random.default_rng(26)
    shape_frame = (12, 9)
    rois = _rois_for_aligner(shape_frame=shape_frame, rng=rng, n_roi=3, include_empty=False)
    with pytest.raises(ValueError):
        aligner.transform_ROIs(
            ROIs=[rois],
            remappingIdx=[_field_identity(shape_frame)],
            normalize=True,
            method_warp=method_warp,
        )


def test_transform_ROIs_legacy_path_warns_and_still_runs(aligner):
    """
    The legacy griddata path is kept so old results can be reproduced. It must
    warn, and it must return the same type and shape as the new path. The fixture
    carries no all-zero ROI: the legacy ``safe`` branch recurses until
    ``RecursionError`` on one (a pre-existing defect of that path, ROICaT #686
    report 2000 finding 6).
    """
    rng = np.random.default_rng(27)
    shape_frame = (14, 12)
    rois = _rois_for_aligner(shape_frame=shape_frame, rng=rng, n_roi=3, include_empty=False)
    field = _field_translate(shape_frame=shape_frame, shift_x=0.5, shift_y=-0.5)

    with pytest.warns(UserWarning, match='legacy'):
        out_legacy = aligner.transform_ROIs(ROIs=[rois], remappingIdx=[field], normalize=True, method_warp='legacy_griddata_cubic')
    assert aligner.params['transform_ROIs']['method_warp'] == 'legacy_griddata_cubic'

    out_new = aligner.transform_ROIs(ROIs=[rois], remappingIdx=[field], normalize=True, method_warp='linear')
    assert type(out_legacy[0]) is type(out_new[0])
    assert out_legacy[0].shape == out_new[0].shape
    assert out_legacy[0].dtype == out_new[0].dtype
    assert np.isfinite(out_legacy[0].data).all()


def test_transform_ROIs_does_not_keep_the_operator_on_the_aligner(aligner):
    """
    ``run_data`` serializes ``aligner.__dict__`` whole, so the operator must stay
    a local variable.
    """
    rng = np.random.default_rng(28)
    shape_frame = (12, 9)
    rois = _rois_for_aligner(shape_frame=shape_frame, rng=rng, n_roi=3, include_empty=False)
    aligner.transform_ROIs(ROIs=[rois], remappingIdx=[_field_identity(shape_frame)], normalize=True, method_warp='linear')
    assert not any(isinstance(v, helpers.Remapping_operator2d) for v in aligner.__dict__.values())
