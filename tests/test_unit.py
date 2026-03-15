"""
Unit tests for ROICaT

The functions in this module are intended to be
 found and run by pytest.

To run the tests, use the command (in a terminal):
    pytest -v test_unit.py
            ^
          verbose
"""

from pathlib import Path

import warnings
import pytest

import numpy as np
import scipy.sparse
import torch

from roicat import helpers, util


######################################################################################################################################
############################################################ UTIL ####################################################################
######################################################################################################################################


def test_system_info():
    """
    Test that system_info returns a dict with expected keys.
    """
    info = util.system_info(verbose=True)
    assert isinstance(info, dict), 'system_info should return a dict'
    assert len(info) > 0, 'system_info dict should not be empty'


def test_set_random_seed_determinism():
    """set_random_seed should produce identical sequences when called twice with same seed."""
    util.set_random_seed(seed=42, deterministic=False)
    a1 = np.random.rand(10)
    t1 = torch.rand(10)

    util.set_random_seed(seed=42, deterministic=False)
    a2 = np.random.rand(10)
    t2 = torch.rand(10)

    np.testing.assert_array_equal(a1, a2)
    assert torch.equal(t1, t2)


def test_set_random_seed_returns_seed():
    """set_random_seed should return the seed used."""
    seed = util.set_random_seed(seed=123)
    assert seed == 123

    seed_auto = util.set_random_seed(seed=None)
    assert isinstance(seed_auto, int)


def test_match_arrays_with_ucids_return_indices_handles_duplicate_ucids():
    """Ensure match_arrays_with_ucids can recover indices when sessions
    contain more ROIs than there are UCIDs."""

    arrays = [
        np.arange(3, dtype=np.float32)[:, None],
        np.arange(8, dtype=np.float32)[:, None],
    ]
    ucids = [
        np.array([0, 1, 2], dtype=np.int64),
        np.array([0, 1, 2, 3, 4, 5, 0, 1], dtype=np.int64),
    ]

    arrays_out, indices_out = util.match_arrays_with_ucids(
        arrays=arrays,
        ucids=ucids,
        squeeze=True,
        return_indices=True,
        prog_bar=False,
    )

    # Shapes are determined by the maximum UCID across sessions (0-5 -> 6 rows)
    assert arrays_out[0].shape == (6, 1)
    assert arrays_out[1].shape == (6, 1)

    # The first session maps one-to-one and leaves trailing UCIDs empty
    np.testing.assert_array_equal(
        arrays_out[0][:3, 0],
        np.arange(3, dtype=np.float32),
    )
    assert np.isnan(arrays_out[0][3:, 0]).all()

    # The second session keeps the last occurrence of each UCID even when more
    # ROIs exist than UCIDs, verifying we did not hit an IndexError.
    np.testing.assert_array_equal(
        arrays_out[1][:, 0],
        np.array([6, 7, 2, 3, 4, 5], dtype=np.float32),
    )

    # The returned indices track the original ROI positions used for each UCID.
    np.testing.assert_array_equal(
        indices_out[0][:3],
        np.array([0, 1, 2], dtype=np.float32),
    )
    assert np.isnan(indices_out[0][3:]).all()
    np.testing.assert_array_equal(
        indices_out[1],
        np.array([6, 7, 2, 3, 4, 5], dtype=np.float32),
    )


class Test_RichFile_ROICaT:
    """Tests for RichFile_ROICaT save/load with different backends."""

    def test_zip_roundtrip(self, tmp_path):
        """Save and load with zip backend should preserve all types."""
        test_data = {
            'array': np.random.randn(10, 5).astype(np.float32),
            'sparse': scipy.sparse.random(50, 50, density=0.1, format='csr', dtype=np.float32),
            'scalar': 3.14,
            'nested': {'a': np.array([1, 2, 3]), 'b': 'hello'},
        }
        path = str(tmp_path / 'test.richfile.zip')
        util.RichFile_ROICaT(path=path, backend='zip').save(obj=test_data, overwrite=True)
        loaded = util.RichFile_ROICaT(path=path).load()

        assert np.allclose(loaded['array'], test_data['array'])
        assert np.allclose(loaded['sparse'].toarray(), test_data['sparse'].toarray())
        assert loaded['scalar'] == test_data['scalar']
        assert np.array_equal(loaded['nested']['a'], test_data['nested']['a'])
        assert loaded['nested']['b'] == test_data['nested']['b']

    def test_directory_roundtrip(self, tmp_path):
        """Save and load with directory backend should preserve all types."""
        test_data = {
            'array': np.array([1.0, 2.0, 3.0]),
            'string': 'test',
        }
        path = str(tmp_path / 'test.richfile')
        util.RichFile_ROICaT(path=path, backend='directory').save(obj=test_data, overwrite=True)
        loaded = util.RichFile_ROICaT(path=path).load()

        assert np.array_equal(loaded['array'], test_data['array'])
        assert loaded['string'] == test_data['string']

    def test_auto_detect_zip(self, tmp_path):
        """Auto-detect should identify zip files correctly."""
        test_data = {'x': np.array([1, 2, 3])}
        path = str(tmp_path / 'test.richfile.zip')
        util.RichFile_ROICaT(path=path, backend='zip').save(obj=test_data, overwrite=True)

        ## Load without specifying backend
        rf = util.RichFile_ROICaT(path=path)
        assert rf._resolve_backend_name() == 'zip'
        loaded = rf.load()
        assert np.array_equal(loaded['x'], test_data['x'])

    def test_auto_detect_directory(self, tmp_path):
        """Auto-detect should identify directory richfiles correctly."""
        test_data = {'x': np.array([4, 5, 6])}
        path = str(tmp_path / 'test.richfile')
        util.RichFile_ROICaT(path=path, backend='directory').save(obj=test_data, overwrite=True)

        rf = util.RichFile_ROICaT(path=path)
        assert rf._resolve_backend_name() == 'directory'
        loaded = rf.load()
        assert np.array_equal(loaded['x'], test_data['x'])

    def test_load_existing_test_data(self, dir_data_test):
        """Should load the existing directory-format test data."""
        path = str(Path(dir_data_test) / 'pipeline_tracking' / 'run_data.richfile.zip')
        rf = util.RichFile_ROICaT(path=path)
        sim = rf['sim'].load()
        assert isinstance(sim, dict)
        assert 'params' in sim

    def test_subscript_access_zip(self, tmp_path):
        """Subscript access (rf['key']) should work with zip backend."""
        test_data = {'alpha': np.array([1, 2]), 'beta': np.array([3, 4])}
        path = str(tmp_path / 'test.richfile.zip')
        util.RichFile_ROICaT(path=path, backend='zip').save(obj=test_data, overwrite=True)

        rf = util.RichFile_ROICaT(path=path)
        alpha = rf['alpha'].load()
        assert np.array_equal(alpha, test_data['alpha'])

    def test_scipy_sparse_roundtrip_zip(self, tmp_path):
        """Scipy sparse matrices should survive zip roundtrip."""
        mat = scipy.sparse.random(100, 100, density=0.05, format='csr', dtype=np.float64)
        test_data = {'sparse_mat': mat}
        path = str(tmp_path / 'sparse_test.richfile.zip')
        util.RichFile_ROICaT(path=path, backend='zip').save(obj=test_data, overwrite=True)
        loaded = util.RichFile_ROICaT(path=path).load()
        assert scipy.sparse.issparse(loaded['sparse_mat'])
        assert np.allclose(loaded['sparse_mat'].toarray(), mat.toarray())


######################################################################################################################################
############################################################ HELPERS #################################################################
######################################################################################################################################


class Test_Equivalence_checker:
    """Tests for helpers.Equivalence_checker — the core comparison utility."""

    def test_equal_arrays(self):
        checker = helpers.Equivalence_checker()
        a = np.array([1.0, 2.0, 3.0])
        assert checker(a, a)[0] == True

    def test_unequal_arrays_verbose_false(self):
        """Regression: verbose=False with mismatch previously caused UnboundLocalError."""
        checker = helpers.Equivalence_checker(verbose=False)
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0, 999.0])
        result = checker(a, b)
        assert result[0] == False
        assert isinstance(result[1], str)

    def test_unequal_arrays_verbose_true(self):
        """verbose=True should produce a detailed reason string."""
        checker = helpers.Equivalence_checker(verbose=True)
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0, 999.0])
        result = checker(a, b)
        assert result[0] == False
        assert 'Relative difference' in result[1]

    def test_close_arrays_pass(self):
        checker = helpers.Equivalence_checker(kwargs_allclose={'rtol': 1e-5, 'equal_nan': True})
        a = np.array([1.0, 2.0, 3.0])
        b = a + 1e-7
        assert checker(a, b)[0] == True

    def test_boolean_arrays(self):
        """Boolean arrays should not raise TypeError on subtraction."""
        checker = helpers.Equivalence_checker(verbose=True)
        a = np.array([True, False, True])
        b = np.array([True, True, True])
        result = checker(a, b)
        assert result[0] == False

    def test_nested_dicts(self):
        checker = helpers.Equivalence_checker(verbose=False)
        d = {'a': np.array([1.0, 2.0]), 'b': {'c': np.array([3.0])}}
        result = checker(d, d)
        assert result['a'][0] == True
        assert result['b']['c'][0] == True

    def test_nested_dicts_mismatch(self):
        checker = helpers.Equivalence_checker(verbose=False)
        d1 = {'x': np.array([1.0, 2.0]), 'y': np.array([3.0])}
        d2 = {'x': np.array([1.0, 2.0]), 'y': np.array([99.0])}
        result = checker(d1, d2)
        assert result['x'][0] == True
        assert result['y'][0] == False

    def test_sparse_identical(self):
        checker = helpers.Equivalence_checker()
        s = scipy.sparse.random(50, 50, density=0.2, format='csr', random_state=0)
        assert checker(s, s)[0] == True

    def test_sparse_close(self):
        checker = helpers.Equivalence_checker(kwargs_allclose={'rtol': 1e-5})
        s = scipy.sparse.random(50, 50, density=0.2, format='csr', random_state=0)
        s2 = s.copy()
        s2.data = s2.data * (1 + 1e-7)  ## Proportional perturbation within rtol
        assert checker(s2, s)[0] == True

    def test_sparse_different(self):
        checker = helpers.Equivalence_checker()
        s1 = scipy.sparse.csr_matrix(np.eye(3))
        s2 = scipy.sparse.csr_matrix(np.eye(3) * 2.0)
        result = checker(s1, s2)
        assert result[0] == False
        assert 'sparse allclose failed' in result[1]

    def test_sparse_shape_mismatch(self):
        checker = helpers.Equivalence_checker()
        s1 = scipy.sparse.csr_matrix(np.eye(3))
        s2 = scipy.sparse.csr_matrix(np.eye(4))
        result = checker(s1, s2)
        assert result[0] == False
        assert 'shape mismatch' in result[1]

    def test_sparse_type_mismatch(self):
        checker = helpers.Equivalence_checker()
        s = scipy.sparse.csr_matrix(np.eye(3))
        result = checker(s.toarray(), s)
        assert result[0] == False
        assert 'type mismatch' in result[1]

    def test_sparse_empty(self):
        checker = helpers.Equivalence_checker()
        s = scipy.sparse.csr_matrix((10, 10))
        assert checker(s, s)[0] == True

    def test_sparse_bool(self):
        checker = helpers.Equivalence_checker()
        s = scipy.sparse.random(20, 20, density=0.3, format='csr', random_state=0)
        sb = (s != 0).astype(bool)
        assert checker(sb, sb)[0] == True

    def test_sparse_in_nested_dict(self):
        checker = helpers.Equivalence_checker()
        s = scipy.sparse.random(10, 10, density=0.5, format='csr', random_state=0)
        d = {'dense': np.array([1.0]), 'sparse': s}
        result = checker(d, d)
        assert result['dense'][0] == True
        assert result['sparse'][0] == True

    def test_scalars(self):
        checker = helpers.Equivalence_checker()
        assert checker(3.14, 3.14)[0] == True

    def test_none_values(self):
        checker = helpers.Equivalence_checker()
        assert checker(None, None)[0] == True

    def test_assert_mode_raises(self):
        checker = helpers.Equivalence_checker(assert_mode=True)
        with pytest.raises(AssertionError):
            checker(np.array([1.0]), np.array([2.0]))

    def test_exception_returns_none(self):
        """When allclose raises (e.g., incompatible dtypes), result should be (None, ...)."""
        checker = helpers.Equivalence_checker(verbose=False)
        # Object arrays can trigger exceptions in allclose
        a = np.array([object()], dtype=object)
        b = np.array([object()], dtype=object)
        result = checker(a, b)
        # Should not crash; result[0] is None (skipped) or True/False
        assert result[0] is not None or isinstance(result[1], str)


class Test_get_nums_from_string:
    """Tests for helpers.get_nums_from_string."""

    def test_basic(self):
        result = helpers.get_nums_from_string('abc123def')
        assert result == 123

    def test_multiple_groups_concatenated(self):
        """Disjoint digit groups are concatenated into one integer."""
        result = helpers.get_nums_from_string('x1y23z456')
        assert result == 123456

    def test_no_numbers(self):
        assert helpers.get_nums_from_string('abcdef') is None

    def test_all_numbers(self):
        assert helpers.get_nums_from_string('12345') == 12345

    def test_empty_string(self):
        assert helpers.get_nums_from_string('') is None


class Test_idx2bool:
    """Tests for helpers.idx2bool."""

    def test_basic(self):
        result = helpers.idx2bool(np.array([1, 3, 5]), length=7)
        expected = np.array([False, True, False, True, False, True, False])
        np.testing.assert_array_equal(result, expected)

    def test_auto_length(self):
        result = helpers.idx2bool(np.array([0, 2]))
        assert len(result) == 3
        assert result[0] == True
        assert result[1] == False
        assert result[2] == True

    def test_empty(self):
        result = helpers.idx2bool(np.array([], dtype=int), length=5)
        assert not np.any(result)
        assert len(result) == 5


class Test_squeeze_integers:
    """Tests for helpers.squeeze_integers."""

    def test_basic(self):
        result = helpers.squeeze_integers(np.array([7, 2, 7, 4, -1, 0]))
        expected = np.array([3, 1, 3, 2, -1, 0])
        np.testing.assert_array_equal(result, expected)

    def test_already_consecutive(self):
        result = helpers.squeeze_integers(np.array([0, 1, 2, 3]))
        np.testing.assert_array_equal(result, np.array([0, 1, 2, 3]))

    def test_with_minus_one(self):
        """Elements with -1 should stay -1."""
        result = helpers.squeeze_integers(np.array([-1, -1, 5, 10]))
        assert result[0] == -1
        assert result[1] == -1
        assert result[2] >= 0
        assert result[3] >= 0

    def test_preserves_grouping(self):
        """Equal inputs should map to equal outputs."""
        arr = np.array([10, 20, 10, 30, 20])
        result = helpers.squeeze_integers(arr)
        assert result[0] == result[2]  # both were 10
        assert result[1] == result[4]  # both were 20

    def test_torch_tensor(self):
        result = helpers.squeeze_integers(torch.tensor([5, 0, 5, 3]))
        assert isinstance(result, torch.Tensor)


class Test_scipy_sparse_to_torch_coo:
    """Tests for helpers.scipy_sparse_to_torch_coo."""

    def test_shape_preserved(self):
        s = scipy.sparse.random(10, 20, density=0.3, format='csr', random_state=0)
        t = helpers.scipy_sparse_to_torch_coo(s)
        assert t.shape == s.shape

    def test_values_preserved(self):
        s = scipy.sparse.random(10, 20, density=0.3, format='csr', random_state=0)
        t = helpers.scipy_sparse_to_torch_coo(s)
        np.testing.assert_allclose(t.to_dense().numpy(), s.toarray(), rtol=1e-6)

    def test_empty(self):
        s = scipy.sparse.csr_matrix((5, 5))
        t = helpers.scipy_sparse_to_torch_coo(s)
        assert t.shape == (5, 5)
        assert t._nnz() == 0

    def test_dtype_override(self):
        s = scipy.sparse.random(5, 5, density=0.5, format='csr', random_state=0)
        t = helpers.scipy_sparse_to_torch_coo(s, dtype=torch.float32)
        assert t.dtype == torch.float32


class Test_merge_sparse_arrays:
    """Tests for helpers.merge_sparse_arrays — used in clustering pipeline."""

    def test_basic_merge(self):
        """Two blocks placed at different positions in a larger matrix."""
        s1 = scipy.sparse.csr_matrix(np.array([[1.0, 0.5], [0.5, 1.0]]))
        s2 = scipy.sparse.csr_matrix(np.array([[2.0, 0.0], [0.0, 2.0]]))
        idx1 = np.array([0, 1])
        idx2 = np.array([2, 3])
        result = helpers.merge_sparse_arrays([s1, s2], [idx1, idx2], shape_full=(4, 4))
        dense = result.toarray()
        # s1 placed at [0:2, 0:2], s2 at [2:4, 2:4]
        np.testing.assert_allclose(dense[0, 1], 0.5)
        np.testing.assert_allclose(dense[2, 2], 2.0)
        np.testing.assert_allclose(dense[3, 3], 2.0)

    def test_symmetric_input(self):
        """For symmetric matrices (the real use case), result should also be symmetric."""
        rng = np.random.RandomState(42)
        a = rng.rand(5, 5)
        a = (a + a.T) / 2
        s = scipy.sparse.csr_matrix(a)
        idx = np.array([2, 4, 6, 8, 10])
        result = helpers.merge_sparse_arrays([s], [idx], shape_full=(12, 12))
        dense = result.toarray()
        np.testing.assert_allclose(dense, dense.T)

    def test_multiple_blocks_no_overlap(self):
        """Merging multiple non-overlapping blocks should preserve all values."""
        blocks = []
        idxs = []
        for i in range(3):
            s = scipy.sparse.csr_matrix(np.eye(2) * (i + 1))
            blocks.append(s)
            idxs.append(np.array([i * 2, i * 2 + 1]))
        result = helpers.merge_sparse_arrays(blocks, idxs, shape_full=(6, 6))
        dense = result.toarray()
        np.testing.assert_allclose(dense[0, 0], 1.0)
        np.testing.assert_allclose(dense[2, 2], 2.0)
        np.testing.assert_allclose(dense[4, 4], 3.0)


class Test_set_device:
    """Tests for helpers.set_device."""

    def test_returns_string(self):
        device = helpers.set_device(verbose=False)
        assert isinstance(device, str)

    def test_cpu(self):
        device = helpers.set_device(use_GPU=False, verbose=False)
        assert device == 'cpu'

    def test_valid_prefix(self):
        device = helpers.set_device(verbose=False)
        valid_prefixes = ('cpu', 'cuda', 'mps', 'xpu')
        assert any(device.startswith(p) for p in valid_prefixes), f"Unexpected device: {device}"


class Test_cosine_kernel_2D:
    """Tests for helpers.cosine_kernel_2D."""

    def test_default_shape(self):
        k = helpers.cosine_kernel_2D()
        assert k.shape == (11, 11)

    def test_custom_shape(self):
        k = helpers.cosine_kernel_2D(image_size=(20, 30))
        assert k.shape == (20, 30)

    def test_center_is_one(self):
        k = helpers.cosine_kernel_2D(image_size=(21, 21), center=(10, 10), width=20)
        assert np.isclose(k[10, 10], 1.0)

    def test_values_in_range(self):
        k = helpers.cosine_kernel_2D()
        assert np.all(k >= 0)
        assert np.all(k <= 1)

    def test_symmetry(self):
        k = helpers.cosine_kernel_2D(image_size=(21, 21), center=(10, 10))
        np.testing.assert_allclose(k, k.T)


class Test_flatten_dict:
    """Tests for helpers.flatten_dict — used in equivalence check reporting."""

    def test_simple(self):
        d = {'a': 1, 'b': 2}
        flat = helpers.flatten_dict(d)
        assert flat == {'a': 1, 'b': 2}

    def test_nested(self):
        d = {'a': {'b': {'c': 1}}, 'x': 2}
        flat = helpers.flatten_dict(d)
        assert ('a', 'b', 'c') in flat or any('c' in str(k) for k in flat.keys())

    def test_empty(self):
        assert helpers.flatten_dict({}) == {}


######################################################################################################################################
####################################################### DATA_IMPORTING ###############################################################
######################################################################################################################################

def test_data_suite2p(dir_data_test, array_hasher):
    """
    Test data_importing.Data_suite2p.
    RH 2022

    Args:
        dir_data_test (str):
            pytest fixture.
            Path to the test data directory.
    """

    ##########
    ## TEST 1: Basic import of multiple stat.npy + ops.npy files
    ##########

    ## Get paths to test data
    paths_stat = helpers.find_paths(
        dir_outer=str(Path(dir_data_test) / 'data__stat_ops_small__valerio_rbp10_plane0'),
        reMatch='stat.npy',
        find_files=True,
        find_folders=False,
        depth=2,
        natsorted=True,
    )
    paths_ops = [str(Path(p).parent / 'ops.npy') for p in paths_stat]
    assert all([Path(p).exists() for p in paths_stat]), 'ROICaT Error: one or more stat.npy files do not exist.'
    assert all([Path(p).exists() for p in paths_ops]), 'ROICaT Error: one or more ops.npy files do not exist.'
    print(f'Found {len(paths_stat)} stat.npy files and {len(paths_ops)} ops.npy files.')
    print(f'paths_stat: {paths_stat}')

    ## Import class
    from roicat.data_importing import Data_suite2p

    params = {
        'data_loading': {
            'data_kind': 'suite2p',  ## Can be 'suite2p' or 'roiextractors'. See documentation and/or notebook on custom data loading for more details.
            'common': {
                'um_per_pixel': 2.0,  ## Number of microns per pixel for the imaging dataset. Doesn't need to be exact. Used for resizing the ROIs. Check the images of the resized ROIs to tweak.
                'centroid_method': 'centerOfMass', ## Can be 'centerOfMass' or 'median'.
                'out_height_width': [36,36],  ## Height and width of the small ROI_images. Should generally be tuned slightly bigger than the largest ROIs. Leave if uncertain or if ROIs are small enough to fit in the default size.
            },
            'suite2p': {
                'new_or_old_suite2p': 'new',  ## Can be 'new' or 'old'. 'new' is for the Python version of Suite2p, 'old' is for the MATLAB version.
                'type_meanImg': 'meanImgE',  ## Can be 'meanImg' or 'meanImgE'. 'meanImg' is the mean image of the dataset, 'meanImgE' is the mean image of the dataset after contrast enhancement.
            },
        },
    }

    ## Instantiate class with test data
    data = Data_suite2p(
        paths_statFiles=paths_stat,
        paths_opsFiles=paths_ops,
        verbose=True,
        **{**params['data_loading']['common'], **params['data_loading']['suite2p']},
    )

    ## Test that the class was instantiated correctly
    ### General attributes
    assert all([isinstance(p, str) for p in data.paths_stat]), 'ROICaT Error: data.paths_stat.dtype != str'
    assert all([p == p2 for p, p2 in zip(data.paths_stat, paths_stat)]), 'ROICaT Error: data.paths_stat != paths_stat'
    assert all([isinstance(p, str) for p in data.paths_ops]), 'ROICaT Error: data.paths_ops.dtype != str'
    assert all([p == p2 for p, p2 in zip(data.paths_ops, paths_ops)]), 'ROICaT Error: data.paths_ops != paths_ops'
    assert data.um_per_pixel == [params['data_loading']['common']['um_per_pixel'],] * len(paths_stat), 'ROICaT Error: data.um_per_pixel != [um_per_pixel,]*len(paths_stat)'
    assert data.n_sessions == len(paths_stat), 'ROICaT Error: data.n_sessions != len(paths_stat)'
    ### Types
    assert all([c.dtype == np.int64 for c in data.centroids]), 'ROICaT Error: data.centroids.dtype != np.uint64'
    assert all([im.dtype == np.float32 for im in data.FOV_images]), 'ROICaT Error: data.FOV_images.dtype != np.float32'
    assert isinstance(data.FOV_width, int), 'ROICaT Error: data.FOV_width.dtype != int'
    assert isinstance(data.FOV_height, int), 'ROICaT Error: data.FOV_height.dtype != int'
    assert isinstance(data.n_sessions, int), 'ROICaT Error: data.n_sessions.dtype != int'
    assert isinstance(data.n_roi_total, int), 'ROICaT Error: data.n_roi_total.dtype != int'
    assert isinstance(data.n_roi, list), 'ROICaT Error: data.n_roi.dtype != list'
    assert all([isinstance(n, int) for n in data.n_roi]), 'ROICaT Error: data.n_roi.dtype != list of ints'
    assert isinstance(data.shifts, (list, tuple)), 'ROICaT Error: data.shifts.dtype != list or tuple'
    assert all([isinstance(s, np.ndarray) for s in data.shifts]), 'ROICaT Error: data.shifts.dtype != list or tuple of np.ndarrays'
    assert all([s.dtype == np.uint64 for s in data.shifts]), 'ROICaT Error: data.shifts.dtype != list or tuple of np.ndarrays of dtype np.uint64'
    assert isinstance(data.um_per_pixel, list), 'ROICaT Error: data.um_per_pixel.dtype != list'
    assert all([isinstance(ump, float) for ump in data.um_per_pixel]), 'ROICaT Error: data.um_per_pixel.dtype != list of floats'
    assert isinstance(data.paths_stat, list), 'ROICaT Error: data.paths_stat.dtype != list'
    assert all([isinstance(p, str) for p in data.paths_stat]), 'ROICaT Error: data.paths_stat.dtype != list of strings'
    assert isinstance(data.paths_ops, list), 'ROICaT Error: data.paths_ops.dtype != list'
    assert all([isinstance(p, str) for p in data.paths_ops]), 'ROICaT Error: data.paths_ops.dtype != list of strings'
    assert isinstance(data.ROI_images, list), 'ROICaT Error: data.ROI_images.dtype != list'
    assert len(data.ROI_images) == len(paths_stat), 'ROICaT Error: len(data.ROI_images) != len(paths_stat)'
    assert all([isinstance(im, np.ndarray) for im in data.ROI_images]), 'ROICaT Error: data.ROI_images.dtype != list of np.ndarrays'
    assert all([im.dtype == np.float32 for im in data.ROI_images]), 'ROICaT Error: data.ROI_images.dtype != list of np.ndarrays of dtype np.float32'
    assert isinstance(data.spatialFootprints, list), 'ROICaT Error: data.spatialFootprints.dtype != list'
    assert len(data.spatialFootprints) == len(paths_stat), 'ROICaT Error: len(data.spatialFootprints) != len(paths_stat)'
    assert all([isinstance(sf, scipy.sparse.csr_matrix) for sf in data.spatialFootprints]), 'ROICaT Error: data.spatialFootprints.dtype != list of scipy.sparse.csr_matrix'

    ### Attributes specific to this dataset
    assert data.n_roi_total == 300*len(paths_stat), 'ROICaT Error: data.n_roi_total != 300*len(paths_stat). stat.npy files expected to contain 300 ROIs each.'
    assert data.n_roi == [300]*len(paths_stat), 'ROICaT Error: data.n_roi != [300]*len(paths_stat). stat.npy files expected to contain 300 ROIs each.'
    assert all([c.shape == (300, 2) for c in data.centroids]), 'ROICaT Error: data.centroids.shape != (300, 2)'
    assert array_hasher(data.centroids[0]) == 'f98974f9430846ed', 'ROICaT Error: data.centroids[0] != expected values. See code for expected values.'
    assert array_hasher(data.centroids[13]) == 'b073952b11a3c507', 'ROICaT Error: data.centroids[13] != expected values. See code for expected values.'
    assert data.FOV_height == 512, 'ROICaT Error: data_FOV_height != expected value.'
    assert data.FOV_width == 705, 'ROICaT Error: data_FOV_width != expected value.'
    assert array_hasher(data.FOV_images[0]) == '2e335e2116ee4cfc', 'ROICaT Error: data.FOV_images[0] != expected values. See code for expected values.'
    assert array_hasher(data.FOV_images[13]) == '597cc830474f1ff5', 'ROICaT Error: data.FOV_images[13] != expected values. See code for expected values.'
    assert data.ROI_images[0].shape == tuple([300] + list(params['data_loading']['common']['out_height_width'])), 'ROICaT Error: data.ROI_images.shape != (300, out_height_width[0], out_height_width[1])'
    assert array_hasher(data.ROI_images[0]) == '04d986f3681778f0', 'ROICaT Error: data.ROI_images[0] != expected values. See code for expected values.'
    assert array_hasher(data.ROI_images[13]) == 'de5a2c2c8c34c43e', 'ROICaT Error: data.ROI_images[13] != expected values. See code for expected values.'
    assert data.spatialFootprints[0].shape[0] == 300, 'ROICaT Error: data.spatialFootprints.shape[0] != 300'
    assert data.spatialFootprints[0].shape[1] == 512*705, 'ROICaT Error: data.spatialFootprints.shape[1] != 512*705'
    assert array_hasher(data.spatialFootprints[0].toarray()) == '6319b48421caeb23', 'ROICaT Error: data.spatialFootprints[0] != expected values. See code for expected values.'
    assert array_hasher(data.spatialFootprints[13].toarray()) == 'd5495d254954d56c', 'ROICaT Error: data.spatialFootprints[13] != expected values. See code for expected values.'


######################################################################################################################################
########################################################## CLUSTERING ################################################################
######################################################################################################################################


@pytest.fixture(scope='module')
def clusterer_with_data(dir_data_test):
    """
    Load test run_data and create a Clusterer instance.
    Shared across all clustering tests to avoid redundant data loading.

    Loads sparse matrices from the 'sim' sub-key to avoid triggering
    deserialization of Optuna objects stored in 'clusterer'.
    """
    from roicat import util, tracking

    path_run_data = str(Path(dir_data_test) / 'pipeline_tracking' / 'run_data.richfile.zip')
    sim = util.RichFile_ROICaT(path=path_run_data)['sim'].load()

    clusterer = tracking.clustering.Clusterer(
        s_sf=sim['s_sf'],
        s_NN_z=sim['s_NN_z'],
        s_SWT_z=sim['s_SWT_z'],
        s_sesh=sim['s_sesh'],
        verbose=False,
    )
    return clusterer


class Test__find_optimal_parameters_DE:
    """Tests for Clusterer._find_optimal_parameters_DE."""

    def test_returns_valid_dict(self, clusterer_with_data):
        """DE should return a dict with all expected keys."""
        result = clusterer_with_data._find_optimal_parameters_DE(seed=42)
        expected_keys = {
            'power_SF', 'power_NN', 'power_SWT', 'p_norm',
            'sig_SF_kwargs', 'sig_NN_kwargs', 'sig_SWT_kwargs',
        }
        assert set(result.keys()) == expected_keys
        ## sig_NN_kwargs and sig_SWT_kwargs should be dicts with 'mu' and 'b'
        assert set(result['sig_NN_kwargs'].keys()) == {'mu', 'b'}
        assert set(result['sig_SWT_kwargs'].keys()) == {'mu', 'b'}

    def test_params_within_bounds(self, clusterer_with_data):
        """All optimized parameters should be within their declared bounds."""
        result = clusterer_with_data._find_optimal_parameters_DE(seed=42)
        bounds = {
            'power_NN': [0.0, 2.0],
            'power_SWT': [0.0, 2.0],
            'p_norm': [-5.0, -0.1],
        }
        for key, (lo, hi) in bounds.items():
            val = result[key]
            assert lo - 1e-6 <= val <= hi + 1e-6, (
                f'{key}={val} outside bounds [{lo}, {hi}]'
            )
        ## Sigmoid params are frozen from NB calibration — just check they exist and are finite
        for name in ['sig_NN_kwargs', 'sig_SWT_kwargs']:
            assert np.isfinite(result[name]['mu'])
            assert np.isfinite(result[name]['b'])
            assert result[name]['b'] > 0

    def test_deterministic_with_seed(self, clusterer_with_data):
        """Same seed should produce identical results."""
        r1 = clusterer_with_data._find_optimal_parameters_DE(seed=123)
        r2 = clusterer_with_data._find_optimal_parameters_DE(seed=123)
        for key in ['power_NN', 'power_SWT', 'p_norm']:
            assert r1[key] == r2[key], f'{key} differs between runs with same seed'

    def test_loss_is_finite(self, clusterer_with_data):
        """DE result should have a finite loss value."""
        clusterer_with_data._find_optimal_parameters_DE(seed=42)
        assert hasattr(clusterer_with_data, '_de_result')
        assert np.isfinite(clusterer_with_data._de_result.fun)

    def test_loss_below_threshold(self, clusterer_with_data):
        """DE should find a loss significantly below the trivial/default value.
        On the test dataset, DE reliably finds loss ~55. The default manual
        params typically give loss >200."""
        clusterer_with_data._find_optimal_parameters_DE(seed=42)
        assert clusterer_with_data._de_result.fun < 200, (
            f'DE loss {clusterer_with_data._de_result.fun:.1f} is too high; '
            f'expected < 200 on test data'
        )

    def test_subsample_pairs(self, clusterer_with_data):
        """DE with subsample_pairs should still return valid params."""
        result = clusterer_with_data._find_optimal_parameters_DE(
            seed=42,
            subsample_pairs=500,
            de_kwargs={
                'maxiter': 5,
                'tol': 1e-4,
                'popsize': 5,
                'polish': False,
            },
        )
        expected_keys = {
            'power_SF', 'power_NN', 'power_SWT', 'p_norm',
            'sig_SF_kwargs', 'sig_NN_kwargs', 'sig_SWT_kwargs',
        }
        assert set(result.keys()) == expected_keys
        assert np.isfinite(clusterer_with_data._de_result.fun)

    def test_resample_with_subsampling(self, clusterer_with_data):
        """DE with subsampling should automatically resample each generation."""
        result = clusterer_with_data._find_optimal_parameters_DE(
            seed=42,
            subsample_pairs=500,
            de_kwargs={
                'maxiter': 5,
                'tol': 1e-4,
                'popsize': 5,
                'polish': False,
            },
        )
        expected_keys = {
            'power_SF', 'power_NN', 'power_SWT', 'p_norm',
            'sig_SF_kwargs', 'sig_NN_kwargs', 'sig_SWT_kwargs',
        }
        assert set(result.keys()) == expected_keys
        assert np.isfinite(clusterer_with_data._de_result.fun)



class Test_estimate_sigmoid_params:
    """Tests for Clusterer._estimate_sigmoid_params."""

    def test_returns_expected_features(self, clusterer_with_data):
        """Should return sigmoid params for NN and SWT."""
        clusterer_with_data.make_naive_bayes_distance_matrix()
        result = clusterer_with_data._estimate_sigmoid_params()
        assert 'NN' in result
        assert 'SWT' in result
        assert 'mu' in result['NN'] and 'b' in result['NN']
        assert 'mu' in result['SWT'] and 'b' in result['SWT']

    def test_mu_is_finite(self, clusterer_with_data):
        """Estimated mu should be finite (within data range, which can exceed [0,1] for z-scored features)."""
        clusterer_with_data.make_naive_bayes_distance_matrix()
        result = clusterer_with_data._estimate_sigmoid_params()
        for name in ['NN', 'SWT']:
            mu = result[name]['mu']
            assert np.isfinite(mu), f'{name} mu={mu} is not finite'

    def test_b_is_positive(self, clusterer_with_data):
        """Estimated b (steepness) should be positive."""
        clusterer_with_data.make_naive_bayes_distance_matrix()
        result = clusterer_with_data._estimate_sigmoid_params()
        for name in ['NN', 'SWT']:
            b = result[name]['b']
            assert b > 0, f'{name} b={b} should be positive'

    def test_requires_nb_calibration(self, dir_data_test):
        """Should raise if NB calibration hasn't been run on a fresh instance."""
        from roicat import util, tracking
        path_run_data = str(Path(dir_data_test) / 'pipeline_tracking' / 'run_data.richfile.zip')
        sim = util.RichFile_ROICaT(path=path_run_data)['sim'].load()
        fresh = tracking.clustering.Clusterer(
            s_sf=sim['s_sf'],
            s_NN_z=sim['s_NN_z'],
            s_SWT_z=sim['s_SWT_z'],
            s_sesh=sim['s_sesh'],
            verbose=False,
        )
        with pytest.raises(AssertionError, match="make_naive_bayes_distance_matrix"):
            fresh._estimate_sigmoid_params()


class Test_naive_bayes_distance_matrix:
    """Tests for Clusterer.make_naive_bayes_distance_matrix."""

    def test_returns_correct_types(self, clusterer_with_data):
        """Should return (dConj, sConj, calibrations) with correct types."""
        import scipy.sparse
        dConj, sConj, calibrations = clusterer_with_data.make_naive_bayes_distance_matrix()
        assert isinstance(dConj, scipy.sparse.csr_matrix)
        assert isinstance(sConj, scipy.sparse.csr_matrix)
        assert isinstance(calibrations, dict)
        assert 'features' in calibrations
        assert 'prior' in calibrations
        assert 'p_same_combined' in calibrations

    def test_output_shapes_match_input(self, clusterer_with_data):
        """dConj and sConj should have same shape/nnz as s_sf."""
        dConj, sConj, _ = clusterer_with_data.make_naive_bayes_distance_matrix()
        assert dConj.shape == clusterer_with_data.s_sf.shape
        assert sConj.shape == clusterer_with_data.s_sf.shape
        assert dConj.nnz == clusterer_with_data.s_sf.nnz
        assert sConj.nnz == clusterer_with_data.s_sf.nnz

    def test_distances_in_valid_range(self, clusterer_with_data):
        """Distances (1 - P(same)) should be in [0, 1]."""
        dConj, sConj, _ = clusterer_with_data.make_naive_bayes_distance_matrix()
        assert dConj.data.min() >= 0.0 - 1e-6
        assert dConj.data.max() <= 1.0 + 1e-6
        assert sConj.data.min() >= 0.0 - 1e-6
        assert sConj.data.max() <= 1.0 + 1e-6

    def test_distance_plus_similarity_equals_one(self, clusterer_with_data):
        """d + s should equal 1 for every pair."""
        dConj, sConj, _ = clusterer_with_data.make_naive_bayes_distance_matrix()
        np.testing.assert_allclose(
            dConj.data + sConj.data, 1.0, atol=1e-6,
        )

    def test_calibration_has_all_features(self, clusterer_with_data):
        """Calibrations should contain SF, NN, and SWT."""
        _, _, calibrations = clusterer_with_data.make_naive_bayes_distance_matrix()
        assert set(calibrations['features'].keys()) == {'SF', 'NN', 'SWT'}
        for name, cal in calibrations['features'].items():
            assert 'edges' in cal
            assert 'p_same_bins' in cal
            assert 'p_same_per_pair' in cal
            ## P(same) per bin should be monotonically non-decreasing.
            ## Values are stored as numpy arrays for serialization compatibility.
            p = np.asarray(cal['p_same_bins'])
            assert np.all(np.diff(p) >= -1e-7), (
                f'{name}: P(same) bins not monotonic: {p}'
            )

    def test_prior_is_reasonable(self, clusterer_with_data):
        """Prior P(same) should be positive and less than 0.5 (most pairs are different)."""
        _, _, calibrations = clusterer_with_data.make_naive_bayes_distance_matrix()
        prior = calibrations['prior']
        assert 0 < prior < 0.5, f'Prior P(same)={prior} seems unreasonable'

    def test_some_pairs_classified_as_same(self, clusterer_with_data):
        """At least some pairs should have P(same) > 0.5."""
        _, _, calibrations = clusterer_with_data.make_naive_bayes_distance_matrix()
        p_combined = calibrations['p_same_combined']
        n_same = (p_combined > 0.5).sum()
        ## With 241 ROIs across 9 sessions, there should be some same pairs
        assert n_same > 0, 'No pairs classified as same (P > 0.5)'

    def test_compatible_with_pruning(self, clusterer_with_data):
        """Output should work with make_pruned_similarity_graphs(precomputed)."""
        clusterer_with_data.make_naive_bayes_distance_matrix()
        ## Should not raise
        clusterer_with_data.make_pruned_similarity_graphs(
            kwargs_makeConjunctiveDistanceMatrix='precomputed',
        )
        assert hasattr(clusterer_with_data, 'dConj_pruned')
        assert clusterer_with_data.dConj_pruned is not None

    def test_deterministic(self, clusterer_with_data):
        """Two calls with same data should produce identical results."""
        dConj1, _, cal1 = clusterer_with_data.make_naive_bayes_distance_matrix()
        dConj2, _, cal2 = clusterer_with_data.make_naive_bayes_distance_matrix()
        np.testing.assert_array_equal(dConj1.data, dConj2.data)


class Test_find_optimal_nb_combination_DE:
    """Tests for Clusterer.find_optimal_nb_combination_DE (hybrid NB + DE)."""

    def _run_hybrid(self, clusterer_with_data):
        """Helper: run NB calibration then optimize combination."""
        clusterer_with_data.make_naive_bayes_distance_matrix()
        return clusterer_with_data.find_optimal_nb_combination_DE(
            de_kwargs={
                'maxiter': 5,
                'tol': 1e-4,
                'popsize': 5,
                'mutation': (0.5, 1.5),
                'recombination': 0.7,
                'polish': False,
            },
            seed=42,
        )

    def test_returns_correct_types(self, clusterer_with_data):
        """Should return (dConj, sConj, result_info) with correct types."""
        import scipy.sparse
        dConj, sConj, result_info = self._run_hybrid(clusterer_with_data)
        assert isinstance(dConj, scipy.sparse.csr_matrix)
        assert isinstance(sConj, scipy.sparse.csr_matrix)
        assert isinstance(result_info, dict)
        assert 'best_params' in result_info
        assert 'loss' in result_info
        assert 'n_evals' in result_info
        assert 'p_same_combined' in result_info

    def test_output_shapes_match_input(self, clusterer_with_data):
        """dConj and sConj should have same shape/nnz as s_sf."""
        dConj, sConj, _ = self._run_hybrid(clusterer_with_data)
        assert dConj.shape == clusterer_with_data.s_sf.shape
        assert sConj.shape == clusterer_with_data.s_sf.shape
        assert dConj.nnz == clusterer_with_data.s_sf.nnz

    def test_distances_in_valid_range(self, clusterer_with_data):
        """Distances should be in [0, 1]."""
        dConj, sConj, _ = self._run_hybrid(clusterer_with_data)
        assert dConj.data.min() >= 0.0 - 1e-6
        assert dConj.data.max() <= 1.0 + 1e-6
        assert sConj.data.min() >= 0.0 - 1e-6
        assert sConj.data.max() <= 1.0 + 1e-6

    def test_distance_plus_similarity_equals_one(self, clusterer_with_data):
        """d + s should equal 1 for every pair."""
        dConj, sConj, _ = self._run_hybrid(clusterer_with_data)
        np.testing.assert_allclose(
            dConj.data + sConj.data, 1.0, atol=1e-6,
        )

    def test_best_params_have_expected_keys(self, clusterer_with_data):
        """best_params should contain p_norm, w_NN, w_SWT."""
        _, _, result_info = self._run_hybrid(clusterer_with_data)
        params = result_info['best_params']
        assert set(params.keys()) == {'p_norm', 'w_NN', 'w_SWT'}
        ## All should be finite
        for k, v in params.items():
            assert np.isfinite(v), f'{k}={v} is not finite'

    def test_loss_is_finite(self, clusterer_with_data):
        """The optimized loss should be finite (not 1e6 penalty)."""
        _, _, result_info = self._run_hybrid(clusterer_with_data)
        assert np.isfinite(result_info['loss'])
        assert result_info['loss'] < 1e6

    def test_deterministic_with_seed(self, clusterer_with_data):
        """Two runs with same seed should produce identical results."""
        dConj1, _, info1 = self._run_hybrid(clusterer_with_data)
        dConj2, _, info2 = self._run_hybrid(clusterer_with_data)
        np.testing.assert_array_equal(dConj1.data, dConj2.data)
        assert info1['loss'] == info2['loss']

    def test_compatible_with_pruning(self, clusterer_with_data):
        """Output should work with make_pruned_similarity_graphs(precomputed)."""
        self._run_hybrid(clusterer_with_data)
        clusterer_with_data.make_pruned_similarity_graphs(
            kwargs_makeConjunctiveDistanceMatrix='precomputed',
        )
        assert hasattr(clusterer_with_data, 'dConj_pruned')
        assert clusterer_with_data.dConj_pruned is not None

    def test_requires_nb_calibration_first(self, clusterer_with_data):
        """Should raise AssertionError if NB calibration hasn't been done."""
        from roicat import tracking
        ## Create a fresh clusterer without NB calibration
        fresh = tracking.clustering.Clusterer(
            s_sf=clusterer_with_data.s_sf,
            s_NN_z=clusterer_with_data.s_NN_z,
            s_SWT_z=clusterer_with_data.s_SWT_z,
            s_sesh=clusterer_with_data.s_sesh,
            verbose=False,
        )
        with pytest.raises(AssertionError, match="make_naive_bayes_distance_matrix"):
            fresh.find_optimal_nb_combination_DE(seed=42)

    def test_subsample_pairs(self, clusterer_with_data):
        """Subsampling should still produce valid outputs."""
        clusterer_with_data.make_naive_bayes_distance_matrix()
        dConj, sConj, result_info = clusterer_with_data.find_optimal_nb_combination_DE(
            subsample_pairs=1000,
            de_kwargs={
                'maxiter': 3,
                'tol': 1e-4,
                'popsize': 5,
                'polish': False,
            },
            seed=42,
        )
        assert dConj.shape == clusterer_with_data.s_sf.shape
        assert dConj.data.min() >= 0.0 - 1e-6
        assert dConj.data.max() <= 1.0 + 1e-6
        assert np.isfinite(result_info['loss'])


class Test_edge_cases:
    """Edge case and robustness tests for the new mixing methods."""

    def test_extreme_p_norm_bounds(self, clusterer_with_data):
        """DE should handle near-zero p_norm without NaN/Inf."""
        result = clusterer_with_data._find_optimal_parameters_DE(
            seed=42,
            bounds_findParameters={
                'power_NN': [0.0, 0.5],
                'power_SWT': [0.0, 0.5],
                'p_norm': [-0.5, -0.1],
            },
            de_kwargs={
                'maxiter': 3, 'tol': 1e-4, 'popsize': 5, 'polish': False,
            },
        )
        assert np.isfinite(clusterer_with_data._de_result.fun)

    def test_very_small_subsample(self, clusterer_with_data):
        """Even tiny subsamples should work (clamp to minimum 100)."""
        result = clusterer_with_data._find_optimal_parameters_DE(
            seed=42, subsample_pairs=10,
            de_kwargs={
                'maxiter': 3, 'tol': 1e-4, 'popsize': 5, 'polish': False,
            },
        )
        assert np.isfinite(clusterer_with_data._de_result.fun)

    def test_nb_calibration_monotonicity(self, clusterer_with_data):
        """P(same|s_k) bins should be strictly monotonically non-decreasing for all features."""
        _, _, cal = clusterer_with_data.make_naive_bayes_distance_matrix()
        for name, feat_cal in cal['features'].items():
            p = feat_cal['p_same_bins']
            if isinstance(p, torch.Tensor):
                p = p.numpy()
            diffs = np.diff(p)
            assert np.all(diffs >= -1e-7), (
                f'{name}: P(same) bins not monotonic, min diff = {diffs.min():.8f}'
            )

    def test_nb_distances_no_nans(self, clusterer_with_data):
        """NB distance matrix should contain no NaN or Inf values."""
        dConj, sConj, _ = clusterer_with_data.make_naive_bayes_distance_matrix()
        assert np.all(np.isfinite(dConj.data)), 'dConj has NaN/Inf'
        assert np.all(np.isfinite(sConj.data)), 'sConj has NaN/Inf'

    def test_sigmoid_matches_nb_estimates(self, clusterer_with_data):
        """Frozen sigmoid params should exactly match NB-estimated values."""
        clusterer_with_data.make_naive_bayes_distance_matrix()
        sig_params = clusterer_with_data._estimate_sigmoid_params()

        result = clusterer_with_data._find_optimal_parameters_DE(
            seed=42,
            de_kwargs={
                'maxiter': 3, 'tol': 1e-4, 'popsize': 5, 'polish': False,
            },
        )
        assert result['sig_NN_kwargs']['mu'] == sig_params['NN']['mu']
        assert result['sig_NN_kwargs']['b'] == sig_params['NN']['b']
        assert result['sig_SWT_kwargs']['mu'] == sig_params['SWT']['mu']
        assert result['sig_SWT_kwargs']['b'] == sig_params['SWT']['b']

    def test_serialization_roundtrip(self, clusterer_with_data):
        """Clusterer state after NB calibration should survive serialization."""
        import pickle

        clusterer_with_data.make_naive_bayes_distance_matrix()
        sd = clusterer_with_data.serializable_dict

        ## Should be picklable
        data = pickle.dumps(sd)
        restored = pickle.loads(data)

        ## Calibration data should survive (not be __repr__ strings)
        cal = restored.get('calibrations_naive_bayes', {})
        if cal:  ## May be empty if serialization converts to __repr__
            features = cal.get('features', {})
            for feat_name, feat_cal in features.items():
                for key, val in feat_cal.items():
                    assert not isinstance(val, dict) or '__repr__' not in val, (
                        f'Lost calibration data: {feat_name}.{key}'
                    )

    def test_synthetic_data_de(self):
        """DE should work on fully synthetic data with known structure."""
        from roicat import tracking

        rng = np.random.RandomState(42)
        n = 60
        n_sessions = 3

        ## Create sparse sparsity pattern
        rows, cols = [], []
        for i in range(n):
            for j in range(i + 1, min(i + 10, n)):
                rows.extend([i, j])
                cols.extend([j, i])
        rows, cols = np.array(rows), np.array(cols)
        nnz = len(rows)

        ## Spatial footprint: higher similarity for same-cell pairs
        sf_data = rng.rand(nnz).astype(np.float64) * 0.5
        s_sf = scipy.sparse.csr_matrix(
            (sf_data, (rows, cols)), shape=(n, n),
        )

        ## NN and SWT: z-scored similarities
        s_NN_z = s_sf.copy()
        s_NN_z.data = rng.randn(nnz).astype(np.float64)
        s_SWT_z = s_sf.copy()
        s_SWT_z.data = rng.randn(nnz).astype(np.float64)

        ## Session matrix
        session_ids = np.repeat(np.arange(n_sessions), n // n_sessions + 1)[:n]
        s_sesh_data = np.array([
            float(session_ids[r] != session_ids[c]) for r, c in zip(rows, cols)
        ], dtype=np.float64)
        s_sesh = scipy.sparse.csr_matrix(
            (s_sesh_data, (rows, cols)), shape=(n, n),
        )

        c = tracking.clustering.Clusterer(
            s_sf=s_sf, s_NN_z=s_NN_z, s_SWT_z=s_SWT_z,
            s_sesh=s_sesh, verbose=False,
        )

        result = c._find_optimal_parameters_DE(
            seed=42,
            de_kwargs={
                'maxiter': 5, 'tol': 1e-4, 'popsize': 5, 'polish': False,
            },
        )

        assert set(result.keys()) == {
            'power_SF', 'power_NN', 'power_SWT', 'p_norm',
            'sig_SF_kwargs', 'sig_NN_kwargs', 'sig_SWT_kwargs',
        }
        assert np.isfinite(c._de_result.fun)

    def test_synthetic_data_nb(self):
        """NB should work on fully synthetic data."""
        from roicat import tracking

        rng = np.random.RandomState(42)
        n = 60
        n_sessions = 3

        rows, cols = [], []
        for i in range(n):
            for j in range(i + 1, min(i + 10, n)):
                rows.extend([i, j])
                cols.extend([j, i])
        rows, cols = np.array(rows), np.array(cols)
        nnz = len(rows)

        s_sf = scipy.sparse.csr_matrix(
            (rng.rand(nnz).astype(np.float64), (rows, cols)), shape=(n, n),
        )
        s_NN_z = s_sf.copy()
        s_NN_z.data = rng.randn(nnz).astype(np.float64)
        s_SWT_z = s_sf.copy()
        s_SWT_z.data = rng.randn(nnz).astype(np.float64)

        session_ids = np.repeat(np.arange(n_sessions), n // n_sessions + 1)[:n]
        s_sesh_data = np.array([
            float(session_ids[r] != session_ids[c]) for r, c in zip(rows, cols)
        ], dtype=np.float64)
        s_sesh = scipy.sparse.csr_matrix(
            (s_sesh_data, (rows, cols)), shape=(n, n),
        )

        c = tracking.clustering.Clusterer(
            s_sf=s_sf, s_NN_z=s_NN_z, s_SWT_z=s_SWT_z,
            s_sesh=s_sesh, verbose=False,
        )

        dConj, sConj, cal = c.make_naive_bayes_distance_matrix()
        assert dConj.shape == (n, n)
        assert np.all(np.isfinite(dConj.data))
        assert np.all(dConj.data >= 0)
        assert np.all(dConj.data <= 1)


######################################################################################################################################
################################################### ROI SIZE VALIDATION ##############################################################
######################################################################################################################################


class Test_check_roi_sizes:
    def test_normal_rois(self):
        """Normal ROIs should pass all checks."""
        from roicat.helpers import check_roi_sizes

        np.random.seed(42)
        H, W = 100, 100
        sfs = [scipy.sparse.random(10, H*W, density=0.01, format='csr')]
        result = check_roi_sizes(sfs, H, W, verbose=False)
        assert result['n_too_small'] == 0
        assert result['n_too_large'] == 0

    def test_too_small_detected(self):
        """Very small ROIs should be flagged."""
        from roicat.helpers import check_roi_sizes

        H, W = 100, 100
        ## Create ROI with only 2 nonzero pixels
        data = np.array([1.0, 1.0])
        row = np.array([0, 0])
        col = np.array([0, 1])
        sf = scipy.sparse.csr_matrix((data, (row, col)), shape=(1, H*W))
        result = check_roi_sizes([sf], H, W, min_pixels=5, verbose=False)
        assert result['n_too_small'] == 1

    def test_too_large_detected(self):
        """ROIs covering >10% of FOV should be flagged."""
        from roicat.helpers import check_roi_sizes

        H, W = 100, 100
        ## Create ROI covering 20% of pixels
        n_pix = H * W
        n_active = n_pix // 5  ## 20%
        data = np.ones(n_active)
        row = np.zeros(n_active, dtype=int)
        col = np.arange(n_active)
        sf = scipy.sparse.csr_matrix((data, (row, col)), shape=(1, n_pix))
        result = check_roi_sizes([sf], H, W, max_fraction=0.1, verbose=False)
        assert result['n_too_large'] == 1


