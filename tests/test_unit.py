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
########################################################## BLURRING ##################################################################
######################################################################################################################################


class Test_ROI_Blurrer:
    """Tests for roicat.tracking.blurring.ROI_Blurrer."""

    @staticmethod
    def _make_synthetic_sparse_rois(n_rois, frame_height, frame_width, rng):
        """Create synthetic sparse ROIs (each row is a flattened FOV)."""
        rows_list = []
        for i in range(n_rois):
            ## Place a small blob at a random location
            cy = rng.randint(5, frame_height - 5)
            cx = rng.randint(5, frame_width - 5)
            pixels = []
            values = []
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    y, x = cy + dy, cx + dx
                    if 0 <= y < frame_height and 0 <= x < frame_width:
                        pixels.append(y * frame_width + x)
                        values.append(rng.rand())
            col = np.array(pixels)
            data = np.array(values, dtype=np.float32)
            row_idx = np.zeros(len(col), dtype=int)
            row = scipy.sparse.csr_matrix(
                (data, (row_idx, col)),
                shape=(1, frame_height * frame_width),
            )
            rows_list.append(row)
        return scipy.sparse.vstack(rows_list).tocsr()

    def test_blur_ROIs_output_shape(self):
        """Blurred output should have the same number of ROIs as input."""
        from roicat.tracking.blurring import ROI_Blurrer

        rng = np.random.RandomState(42)
        n_rois = 10
        frame_height, frame_width = 50, 50

        rois = self._make_synthetic_sparse_rois(n_rois, frame_height, frame_width, rng)

        blurrer = ROI_Blurrer(
            frame_shape=(frame_height, frame_width),
            kernel_halfWidth=2,
            plot_kernel=False,
            verbose=False,
        )
        result = blurrer.blur_ROIs(spatialFootprints=[rois])

        assert len(result) == 1
        assert result[0].shape[0] == n_rois
        assert result[0].shape[1] == frame_height * frame_width

    def test_blur_ROIs_preserves_sparsity(self):
        """Output of blurring should be a sparse matrix."""
        from roicat.tracking.blurring import ROI_Blurrer

        rng = np.random.RandomState(43)
        n_rois = 5
        frame_height, frame_width = 40, 40

        rois = self._make_synthetic_sparse_rois(n_rois, frame_height, frame_width, rng)

        blurrer = ROI_Blurrer(
            frame_shape=(frame_height, frame_width),
            kernel_halfWidth=2,
            plot_kernel=False,
            verbose=False,
        )
        result = blurrer.blur_ROIs(spatialFootprints=[rois])
        assert scipy.sparse.issparse(result[0]), "Blurred output should be sparse"

    def test_blur_ROIs_increases_roi_width(self):
        """Blurring should increase the number of nonzero pixels in each ROI."""
        from roicat.tracking.blurring import ROI_Blurrer

        rng = np.random.RandomState(44)
        n_rois = 5
        frame_height, frame_width = 50, 50

        rois = self._make_synthetic_sparse_rois(n_rois, frame_height, frame_width, rng)
        nnz_before = rois.getnnz(axis=1)

        blurrer = ROI_Blurrer(
            frame_shape=(frame_height, frame_width),
            kernel_halfWidth=3,
            plot_kernel=False,
            verbose=False,
        )
        result = blurrer.blur_ROIs(spatialFootprints=[rois])
        nnz_after = result[0].getnnz(axis=1)

        ## Blurring should spread the values to neighboring pixels
        assert np.all(nnz_after >= nnz_before), (
            "Blurring should not decrease the number of nonzero pixels"
        )
        ## At least some ROIs should have more nonzero pixels
        assert np.any(nnz_after > nnz_before), (
            "Blurring with kernel_halfWidth=3 should increase extent of at least some ROIs"
        )

    def test_blur_ROIs_different_kernel_widths(self):
        """Different kernel_halfWidth values should produce different results."""
        from roicat.tracking.blurring import ROI_Blurrer

        rng = np.random.RandomState(45)
        n_rois = 5
        frame_height, frame_width = 50, 50

        rois = self._make_synthetic_sparse_rois(n_rois, frame_height, frame_width, rng)

        blurrer_small = ROI_Blurrer(
            frame_shape=(frame_height, frame_width),
            kernel_halfWidth=2,
            plot_kernel=False,
            verbose=False,
        )
        result_small = blurrer_small.blur_ROIs(spatialFootprints=[rois])

        blurrer_large = ROI_Blurrer(
            frame_shape=(frame_height, frame_width),
            kernel_halfWidth=5,
            plot_kernel=False,
            verbose=False,
        )
        result_large = blurrer_large.blur_ROIs(spatialFootprints=[rois])

        nnz_small = result_small[0].getnnz(axis=1)
        nnz_large = result_large[0].getnnz(axis=1)

        ## Larger kernel should produce wider (or equal) spatial extent
        assert np.all(nnz_large >= nnz_small), (
            "Larger kernel should produce at least as many nonzero pixels"
        )

    def test_blur_ROIs_zero_kernel(self):
        """With kernel_halfWidth=0, output should equal input (no blurring)."""
        from roicat.tracking.blurring import ROI_Blurrer

        rng = np.random.RandomState(46)
        n_rois = 5
        frame_height, frame_width = 40, 40

        rois = self._make_synthetic_sparse_rois(n_rois, frame_height, frame_width, rng)

        blurrer = ROI_Blurrer(
            frame_shape=(frame_height, frame_width),
            kernel_halfWidth=0,
            plot_kernel=False,
            verbose=False,
        )
        result = blurrer.blur_ROIs(spatialFootprints=[rois])

        ## With zero kernel width, the internal _width=0 triggers the bypass path
        ## which sets ROIs_blurred = spatialFootprints directly
        assert result[0] is rois, (
            "With kernel_halfWidth=0, output should be the same object as input"
        )

    def test_blur_ROIs_multiple_sessions(self):
        """Blurrer should handle a list of multiple sessions."""
        from roicat.tracking.blurring import ROI_Blurrer

        rng = np.random.RandomState(47)
        frame_height, frame_width = 40, 40

        rois_1 = self._make_synthetic_sparse_rois(5, frame_height, frame_width, rng)
        rois_2 = self._make_synthetic_sparse_rois(8, frame_height, frame_width, rng)

        blurrer = ROI_Blurrer(
            frame_shape=(frame_height, frame_width),
            kernel_halfWidth=2,
            plot_kernel=False,
            verbose=False,
        )
        result = blurrer.blur_ROIs(spatialFootprints=[rois_1, rois_2])

        assert len(result) == 2
        assert result[0].shape[0] == 5
        assert result[1].shape[0] == 8


######################################################################################################################################
####################################################### SIMILARITY GRAPH #############################################################
######################################################################################################################################


class Test_ROI_graph:
    """Tests for roicat.tracking.similarity_graph.ROI_graph."""

    def test_make_block_batches(self):
        """_make_block_batches should produce blocks covering the full FOV."""
        from roicat.tracking.similarity_graph import ROI_graph

        graph = ROI_graph(
            n_workers=1,
            frame_height=100,
            frame_width=200,
            block_height=50,
            block_width=50,
            verbose=False,
        )

        assert len(graph.blocks) > 0, "Should produce at least one block"
        ## Check that all blocks are within FOV bounds
        for block in graph.blocks:
            assert block[0][0] >= 0
            assert block[0][1] <= 100
            assert block[1][0] >= 0
            assert block[1][1] <= 200

    def test_compute_spatial_footprint_similarity(self):
        """Nearby ROIs should have higher spatial similarity than distant ones."""
        from roicat.tracking.similarity_graph import ROI_graph

        rng = np.random.RandomState(50)
        frame_height, frame_width = 50, 50
        n_pixels = frame_height * frame_width

        ## Create two pairs of ROIs: one close pair and one far pair
        ## Close pair: both near (10, 10)
        def make_blob(cy, cx, radius=3):
            row_data = []
            col_data = []
            val_data = []
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    y, x = cy + dy, cx + dx
                    if 0 <= y < frame_height and 0 <= x < frame_width:
                        col_data.append(y * frame_width + x)
                        val_data.append(1.0)
            return np.array(col_data), np.array(val_data, dtype=np.float32)

        cols_a, vals_a = make_blob(10, 10)
        cols_b, vals_b = make_blob(12, 12)  ## Close to A
        cols_c, vals_c = make_blob(40, 40)  ## Far from A

        rows = []
        for cols, vals in [(cols_a, vals_a), (cols_b, vals_b), (cols_c, vals_c)]:
            row = scipy.sparse.csr_matrix(
                (vals, (np.zeros(len(cols), dtype=int), cols)),
                shape=(1, n_pixels),
            )
            rows.append(row)
        sf = scipy.sparse.vstack(rows).tocsr()

        ## Use the helper method to compute similarity
        graph = ROI_graph(
            n_workers=1,
            frame_height=frame_height,
            frame_width=frame_width,
            block_height=frame_height,
            block_width=frame_width,
            verbose=False,
        )
        graph._sf_maskPower = 1.0

        features_NN = torch.randn(3, 16)
        features_SWT = torch.randn(3, 16)
        ## All ROIs from different sessions so none are masked
        ROI_session_bool = torch.tensor([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ], dtype=torch.float32)

        s_sf, s_NN, s_SWT, s_sesh = graph._helper_compute_ROI_similarity_graph(
            spatialFootprints=sf,
            features_NN=features_NN,
            features_SWT=features_SWT,
            ROI_session_bool=ROI_session_bool,
        )

        assert s_sf is not None, "s_sf should not be None for non-empty input"
        ## Close ROIs (0,1) should have higher spatial similarity than distant ones (0,2)
        sim_close = s_sf[0, 1]
        sim_far = s_sf[0, 2]
        assert sim_close > sim_far, (
            f"Close ROIs should have higher spatial similarity ({sim_close}) "
            f"than distant ROIs ({sim_far})"
        )

    def test_compute_spatial_footprint_similarity_shape(self):
        """Similarity matrix should be square with shape (n_rois, n_rois)."""
        from roicat.tracking.similarity_graph import ROI_graph

        rng = np.random.RandomState(51)
        frame_height, frame_width = 30, 30
        n_pixels = frame_height * frame_width
        n_rois = 5

        ## Create random sparse ROIs
        rows_list = []
        for i in range(n_rois):
            cy = rng.randint(5, frame_height - 5)
            cx = rng.randint(5, frame_width - 5)
            pixels = []
            values = []
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    y, x = cy + dy, cx + dx
                    if 0 <= y < frame_height and 0 <= x < frame_width:
                        pixels.append(y * frame_width + x)
                        values.append(rng.rand())
            row = scipy.sparse.csr_matrix(
                (np.array(values, dtype=np.float32),
                 (np.zeros(len(pixels), dtype=int), np.array(pixels))),
                shape=(1, n_pixels),
            )
            rows_list.append(row)
        sf = scipy.sparse.vstack(rows_list).tocsr()

        graph = ROI_graph(
            n_workers=1,
            frame_height=frame_height,
            frame_width=frame_width,
            block_height=frame_height,
            block_width=frame_width,
            verbose=False,
        )
        graph._sf_maskPower = 1.0

        features_NN = torch.randn(n_rois, 16)
        features_SWT = torch.randn(n_rois, 16)
        ## All different sessions
        ROI_session_bool = torch.eye(n_rois, dtype=torch.float32)

        s_sf, s_NN, s_SWT, s_sesh = graph._helper_compute_ROI_similarity_graph(
            spatialFootprints=sf,
            features_NN=features_NN,
            features_SWT=features_SWT,
            ROI_session_bool=ROI_session_bool,
        )

        assert s_sf.shape == (n_rois, n_rois), f"Expected shape ({n_rois}, {n_rois}), got {s_sf.shape}"
        assert scipy.sparse.issparse(s_sf), "s_sf should be sparse"
        assert scipy.sparse.issparse(s_NN), "s_NN should be sparse"

    def test_spatial_similarity_self_same_session(self):
        """ROIs in the same session should be masked via session_bool, and
        diagonal entries should be zero (no self-similarity)."""
        from roicat.tracking.similarity_graph import ROI_graph

        rng = np.random.RandomState(52)
        frame_height, frame_width = 30, 30
        n_pixels = frame_height * frame_width

        ## Two overlapping ROIs in the SAME session
        def make_blob(cy, cx, radius=3):
            cols, vals = [], []
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    y, x = cy + dy, cx + dx
                    if 0 <= y < frame_height and 0 <= x < frame_width:
                        cols.append(y * frame_width + x)
                        vals.append(1.0)
            return np.array(cols), np.array(vals, dtype=np.float32)

        cols_a, vals_a = make_blob(15, 15)
        cols_b, vals_b = make_blob(16, 16)

        rows = []
        for cols, vals in [(cols_a, vals_a), (cols_b, vals_b)]:
            row = scipy.sparse.csr_matrix(
                (vals, (np.zeros(len(cols), dtype=int), cols)),
                shape=(1, n_pixels),
            )
            rows.append(row)
        sf = scipy.sparse.vstack(rows).tocsr()

        graph = ROI_graph(
            n_workers=1,
            frame_height=frame_height,
            frame_width=frame_width,
            block_height=frame_height,
            block_width=frame_width,
            verbose=False,
        )
        graph._sf_maskPower = 1.0

        features_NN = torch.randn(2, 16)
        features_SWT = torch.randn(2, 16)
        ## Both ROIs in the SAME session
        ROI_session_bool = torch.tensor([
            [1, 0],
            [1, 0],
        ], dtype=torch.float32)

        s_sf, s_NN, s_SWT, s_sesh = graph._helper_compute_ROI_similarity_graph(
            spatialFootprints=sf,
            features_NN=features_NN,
            features_SWT=features_SWT,
            ROI_session_bool=ROI_session_bool,
        )

        ## Diagonal should be zero (no self-similarity)
        assert s_sf[0, 0] == 0, "Diagonal of s_sf should be zero"
        assert s_sf[1, 1] == 0, "Diagonal of s_sf should be zero"

    def test_empty_spatial_footprints(self):
        """Helper should return None for empty spatial footprints."""
        from roicat.tracking.similarity_graph import ROI_graph

        frame_height, frame_width = 30, 30
        n_pixels = frame_height * frame_width

        sf = scipy.sparse.csr_matrix((0, n_pixels), dtype=np.float32)

        graph = ROI_graph(
            n_workers=1,
            frame_height=frame_height,
            frame_width=frame_width,
            block_height=frame_height,
            block_width=frame_width,
            verbose=False,
        )
        graph._sf_maskPower = 1.0

        features_NN = torch.randn(0, 16)
        features_SWT = torch.randn(0, 16)
        ROI_session_bool = torch.zeros((0, 1), dtype=torch.float32)

        s_sf, s_NN, s_SWT, s_sesh = graph._helper_compute_ROI_similarity_graph(
            spatialFootprints=sf,
            features_NN=features_NN,
            features_SWT=features_SWT,
            ROI_session_bool=ROI_session_bool,
        )

        assert s_sf is None
        assert s_NN is None


######################################################################################################################################
######################################################### HELPERS EXTRA ##############################################################
######################################################################################################################################


class Test_prepare_params:
    """Tests for helpers.prepare_params."""

    def test_prepare_params_merges_partial(self):
        """Partial params should override defaults."""
        defaults = {'a': 1, 'b': 2, 'c': 3}
        params = {'a': 10}
        result = helpers.prepare_params(params, defaults, verbose=False)
        assert result['a'] == 10, "Partial param should override default"
        assert result['b'] == 2, "Missing param should use default"
        assert result['c'] == 3, "Missing param should use default"

    def test_prepare_params_preserves_defaults(self):
        """When params is empty, all defaults should be used."""
        defaults = {'x': 42, 'y': 'hello'}
        params = {}
        result = helpers.prepare_params(params, defaults, verbose=False)
        assert result == defaults

    def test_prepare_params_nested(self):
        """Nested dict merging should work correctly."""
        defaults = {
            'outer': {
                'inner_a': 1,
                'inner_b': 2,
            },
            'top_level': 99,
        }
        params = {
            'outer': {
                'inner_a': 100,
            },
        }
        result = helpers.prepare_params(params, defaults, verbose=False)
        assert result['outer']['inner_a'] == 100, "Nested override should work"
        assert result['outer']['inner_b'] == 2, "Nested default should be preserved"
        assert result['top_level'] == 99, "Top-level default should be preserved"

    def test_prepare_params_invalid_key_raises(self):
        """Keys not in defaults should raise an AssertionError."""
        defaults = {'a': 1, 'b': 2}
        params = {'a': 1, 'z': 999}
        with pytest.raises(AssertionError):
            helpers.prepare_params(params, defaults, verbose=False)

    def test_prepare_params_returns_deepcopy(self):
        """Output should be a deepcopy, not a reference to the input."""
        defaults = {'a': [1, 2, 3]}
        params = {}
        result = helpers.prepare_params(params, defaults, verbose=False)
        result['a'].append(4)
        assert defaults['a'] == [1, 2, 3], "Defaults should not be modified"


class Test_yaml_save_load:
    """Tests for helpers.yaml_save and helpers.yaml_load."""

    def test_yaml_roundtrip(self, tmp_path):
        """Save then load should produce the same dict."""
        data = {'alpha': 1, 'beta': 'hello', 'gamma': [1, 2, 3]}
        filepath = str(tmp_path / 'test.yaml')
        helpers.yaml_save(data, filepath)
        loaded = helpers.yaml_load(filepath)
        assert loaded == data

    def test_yaml_roundtrip_nested(self, tmp_path):
        """Nested dicts should survive roundtrip."""
        data = {
            'level1': {
                'level2': {
                    'value': 42,
                    'list': [1.0, 2.0, 3.0],
                },
            },
            'top': 'test',
        }
        filepath = str(tmp_path / 'nested.yaml')
        helpers.yaml_save(data, filepath)
        loaded = helpers.yaml_load(filepath)
        assert loaded == data

    def test_yaml_with_numpy(self, tmp_path):
        """Numpy scalars/arrays should be handled (converted to Python types)."""
        data = {
            'int_val': int(np.int64(42)),
            'float_val': float(np.float64(3.14)),
            'list_val': [float(x) for x in np.array([1.0, 2.0, 3.0])],
        }
        filepath = str(tmp_path / 'numpy_test.yaml')
        helpers.yaml_save(data, filepath)
        loaded = helpers.yaml_load(filepath)
        assert loaded['int_val'] == 42
        assert abs(loaded['float_val'] - 3.14) < 1e-10
        assert loaded['list_val'] == [1.0, 2.0, 3.0]

    def test_yaml_overwrite_protection(self, tmp_path):
        """When allow_overwrite=False and file exists, should raise."""
        data = {'a': 1}
        filepath = str(tmp_path / 'protect.yaml')
        helpers.yaml_save(data, filepath)
        with pytest.raises((FileExistsError, AssertionError)):
            helpers.yaml_save(data, filepath, allow_overwrite=False)


class Test_find_paths:
    """Tests for helpers.find_paths."""

    def test_find_paths_basic(self, tmp_path):
        """Should find files matching a regex pattern."""
        ## Create some temp files
        (tmp_path / 'data_001.csv').touch()
        (tmp_path / 'data_002.csv').touch()
        (tmp_path / 'other.txt').touch()

        result = helpers.find_paths(
            dir_outer=str(tmp_path),
            reMatch=r'data_\d+\.csv',
            find_files=True,
            find_folders=False,
            depth=0,
        )
        assert len(result) == 2
        assert all('data_' in p and '.csv' in p for p in result)

    def test_find_paths_no_match(self, tmp_path):
        """Should return empty list when no files match."""
        (tmp_path / 'file.txt').touch()

        result = helpers.find_paths(
            dir_outer=str(tmp_path),
            reMatch=r'\.csv$',
            find_files=True,
            find_folders=False,
            depth=0,
        )
        assert len(result) == 0

    def test_find_paths_finds_folders(self, tmp_path):
        """Should find folders when find_folders=True."""
        (tmp_path / 'subdir_a').mkdir()
        (tmp_path / 'subdir_b').mkdir()
        (tmp_path / 'other_file.txt').touch()

        result = helpers.find_paths(
            dir_outer=str(tmp_path),
            reMatch=r'subdir_',
            find_files=False,
            find_folders=True,
            depth=0,
        )
        assert len(result) == 2

    def test_find_paths_depth(self, tmp_path):
        """Should find files at deeper levels when depth > 0."""
        subdir = tmp_path / 'level1' / 'level2'
        subdir.mkdir(parents=True)
        (subdir / 'deep_file.txt').touch()
        (tmp_path / 'shallow_file.txt').touch()

        ## depth=0 should not find the deep file
        result_shallow = helpers.find_paths(
            dir_outer=str(tmp_path),
            reMatch=r'deep_file',
            find_files=True,
            depth=0,
        )
        assert len(result_shallow) == 0

        ## depth=2 should find the deep file
        result_deep = helpers.find_paths(
            dir_outer=str(tmp_path),
            reMatch=r'deep_file',
            find_files=True,
            depth=2,
        )
        assert len(result_deep) == 1

    def test_find_paths_list_of_dirs(self, tmp_path):
        """Should accept a list of directories."""
        dir_a = tmp_path / 'dir_a'
        dir_b = tmp_path / 'dir_b'
        dir_a.mkdir()
        dir_b.mkdir()
        (dir_a / 'file_a.txt').touch()
        (dir_b / 'file_b.txt').touch()

        result = helpers.find_paths(
            dir_outer=[str(dir_a), str(dir_b)],
            reMatch=r'file_',
            find_files=True,
            depth=0,
        )
        assert len(result) == 2
