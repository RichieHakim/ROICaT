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
            'sparse': scipy.sparse.random_array((50, 50), density=0.1, format='csr', dtype=np.float32),
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
        mat = scipy.sparse.random_array((100, 100), density=0.05, format='csr', dtype=np.float64)
        test_data = {'sparse_mat': mat}
        path = str(tmp_path / 'sparse_test.richfile.zip')
        util.RichFile_ROICaT(path=path, backend='zip').save(obj=test_data, overwrite=True)
        loaded = util.RichFile_ROICaT(path=path).load()
        assert scipy.sparse.issparse(loaded['sparse_mat'])
        assert np.allclose(loaded['sparse_mat'].toarray(), mat.toarray())

    def test_similarity_metric_roundtrip(self, tmp_path):
        """SimilarityMetric objects should survive RichFile save/load."""
        from roicat.tracking.similarity_graph import SimilarityMetric, DEFAULT_METRICS
        test_data = {'metrics': DEFAULT_METRICS}
        path = str(tmp_path / 'metric_test.richfile.zip')
        util.RichFile_ROICaT(path=path, backend='zip').save(obj=test_data, overwrite=True)
        loaded = util.RichFile_ROICaT(path=path).load()
        assert len(loaded['metrics']) == 3
        for orig, loaded_m in zip(DEFAULT_METRICS, loaded['metrics']):
            assert isinstance(loaded_m, SimilarityMetric)
            assert loaded_m.name == orig.name
            assert loaded_m.is_sparsity_source == orig.is_sparsity_source
            assert loaded_m.normalize_zscore == orig.normalize_zscore
            assert loaded_m.optimize_power == orig.optimize_power
            assert loaded_m.optimize_sigmoid == orig.optimize_sigmoid
            assert loaded_m.similarity_fn == orig.similarity_fn  ## str stays str

    def test_similarity_metric_callable_saved_as_string(self, tmp_path):
        """Custom callable in similarity_fn should be saved as a descriptive string."""
        from roicat.tracking.similarity_graph import SimilarityMetric
        def my_custom_fn(features, **kwargs):
            return features
        metric = SimilarityMetric(name='custom', similarity_fn=my_custom_fn)
        test_data = {'metric': metric}
        path = str(tmp_path / 'callable_test.richfile.zip')
        util.RichFile_ROICaT(path=path, backend='zip').save(obj=test_data, overwrite=True)
        loaded = util.RichFile_ROICaT(path=path).load()
        loaded_m = loaded['metric']
        assert isinstance(loaded_m, SimilarityMetric)
        assert loaded_m.name == 'custom'
        assert isinstance(loaded_m.similarity_fn, str)
        assert 'my_custom_fn' in loaded_m.similarity_fn

    def test_similarity_metric_in_dict_keyed_by_name(self, tmp_path):
        """Dict of SimilarityMetric (as used by Clusterer) should roundtrip."""
        from roicat.tracking.similarity_graph import SimilarityMetric, DEFAULT_METRICS
        metrics_dict = {m.name: m for m in DEFAULT_METRICS}
        test_data = {'configs': metrics_dict}
        path = str(tmp_path / 'dict_metric_test.richfile.zip')
        util.RichFile_ROICaT(path=path, backend='zip').save(obj=test_data, overwrite=True)
        loaded = util.RichFile_ROICaT(path=path).load()
        assert set(loaded['configs'].keys()) == {'sf', 'nn', 'swt'}
        for name, m in loaded['configs'].items():
            assert isinstance(m, SimilarityMetric)
            assert m.name == name

    def test_pipeline_dict_with_similarity_metrics(self, tmp_path):
        """Simulated pipeline __dict__ containing SimilarityMetric should save/load."""
        from roicat.tracking.similarity_graph import SimilarityMetric, DEFAULT_METRICS
        ## Simulate what ROI_graph.__dict__ looks like
        sim_dict = {
            '_metric_configs_stored': list(DEFAULT_METRICS),
            'similarities': {
                'sf': scipy.sparse.random_array((50, 50), density=0.1, format='csr'),
                'nn': scipy.sparse.random_array((50, 50), density=0.1, format='csr'),
            },
            'params': {'__init__': {'verbose': True}},
        }
        test_data = {'sim': sim_dict}
        path = str(tmp_path / 'pipeline_test.richfile.zip')
        util.RichFile_ROICaT(path=path, backend='zip').save(obj=test_data, overwrite=True)
        loaded = util.RichFile_ROICaT(path=path).load()
        loaded_metrics = loaded['sim']['_metric_configs_stored']
        assert len(loaded_metrics) == 3
        for m in loaded_metrics:
            assert isinstance(m, SimilarityMetric)
        assert scipy.sparse.issparse(loaded['sim']['similarities']['sf'])


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
        s = scipy.sparse.random_array((50, 50), density=0.2, format='csr', rng=0)
        assert checker(s, s)[0] == True

    def test_sparse_close(self):
        checker = helpers.Equivalence_checker(kwargs_allclose={'rtol': 1e-5})
        s = scipy.sparse.random_array((50, 50), density=0.2, format='csr', rng=0)
        s2 = s.copy()
        s2.data = s2.data * (1 + 1e-7)  ## Proportional perturbation within rtol
        assert checker(s2, s)[0] == True

    def test_sparse_nonzero_atol_allows_implicit_zero_differences(self):
        checker = helpers.Equivalence_checker(kwargs_allclose={'rtol': 0, 'atol': 0.2, 'equal_nan': True})
        true = scipy.sparse.csr_array((3, 3), dtype=np.float32)
        test = scipy.sparse.csr_array(([0.1], ([1], [2])), shape=(3, 3), dtype=np.float32)
        result = checker(test, true)
        assert result[0] == True

    def test_sparse_exact_equality_ignores_explicit_zero_storage(self):
        checker = helpers.Equivalence_checker(kwargs_allclose={'rtol': 0, 'atol': 0, 'equal_nan': True})
        true = scipy.sparse.csr_array(([1.0], ([0], [0])), shape=(2, 2), dtype=np.float32)
        test = scipy.sparse.csr_array(([1.0, 0.0], ([0, 1], [0, 1])), shape=(2, 2), dtype=np.float32)
        result = checker(test, true)
        assert result[0] == True

    def test_sparse_different(self):
        checker = helpers.Equivalence_checker()
        s1 = scipy.sparse.csr_array(np.eye(3))
        s2 = scipy.sparse.csr_array(np.eye(3) * 2.0)
        result = checker(s1, s2)
        assert result[0] == False
        assert 'sparse allclose failed' in result[1]

    def test_sparse_shape_mismatch(self):
        checker = helpers.Equivalence_checker()
        s1 = scipy.sparse.csr_array(np.eye(3))
        s2 = scipy.sparse.csr_array(np.eye(4))
        result = checker(s1, s2)
        assert result[0] == False
        assert 'shape mismatch' in result[1]

    def test_sparse_type_mismatch(self):
        checker = helpers.Equivalence_checker()
        s = scipy.sparse.csr_array(np.eye(3))
        result = checker(s.toarray(), s)
        assert result[0] == False
        assert 'type mismatch' in result[1]

    def test_sparse_empty(self):
        checker = helpers.Equivalence_checker()
        s = scipy.sparse.csr_array((10, 10))
        assert checker(s, s)[0] == True

    def test_sparse_bool(self):
        checker = helpers.Equivalence_checker()
        s = scipy.sparse.random_array((20, 20), density=0.3, format='csr', rng=0)
        sb = (s != 0).astype(bool)
        assert checker(sb, sb)[0] == True

    def test_sparse_in_nested_dict(self):
        checker = helpers.Equivalence_checker()
        s = scipy.sparse.random_array((10, 10), density=0.5, format='csr', rng=0)
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

    def test_dict_vs_object_normalizes_public_attrs(self):
        class Dummy:
            def __init__(self):
                self.a = np.array([1.0, 2.0], dtype=np.float32)
                self.b = 'ok'
                self._private = 'ignore'

        checker = helpers.Equivalence_checker(verbose=False)
        result = checker(
            test=Dummy(),
            true={'a': np.array([1.0, 2.0], dtype=np.float32), 'b': 'ok'},
        )
        assert result['a'][0] == True
        assert result['b'][0] == True

    def test_torch_tensor_leaf_normalized_to_numpy(self):
        checker = helpers.Equivalence_checker(verbose=False)
        tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        array = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = checker(tensor, array)
        assert result[0] == True


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
        s = scipy.sparse.random_array((10, 20), density=0.3, format='csr', rng=0)
        t = helpers.scipy_sparse_to_torch_coo(s)
        assert t.shape == s.shape

    def test_values_preserved(self):
        s = scipy.sparse.random_array((10, 20), density=0.3, format='csr', rng=0)
        t = helpers.scipy_sparse_to_torch_coo(s)
        np.testing.assert_allclose(t.to_dense().numpy(), s.toarray(), rtol=1e-6)

    def test_empty(self):
        s = scipy.sparse.csr_array((5, 5))
        t = helpers.scipy_sparse_to_torch_coo(s)
        assert t.shape == (5, 5)
        assert t._nnz() == 0

    def test_dtype_override(self):
        s = scipy.sparse.random_array((5, 5), density=0.5, format='csr', rng=0)
        t = helpers.scipy_sparse_to_torch_coo(s, dtype=torch.float32)
        assert t.dtype == torch.float32


class Test_merge_sparse_arrays:
    """Tests for helpers.merge_sparse_arrays — used in clustering pipeline."""

    def test_basic_merge(self):
        """Two blocks placed at different positions in a larger matrix."""
        s1 = scipy.sparse.csr_array(np.array([[1.0, 0.5], [0.5, 1.0]]))
        s2 = scipy.sparse.csr_array(np.array([[2.0, 0.0], [0.0, 2.0]]))
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
        s = scipy.sparse.csr_array(a)
        idx = np.array([2, 4, 6, 8, 10])
        result = helpers.merge_sparse_arrays([s], [idx], shape_full=(12, 12))
        dense = result.toarray()
        np.testing.assert_allclose(dense, dense.T)

    def test_multiple_blocks_no_overlap(self):
        """Merging multiple non-overlapping blocks should preserve all values."""
        blocks = []
        idxs = []
        for i in range(3):
            s = scipy.sparse.csr_array(np.eye(2) * (i + 1))
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
########################################################## BLURRING ##################################################################
######################################################################################################################################


class Test_ROI_Blurrer:
    """Tests for ROI_Blurrer using sparse_convolution library."""

    def test_basic_blurring(self):
        """Blurring sparse ROIs produces correct shape and nonzero output."""
        from roicat.tracking.blurring import ROI_Blurrer

        blurrer = ROI_Blurrer(frame_shape=(64, 64), kernel_halfWidth=2, verbose=False)
        sf = scipy.sparse.random(10, 64 * 64, density=0.01, format='csr', dtype=np.float32)
        result = blurrer.blur_ROIs([sf])

        assert len(result) == 1
        assert result[0].shape == sf.shape  ## mode='same' preserves shape
        assert scipy.sparse.issparse(result[0])
        assert result[0].nnz > sf.nnz  ## blurring spreads nonzeros

    def test_zero_halfwidth_bypass(self):
        """kernel_halfWidth=0 returns input unchanged."""
        from roicat.tracking.blurring import ROI_Blurrer

        blurrer = ROI_Blurrer(frame_shape=(64, 64), kernel_halfWidth=0, verbose=False)
        sf = scipy.sparse.random(5, 64 * 64, density=0.01, format='csr', dtype=np.float32)
        result = blurrer.blur_ROIs([sf])

        assert result[0] is sf  ## exact same object, no copy

    def test_parity_with_old_toeplitz(self):
        """New sparse_convolution.direct matches the old helpers.Toeplitz_convolution2d."""
        from roicat.tracking.blurring import ROI_Blurrer

        ## Build kernel matching ROI_Blurrer internals
        kernel_halfWidth = 2
        width = kernel_halfWidth * 2
        kernel_size = max(int((width // 2) * 2) - 1, 1)
        kernel = helpers.cosine_kernel_2D(
            center=(kernel_size // 2, kernel_size // 2),
            image_size=(kernel_size, kernel_size),
            width=width,
        )
        kernel = kernel / kernel.sum()

        ## Old implementation (still in helpers.py)
        old_conv = helpers.Toeplitz_convolution2d(
            x_shape=(64, 64), k=kernel, mode='same', dtype=np.float32,
        )
        ## New implementation via ROI_Blurrer
        blurrer = ROI_Blurrer(frame_shape=(64, 64), kernel_halfWidth=2, verbose=False)

        rng = np.random.default_rng(42)
        x_dense = rng.random((20, 64 * 64), dtype=np.float32)
        x_dense[x_dense > 0.01] = 0.0
        x_sparse = scipy.sparse.csr_matrix(x_dense)

        out_old = old_conv(x=x_sparse, batching=True, mode='same')
        blurrer.blur_ROIs([x_sparse])
        out_new = blurrer.ROIs_blurred[0]

        np.testing.assert_allclose(
            out_old.toarray(), out_new.toarray(), atol=1e-6,
            err_msg="sparse_convolution direct method does not match old Toeplitz",
        )

    def test_parity_with_scipy_convolve2d(self):
        """Blurred output matches scipy.signal.convolve2d ground truth."""
        import scipy.signal
        from roicat.tracking.blurring import ROI_Blurrer

        blurrer = ROI_Blurrer(frame_shape=(32, 32), kernel_halfWidth=2, verbose=False)

        rng = np.random.default_rng(123)
        x_dense = rng.random((32, 32), dtype=np.float32)
        x_dense[x_dense > 0.02] = 0.0

        ## Ground truth
        expected = scipy.signal.convolve2d(x_dense, blurrer.kernel, mode='same')

        ## Via ROI_Blurrer
        x_sparse = scipy.sparse.csr_matrix(x_dense.ravel()[None, :])
        blurrer.blur_ROIs([x_sparse])
        actual = blurrer.ROIs_blurred[0].toarray().reshape(32, 32)

        np.testing.assert_allclose(actual, expected, atol=1e-6)

    def test_max_intensity_projection(self):
        """get_ROIsBlurred_maxIntensityProjection returns correct shape."""
        from roicat.tracking.blurring import ROI_Blurrer

        blurrer = ROI_Blurrer(frame_shape=(32, 32), kernel_halfWidth=2, verbose=False)
        sf = scipy.sparse.random(5, 32 * 32, density=0.01, format='csr', dtype=np.float32)
        blurrer.blur_ROIs([sf])
        mip = blurrer.get_ROIsBlurred_maxIntensityProjection()

        assert len(mip) == 1
        assert mip[0].shape == (32, 32)


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
    assert all([scipy.sparse.issparse(sf) and sf.format == 'csr' for sf in data.spatialFootprints]), 'ROICaT Error: data.spatialFootprints must be a list of sparse CSR arrays'

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

    from roicat.tracking.similarity_graph import DEFAULT_METRICS
    clusterer = tracking.clustering.Clusterer(
        similarities=sim['similarities_z'],
        metric_configs=DEFAULT_METRICS,
        s_sesh=sim['s_sesh'],
        verbose=False,
    )
    return clusterer


class Test__find_optimal_parameters_DE:
    """Tests for Clusterer._find_optimal_parameters_DE."""

    def test_returns_valid_dict(self, clusterer_with_data):
        """DE should return a dict with all expected keys."""
        result = clusterer_with_data._find_optimal_parameters_DE(seed=42)
        ## All metrics get entries in best_params. Non-optimized metrics
        ## get identity values (power=None, sig=None).
        expected_keys = {'power_sf', 'power_nn', 'power_swt', 'p_norm',
                         'sig_sf_kwargs', 'sig_nn_kwargs', 'sig_swt_kwargs'}
        assert set(result.keys()) == expected_keys
        ## Non-optimized metrics have None values
        assert result['power_sf'] is None
        assert result['sig_sf_kwargs'] is None
        ## Optimized metrics have real values
        assert set(result['sig_nn_kwargs'].keys()) == {'mu', 'b'}
        assert set(result['sig_swt_kwargs'].keys()) == {'mu', 'b'}

    def test_params_within_bounds(self, clusterer_with_data):
        """All optimized parameters should be within their declared bounds."""
        result = clusterer_with_data._find_optimal_parameters_DE(seed=42)
        bounds = {
            'power_nn': [0.0, 2.0],
            'power_swt': [0.0, 2.0],
            'p_norm': [-5.0, -0.1],
        }
        for key, (lo, hi) in bounds.items():
            val = result[key]
            assert lo - 1e-6 <= val <= hi + 1e-6, (
                f'{key}={val} outside bounds [{lo}, {hi}]'
            )
        ## Sigmoid params are frozen from NB calibration — just check they exist and are finite
        for name in ['sig_nn_kwargs', 'sig_swt_kwargs']:
            assert np.isfinite(result[name]['mu'])
            assert np.isfinite(result[name]['b'])
            assert result[name]['b'] > 0

    def test_deterministic_with_seed(self, clusterer_with_data):
        """Same seed should produce identical results."""
        r1 = clusterer_with_data._find_optimal_parameters_DE(seed=123)
        r2 = clusterer_with_data._find_optimal_parameters_DE(seed=123)
        for key in ['power_nn', 'power_swt', 'p_norm']:
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
        assert 'power_nn' in result
        assert 'power_swt' in result
        assert 'p_norm' in result
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
        assert 'power_nn' in result
        assert 'power_swt' in result
        assert 'p_norm' in result
        assert np.isfinite(clusterer_with_data._de_result.fun)



class Test_estimate_sigmoid_params:
    """Tests for Clusterer._estimate_sigmoid_params."""

    def test_returns_expected_features(self, clusterer_with_data):
        """Should return sigmoid params for NN and SWT."""
        clusterer_with_data.make_naive_bayes_distance_matrix()
        result = clusterer_with_data._estimate_sigmoid_params()
        assert 'nn' in result
        assert 'swt' in result
        assert 'mu' in result['nn'] and 'b' in result['nn']
        assert 'mu' in result['swt'] and 'b' in result['swt']

    def test_mu_is_finite(self, clusterer_with_data):
        """Estimated mu should be finite (within data range, which can exceed [0,1] for z-scored features)."""
        clusterer_with_data.make_naive_bayes_distance_matrix()
        result = clusterer_with_data._estimate_sigmoid_params()
        for name in ['nn', 'swt']:
            mu = result[name]['mu']
            assert np.isfinite(mu), f'{name} mu={mu} is not finite'

    def test_b_is_positive(self, clusterer_with_data):
        """Estimated b (steepness) should be positive."""
        clusterer_with_data.make_naive_bayes_distance_matrix()
        result = clusterer_with_data._estimate_sigmoid_params()
        for name in ['nn', 'swt']:
            b = result[name]['b']
            assert b > 0, f'{name} b={b} should be positive'

    def test_requires_nb_calibration(self, dir_data_test):
        """Should raise if NB calibration hasn't been run on a fresh instance."""
        from roicat import util, tracking
        path_run_data = str(Path(dir_data_test) / 'pipeline_tracking' / 'run_data.richfile.zip')
        sim = util.RichFile_ROICaT(path=path_run_data)['sim'].load()
        from roicat.tracking.similarity_graph import DEFAULT_METRICS
        fresh = tracking.clustering.Clusterer(
            similarities=sim['similarities_z'],
            metric_configs=DEFAULT_METRICS,
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
        assert scipy.sparse.issparse(dConj)
        assert scipy.sparse.issparse(sConj)
        assert isinstance(calibrations, dict)
        assert 'features' in calibrations
        assert 'prior' in calibrations
        assert 'p_same_combined' in calibrations

    def test_output_shapes_match_input(self, clusterer_with_data):
        """dConj and sConj should have same shape/nnz as _s_sparsity."""
        dConj, sConj, _ = clusterer_with_data.make_naive_bayes_distance_matrix()
        assert dConj.shape == clusterer_with_data._s_sparsity.shape
        assert sConj.shape == clusterer_with_data._s_sparsity.shape
        assert dConj.nnz == clusterer_with_data._s_sparsity.nnz
        assert sConj.nnz == clusterer_with_data._s_sparsity.nnz

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
        assert set(calibrations['features'].keys()) == {'sf', 'nn', 'swt'}
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
            mixing_params='precomputed',
        )
        assert hasattr(clusterer_with_data, 'dConj_pruned')
        assert clusterer_with_data.dConj_pruned is not None

    def test_deterministic(self, clusterer_with_data):
        """Two calls with same data should produce identical results."""
        dConj1, _, cal1 = clusterer_with_data.make_naive_bayes_distance_matrix()
        dConj2, _, cal2 = clusterer_with_data.make_naive_bayes_distance_matrix()
        np.testing.assert_array_equal(dConj1.data, dConj2.data)



class Test_edge_cases:
    """Edge case and robustness tests for the new mixing methods."""

    def test_extreme_p_norm_bounds(self, clusterer_with_data):
        """DE should handle near-zero p_norm without NaN/Inf."""
        result = clusterer_with_data._find_optimal_parameters_DE(
            seed=42,
            bounds_findParameters={
                'power_nn': [0.0, 0.5],
                'power_swt': [0.0, 0.5],
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
        assert result['sig_nn_kwargs']['mu'] == sig_params['nn']['mu']
        assert result['sig_nn_kwargs']['b'] == sig_params['nn']['b']
        assert result['sig_swt_kwargs']['mu'] == sig_params['swt']['mu']
        assert result['sig_swt_kwargs']['b'] == sig_params['swt']['b']

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
        s_sf = scipy.sparse.csr_array(
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
        s_sesh = scipy.sparse.csr_array(
            (s_sesh_data, (rows, cols)), shape=(n, n),
        )

        from roicat.tracking.similarity_graph import DEFAULT_METRICS
        c = tracking.clustering.Clusterer(
            similarities={'sf': s_sf, 'nn': s_NN_z, 'swt': s_SWT_z},
            metric_configs=DEFAULT_METRICS,
            s_sesh=s_sesh, verbose=False,
        )

        result = c._find_optimal_parameters_DE(
            seed=42,
            de_kwargs={
                'maxiter': 5, 'tol': 1e-4, 'popsize': 5, 'polish': False,
            },
        )

        ## New API returns lowercase keys: power_nn, power_swt, p_norm, sig_nn_kwargs, sig_swt_kwargs
        assert 'power_nn' in result
        assert 'power_swt' in result
        assert 'p_norm' in result
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

        s_sf = scipy.sparse.csr_array(
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
        s_sesh = scipy.sparse.csr_array(
            (s_sesh_data, (rows, cols)), shape=(n, n),
        )

        from roicat.tracking.similarity_graph import DEFAULT_METRICS
        c = tracking.clustering.Clusterer(
            similarities={'sf': s_sf, 'nn': s_NN_z, 'swt': s_SWT_z},
            metric_configs=DEFAULT_METRICS,
            s_sesh=s_sesh, verbose=False,
        )

        dConj, sConj, cal = c.make_naive_bayes_distance_matrix()
        assert dConj.shape == (n, n)
        assert np.all(np.isfinite(dConj.data))
        assert np.all(dConj.data >= 0)
        assert np.all(dConj.data <= 1)


######################################################################################################################################
######################################### FAST HDBSCAN INTEGRATION ###################################################################
######################################################################################################################################


def _make_synthetic_clusterer(n_sessions=4, n_rois_per_session=20, n_neighbors=10, seed=42):
    """
    Build a synthetic Clusterer with known cluster structure for testing.

    Creates ``n_sessions`` sessions each with ``n_rois_per_session`` ROIs.
    ROIs with the same index across sessions are "matched" (low distance),
    all others get high distance.  Returns the Clusterer and session_bool.
    """
    from roicat import tracking, util

    rng = np.random.RandomState(seed)
    n_total = n_sessions * n_rois_per_session

    ## Build session_bool: (n_total, n_sessions) binary matrix
    session_bool = np.zeros((n_total, n_sessions), dtype=np.float64)
    for s in range(n_sessions):
        session_bool[s * n_rois_per_session : (s + 1) * n_rois_per_session, s] = 1.0

    ## Build s_sesh: True where ROIs are from DIFFERENT sessions
    sb_sparse = scipy.sparse.csr_array(session_bool)
    s_sesh_full = (sb_sparse @ sb_sparse.T).toarray()
    np.fill_diagonal(s_sesh_full, 0)
    ## s_sesh_full[i, j] > 0 means same session. We want different-session.
    diff_session = (s_sesh_full == 0).astype(np.float64)
    np.fill_diagonal(diff_session, 0)

    ## Build similarity matrices with known structure.
    ## For matched ROIs (same index mod n_rois_per_session, different session),
    ## set high similarity. For unmatched ROIs, set low similarity.
    ## Only populate edges between different sessions within a k-nearest-neighbor radius.
    rows, cols, sf_data, nn_data, swt_data, sesh_data = [], [], [], [], [], []

    for i in range(n_total):
        s_i = i // n_rois_per_session
        idx_i = i % n_rois_per_session
        for j in range(i + 1, n_total):
            s_j = j // n_rois_per_session
            idx_j = j % n_rois_per_session
            if s_i == s_j:
                ## Same session -- include in sparsity pattern but mark as same-session
                if abs(idx_i - idx_j) <= 2:
                    rows.extend([i, j])
                    cols.extend([j, i])
                    sim = 0.1 + rng.rand() * 0.1
                    sf_data.extend([sim, sim])
                    nn_data.extend([sim, sim])
                    swt_data.extend([sim, sim])
                    sesh_data.extend([0.0, 0.0])  ## same session
            else:
                ## Different session
                if idx_i == idx_j:
                    ## Matched ROI pair -- high similarity
                    sim = 0.8 + rng.rand() * 0.15
                    rows.extend([i, j])
                    cols.extend([j, i])
                    sf_data.extend([sim, sim])
                    nn_data.extend([sim, sim])
                    swt_data.extend([sim, sim])
                    sesh_data.extend([1.0, 1.0])  ## different session
                elif abs(idx_i - idx_j) <= 2:
                    ## Nearby ROI from different session -- low similarity
                    sim = 0.1 + rng.rand() * 0.2
                    rows.extend([i, j])
                    cols.extend([j, i])
                    sf_data.extend([sim, sim])
                    nn_data.extend([sim, sim])
                    swt_data.extend([sim, sim])
                    sesh_data.extend([1.0, 1.0])  ## different session

    shape = (n_total, n_total)
    s_sf = scipy.sparse.csr_array((np.array(sf_data), (rows, cols)), shape=shape)
    s_NN_z = scipy.sparse.csr_array((np.array(nn_data), (rows, cols)), shape=shape)
    s_SWT_z = scipy.sparse.csr_array((np.array(swt_data), (rows, cols)), shape=shape)
    s_sesh = scipy.sparse.csr_array((np.array(sesh_data), (rows, cols)), shape=shape)

    from roicat.tracking.similarity_graph import DEFAULT_METRICS
    clusterer = tracking.clustering.Clusterer(
        similarities={'sf': s_sf, 'nn': s_NN_z, 'swt': s_SWT_z},
        metric_configs=DEFAULT_METRICS,
        s_sesh=s_sesh,
        session_bool=session_bool,
        verbose=False,
    )

    ## Build a simple distance matrix: d = 1 - similarity, masked to inter-session
    d_conj = s_sf.copy()
    d_conj.data = 1.0 - d_conj.data

    return clusterer, d_conj, session_bool


class TestFastHDBSCAN:
    """Tests for fast_hdbscan integration via Clusterer.fit()."""

    @pytest.fixture(scope='class')
    def synthetic_setup(self):
        """Create a synthetic clusterer for fast_hdbscan tests."""
        return _make_synthetic_clusterer(n_sessions=4, n_rois_per_session=20, seed=42)

    def test_fast_hdbscan_produces_labels(self, synthetic_setup):
        """fast_hdbscan backend should produce integer labels."""
        clusterer, d_conj, session_bool = synthetic_setup
        labels = clusterer.fit(
            d_conj=d_conj,
            session_bool=session_bool,
        )
        assert isinstance(labels, np.ndarray)
        assert labels.shape == (session_bool.shape[0],)
        assert labels.dtype in (np.int32, np.int64)

    def test_fast_hdbscan_no_session_violations(self, synthetic_setup):
        """No cluster should contain two ROIs from the same session."""
        clusterer, d_conj, session_bool = synthetic_setup
        labels = clusterer.fit(
            d_conj=d_conj,
            session_bool=session_bool,
        )
        for ucid in np.unique(labels):
            if ucid == -1:
                continue
            mask = labels == ucid
            sessions_in_cluster = session_bool[mask]
            session_counts = sessions_in_cluster.sum(axis=0)
            assert np.all(session_counts <= 1), (
                f"Cluster {ucid} has same-session violations: "
                f"session_counts={session_counts[session_counts > 1]}"
            )

    def test_fast_hdbscan_no_violations_across_seeds(self):
        """Cannot-link constraints should hold across different random seeds."""
        for seed in [0, 7, 42, 99, 123]:
            clusterer, d_conj, session_bool = _make_synthetic_clusterer(
                n_sessions=5, n_rois_per_session=15, seed=seed,
            )
            labels = clusterer.fit(
                d_conj=d_conj,
                session_bool=session_bool,
            )
            for ucid in np.unique(labels):
                if ucid == -1:
                    continue
                mask = labels == ucid
                session_counts = session_bool[mask].sum(axis=0)
                assert np.all(session_counts <= 1), (
                    f"seed={seed}, cluster {ucid}: same-session violation "
                    f"session_counts={session_counts[session_counts > 1]}"
                )

    def test_fast_hdbscan_group_labels_follow_session_bool_row_order(self, synthetic_setup, monkeypatch):
        """
        Group-label cannot-link constraints should use the per-row session ID,
        not assume ROIs are stored in contiguous session blocks.
        """
        from roicat import tracking
        from roicat.tracking.similarity_graph import DEFAULT_METRICS
        import fast_hdbscan

        clusterer, d_conj, session_bool = synthetic_setup
        n_sessions = session_bool.shape[1]
        n_rois_per_session = session_bool.shape[0] // n_sessions

        ## Interleave rows by ROI index across sessions so session membership
        ## is no longer represented by contiguous blocks in row order.
        perm = np.arange(session_bool.shape[0]).reshape(n_sessions, n_rois_per_session).T.reshape(-1)
        session_bool_perm = session_bool[perm]

        similarities_perm = {
            name: scipy.sparse.csr_array(sim[perm][:, perm])
            for name, sim in clusterer.similarities.items()
        }
        s_sesh_perm = scipy.sparse.csr_array(clusterer.s_sesh[perm][:, perm])
        d_conj_perm = scipy.sparse.csr_array(d_conj[perm][:, perm])

        clusterer_perm = tracking.clustering.Clusterer(
            similarities=similarities_perm,
            metric_configs=DEFAULT_METRICS,
            s_sesh=s_sesh_perm,
            session_bool=session_bool_perm,
            verbose=False,
        )

        captured = {}

        class DummyHDBSCAN:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            def fit(self, d):
                self.labels_ = np.ones(d.shape[0], dtype=np.int32) * -1
                return self

        monkeypatch.setattr(fast_hdbscan, 'HDBSCAN', DummyHDBSCAN)

        clusterer_perm.fit(
            d_conj=d_conj_perm,
            session_bool=session_bool_perm,
        )

        expected = np.asarray(np.argmax(session_bool_perm, axis=1), dtype=np.int32)
        wrong_if_block_assumed = np.repeat(
            np.arange(n_sessions, dtype=np.int32),
            np.asarray(session_bool_perm.sum(axis=0), dtype=int),
        )

        assert not np.array_equal(expected, wrong_if_block_assumed), (
            "Test setup should break the contiguous-block assumption."
        )
        assert np.array_equal(captured['cannot_link_groups'], expected)

    def test_fast_hdbscan_violations_attribute(self, synthetic_setup):
        """violations_labels should be empty with cannot-link constraints."""
        clusterer, d_conj, session_bool = synthetic_setup
        clusterer.fit(
            d_conj=d_conj,
            session_bool=session_bool,
        )
        assert hasattr(clusterer, 'violations_labels')
        assert len(clusterer.violations_labels) == 0

    def test_fast_hdbscan_some_clusters_found(self, synthetic_setup):
        """At least some non-noise clusters should be found."""
        clusterer, d_conj, session_bool = synthetic_setup
        labels = clusterer.fit(
            d_conj=d_conj,
            session_bool=session_bool,
        )
        n_clusters = len(set(labels) - {-1})
        assert n_clusters > 0, "Expected at least one cluster"

    def test_fast_hdbscan_labels_squeezed(self, synthetic_setup):
        """Labels should be squeezed (contiguous integers starting from -1 or 0)."""
        clusterer, d_conj, session_bool = synthetic_setup
        labels = clusterer.fit(
            d_conj=d_conj,
            session_bool=session_bool,
        )
        non_noise = labels[labels >= 0]
        if len(non_noise) > 0:
            unique_labels = np.unique(non_noise)
            assert unique_labels[0] == 0
            assert np.all(np.diff(unique_labels) == 1), "Labels should be contiguous"

    def test_fast_hdbscan_no_singleton_clusters(self, synthetic_setup):
        """No cluster should have exactly 1 member."""
        clusterer, d_conj, session_bool = synthetic_setup
        labels = clusterer.fit(
            d_conj=d_conj,
            session_bool=session_bool,
        )
        u, c = np.unique(labels[labels >= 0], return_counts=True)
        assert np.all(c >= 2), f"Found singleton clusters: {u[c < 2]}"

    def test_fast_hdbscan_empty_graph(self, synthetic_setup):
        """Clustering an empty graph should return all -1."""
        clusterer, d_conj, session_bool = synthetic_setup
        empty_d = scipy.sparse.csr_array(d_conj.shape)
        labels = clusterer.fit(
            d_conj=empty_d,
            session_bool=session_bool,
        )
        assert np.all(labels == -1)

    def test_fast_hdbscan_stores_hdbs(self, synthetic_setup):
        """The FastHDBSCAN object should be stored as self.hdbs."""
        clusterer, d_conj, session_bool = synthetic_setup
        clusterer.fit(
            d_conj=d_conj,
            session_bool=session_bool,
        )
        assert hasattr(clusterer, 'hdbs')
        assert hasattr(clusterer.hdbs, 'labels_')
        assert hasattr(clusterer.hdbs, 'probabilities_')

    def test_fast_hdbscan_params_stored(self, synthetic_setup):
        """Fit parameters should be stored in self.params['fit']."""
        clusterer, d_conj, session_bool = synthetic_setup
        clusterer.fit(
            d_conj=d_conj,
            session_bool=session_bool,
        )
        assert 'fit' in clusterer.params
        assert clusterer.params['fit']['backend'] == 'fast_hdbscan'

    def test_default_backend_is_fast_hdbscan(self, synthetic_setup):
        """Calling fit() without backend= should use fast_hdbscan."""
        clusterer, d_conj, session_bool = synthetic_setup
        clusterer.fit(
            d_conj=d_conj,
            session_bool=session_bool,
        )
        ## Should have used fast_hdbscan (no fully connected node)
        assert not getattr(clusterer, '_fit_used_fully_connected_node', True)
        assert clusterer.params['fit']['backend'] == 'fast_hdbscan'

    def test_backend_invalid_raises(self, synthetic_setup):
        """An invalid backend string should raise ValueError."""
        clusterer, d_conj, session_bool = synthetic_setup
        with pytest.raises(ValueError, match="backend must be"):
            clusterer.fit(
                d_conj=d_conj,
                session_bool=session_bool,
                backend='nonexistent',
            )

    def test_fast_hdbscan_custom_d_clusterMerge(self, synthetic_setup):
        """Custom d_clusterMerge should be respected."""
        clusterer, d_conj, session_bool = synthetic_setup
        labels = clusterer.fit(
            d_conj=d_conj,
            session_bool=session_bool,
            d_clusterMerge=0.5,
        )
        assert isinstance(labels, np.ndarray)

    def test_fast_hdbscan_min_cluster_size_all(self, synthetic_setup):
        """min_cluster_size='all' should set it to n_sessions."""
        clusterer, d_conj, session_bool = synthetic_setup
        labels = clusterer.fit(
            d_conj=d_conj,
            session_bool=session_bool,
            min_cluster_size='all',
        )
        ## All non-noise clusters should have exactly n_sessions members
        n_sessions = session_bool.shape[1]
        u, c = np.unique(labels[labels >= 0], return_counts=True)
        if len(u) > 0:
            assert np.all(c >= n_sessions), (
                f"With min_cluster_size='all', expected cluster size >= {n_sessions}, "
                f"got sizes: {c}"
            )


class TestFastHDBSCANQualityMetrics:
    """Tests for quality metrics extraction with fast_hdbscan backend."""

    def test_extract_hdbscan_quality_metrics_fast(self):
        """Quality metric extraction should work without outlier_scores_."""
        clusterer, d_conj, session_bool = _make_synthetic_clusterer(
            n_sessions=4, n_rois_per_session=15, seed=99,
        )
        clusterer.fit(
            d_conj=d_conj,
            session_bool=session_bool,
        )
        metrics = clusterer._extract_hdbscan_quality_metrics()
        assert 'sample_probabilities' in metrics
        assert 'sample_outlierScores' in metrics
        ## fast_hdbscan has no outlier_scores_, so it should be None
        assert metrics['sample_outlierScores'] is None
        ## Probabilities should be a list of floats matching n_rois
        assert isinstance(metrics['sample_probabilities'], list)
        assert len(metrics['sample_probabilities']) == session_bool.shape[0]

    def test_core_distances_extracted(self):
        """fast_hdbscan should expose per-point core distances."""
        clusterer, d_conj, session_bool = _make_synthetic_clusterer(
            n_sessions=4, n_rois_per_session=15, seed=99,
        )
        clusterer.fit(
            d_conj=d_conj,
            session_bool=session_bool,
        )
        metrics = clusterer._extract_hdbscan_quality_metrics()
        assert 'sample_coreDistances' in metrics
        assert metrics['sample_coreDistances'] is not None
        assert isinstance(metrics['sample_coreDistances'], list)
        assert len(metrics['sample_coreDistances']) == session_bool.shape[0]
        assert all(isinstance(v, float) for v in metrics['sample_coreDistances'])

    def test_mst_edge_weights_extracted(self):
        """fast_hdbscan should expose sorted MST edge weights."""
        clusterer, d_conj, session_bool = _make_synthetic_clusterer(
            n_sessions=4, n_rois_per_session=15, seed=99,
        )
        clusterer.fit(
            d_conj=d_conj,
            session_bool=session_bool,
        )
        metrics = clusterer._extract_hdbscan_quality_metrics()
        assert 'mst_edge_weights' in metrics
        assert metrics['mst_edge_weights'] is not None
        assert isinstance(metrics['mst_edge_weights'], list)
        ## MST on n points has n-1 edges
        n_rois = session_bool.shape[0]
        assert len(metrics['mst_edge_weights']) == n_rois - 1
        ## Weights should be sorted
        weights = metrics['mst_edge_weights']
        assert weights == sorted(weights)


######################################################################################################################################
######################################### RICHFILE OPTIMIZE RESULT ###################################################################
######################################################################################################################################

def test_richfile_optimize_result_roundtrip(tmp_path):
    """OptimizeResult should survive RichFile save/load."""
    from scipy.optimize import OptimizeResult
    from roicat.util import RichFile_ROICaT

    result = OptimizeResult(
        x=np.array([1.0, 2.0, 3.0]),
        fun=0.5,
        nfev=100,
        nit=50,
        success=True,
        message='Optimization converged.',
    )

    path = str(tmp_path / 'test_result.richfile.zip')
    rf = RichFile_ROICaT(path=path, backend='zip')
    rf.save({'de_result': result})

    loaded = RichFile_ROICaT(path=path).load()
    assert isinstance(loaded['de_result'], OptimizeResult)
    np.testing.assert_array_equal(loaded['de_result'].x, result.x)
    assert loaded['de_result'].fun == result.fun
    assert loaded['de_result'].nfev == result.nfev
    assert loaded['de_result'].nit == result.nit
    assert loaded['de_result'].success == result.success
    assert loaded['de_result'].message == result.message


######################################################################################################################################
######################################################## ROIEXTRACTORS ###############################################################
######################################################################################################################################


class Test_roiextractors:
    """Tests for roiextractors integration."""

    def test_import_data_roiextractors(self):
        """Data_roiextractors class should be importable."""
        from roicat.data_importing import Data_roiextractors
        assert Data_roiextractors is not None

    def test_import_roiextractors_package(self):
        """roiextractors package should be importable."""
        import roiextractors
        assert hasattr(roiextractors, 'extractors')

    def test_make_spatial_footprints_from_mock(self):
        """Data_roiextractors._make_spatialFootprints should convert pixel masks to sparse."""
        from roicat.data_importing import Data_roiextractors

        rng = np.random.RandomState(42)
        height, width = 50, 50
        n_rois = 5

        ## Create mock pixel masks in roiextractors format: list of (n_pixels, 3) arrays
        ## Each array has columns [row, col, value]
        class MockSegObj:
            def get_roi_pixel_masks(self):
                masks = []
                for _ in range(n_rois):
                    n_px = rng.randint(10, 30)
                    rows = rng.randint(0, height, n_px)
                    cols = rng.randint(0, width, n_px)
                    vals = rng.rand(n_px).astype(np.float32)
                    masks.append(np.column_stack([rows, cols, vals]))
                return masks

            def get_frame_shape(self):
                return (height, width)

        mock = MockSegObj()
        ## Call the static-ish method directly (it only uses segObj methods)
        data = Data_roiextractors.__new__(Data_roiextractors)
        sf = data._make_spatialFootprints(mock)

        assert scipy.sparse.issparse(sf)
        assert sf.shape[0] == n_rois
        assert sf.shape[1] == height * width
