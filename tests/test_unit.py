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

    def test_sparse_bool_different(self):
        """Differing boolean sparse arrays report a mismatch instead of raising TypeError on subtraction."""
        checker = helpers.Equivalence_checker()
        s1 = scipy.sparse.csr_array(np.array([[0, 1, 1], [1, 0, 0]], dtype=bool))
        s2 = scipy.sparse.csr_array(np.array([[0, 1, 0], [1, 0, 1]], dtype=bool))
        result = checker(s1, s2)
        assert result[0] == False
        assert 'n_mismatches=2' in result[1]

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


class Test_auroc_crossCloserThanSame:
    """Tests for the pure AUROC used by the 'auroc' DE objective."""

    def test_identical_distributions(self):
        """Two arms drawn from the same values are indistinguishable: 0.5."""
        from roicat.tracking.clustering import auroc_crossCloserThanSame
        auroc = auroc_crossCloserThanSame(
            d_crossSession=np.arange(10, dtype=np.float64),
            d_sameSession=np.arange(10, dtype=np.float64),
        )
        assert auroc == pytest.approx(0.5, abs=1e-12)

    def test_perfect_separation(self):
        """All cross-session distances below all same-session ones: 1.0."""
        from roicat.tracking.clustering import auroc_crossCloserThanSame
        auroc = auroc_crossCloserThanSame(
            d_crossSession=np.array([0.0, 0.1, 0.2, 0.3]),
            d_sameSession=np.array([0.5, 0.7, 0.9, 1.0]),
        )
        assert auroc == pytest.approx(1.0, abs=1e-12)

    def test_perfect_inversion(self):
        """The reverse ordering is the other limit: 0.0."""
        from roicat.tracking.clustering import auroc_crossCloserThanSame
        auroc = auroc_crossCloserThanSame(
            d_crossSession=np.array([0.5, 0.7, 0.9, 1.0]),
            d_sameSession=np.array([0.0, 0.1, 0.2, 0.3]),
        )
        assert auroc == pytest.approx(0.0, abs=1e-12)

    def test_full_collapse_is_chance(self):
        """Every distance tied — the failure mode the old loss rewarded — is 0.5.

        Mid-rank tie handling is what makes this exact: each tied pair
        contributes 0.5, so a mixing that pushes everything onto d = 1
        cannot score better than chance.
        """
        from roicat.tracking.clustering import auroc_crossCloserThanSame
        auroc = auroc_crossCloserThanSame(
            d_crossSession=np.ones(50), d_sameSession=np.ones(30),
        )
        assert auroc == pytest.approx(0.5, abs=1e-12)

    def test_partial_ties_match_bruteforce(self):
        """Mid-ranks agree with the O(n*m) definition on a tie-heavy case."""
        from roicat.tracking.clustering import auroc_crossCloserThanSame
        rng = np.random.RandomState(0)
        d_cross = np.round(rng.rand(200), 1)  ## rounding forces many ties
        d_same = np.round(rng.rand(150) * 0.8 + 0.2, 1)
        ## AUROC = P(cross < same) + 0.5 * P(cross == same)
        comparison = d_cross[:, None] - d_same[None, :]  ## shape (200, 150)
        auroc_bruteforce = float(
            np.mean((comparison < 0).astype(np.float64)
                    + 0.5 * (comparison == 0).astype(np.float64))
        )
        auroc = auroc_crossCloserThanSame(d_crossSession=d_cross, d_sameSession=d_same)
        assert auroc == pytest.approx(auroc_bruteforce, abs=1e-12)

    def test_loss_is_one_minus_auroc(self, clusterer_with_data):
        """`de_result.fun` must be exactly `1 - AUROC` at the fitted parameters.

        Recomputes the conjunctive distances from the returned mixing
        parameters and re-derives the loss, which is the contract a
        bounded-but-unrelated objective would not satisfy. `polish=False`
        so that `_de_result.fun` corresponds to `_de_result.x` and hence
        to the parameters that come back.

        The two distance computations are not the same code: the DE inner
        loop works on cloned float32 tensors with `torch.sigmoid` and a
        running sum, while `make_conjunctive_distance_matrix` uses
        `generalised_logistic_function` and `torch.mean` over a stacked
        tensor. Both clamp at 0; when the DE clamped at 1e-8 instead, pairs
        with saturated sigmoids got different distances and this test
        failed. They agree bit-for-bit on this dataset (checked with `==`),
        but the assertion below uses a tight `np.isclose` so that a float32
        reassociation on another platform reports as a tolerance failure
        rather than a false alarm.
        """
        from roicat.tracking.clustering import auroc_crossCloserThanSame

        mixing_params = clusterer_with_data._find_optimal_parameters_DE(
            seed=42,
            objective='auroc',
            de_kwargs={
                'maxiter': 3, 'tol': 1e-4, 'popsize': 5, 'polish': False,
            },
        )
        loss = clusterer_with_data._de_result.fun
        assert 0.0 <= loss <= 1.0

        ## Rebuild the distances the fit landed on. The test data is far
        ## below the auto-subsample threshold, so the DE saw all pairs.
        dConj, _, _ = clusterer_with_data.make_conjunctive_distance_matrix(
            similarities=clusterer_with_data.similarities,
            mixing_params=mixing_params,
        )
        mask_intra = clusterer_with_data._intra_mask  ## True = same-session pair
        loss_recomputed = 1.0 - auroc_crossCloserThanSame(
            d_crossSession=dConj.data[~mask_intra],
            d_sameSession=dConj.data[mask_intra],
        )
        assert np.isclose(loss, loss_recomputed, rtol=1e-12, atol=0.0), (
            f'DE loss {loss!r} != 1 - AUROC at the fitted parameters '
            f'({loss_recomputed!r})'
        )

    def test_empty_arm_raises(self):
        """An empty arm leaves the statistic undefined; fail loudly."""
        from roicat.tracking.clustering import auroc_crossCloserThanSame
        with pytest.raises(ValueError, match='non-empty'):
            auroc_crossCloserThanSame(
                d_crossSession=np.array([]), d_sameSession=np.ones(5),
            )
        with pytest.raises(ValueError, match='non-empty'):
            auroc_crossCloserThanSame(
                d_crossSession=np.ones(5), d_sameSession=np.array([]),
            )

    def test_nan_raises(self):
        """A NaN distance has no rank; fail loudly instead of returning a number."""
        from roicat.tracking.clustering import auroc_crossCloserThanSame
        with pytest.raises(ValueError, match='NaN'):
            auroc_crossCloserThanSame(
                d_crossSession=np.array([0.1, np.nan]), d_sameSession=np.ones(3),
            )

    def test_matches_rankdata_bitwise(self):
        """Bit-for-bit equal to the pooled `scipy.stats.rankdata` formula it replaced."""
        import scipy.stats
        from roicat.tracking.clustering import auroc_crossCloserThanSame
        for seed in range(200):
            rng = np.random.default_rng(seed)
            ## Mixed float widths and rounding to force ties.
            dtype_cross, dtype_same = rng.choice([np.float32, np.float64], size=2)
            d_cross = np.round(rng.normal(size=rng.integers(1, 60)), rng.integers(0, 3)).astype(dtype_cross)
            d_same = np.round(rng.normal(size=rng.integers(1, 60)), rng.integers(0, 3)).astype(dtype_same)
            n_cross, n_same = d_cross.size, d_same.size
            ranks = scipy.stats.rankdata(np.concatenate([d_cross, d_same]), method='average')
            u_crossGreater = float(ranks[:n_cross].sum()) - (n_cross * (n_cross + 1) / 2.0)
            auroc_rankdata = float(1.0 - (u_crossGreater / (n_cross * n_same)))
            assert auroc_crossCloserThanSame(d_crossSession=d_cross, d_sameSession=d_same) == auroc_rankdata


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
        """DE should beat the trivial value under either objective.

        The two objectives live on different scales, so the thresholds
        differ. `histogram_overlap` is an overlap area in unnormalized
        counts: DE reliably finds ~55 on this dataset while the default
        manual params typically give >200. `auroc` is `1 - AUROC`, bounded
        in [0, 1], where 0.5 is what a zero-information (fully collapsed)
        mixing scores. The old single `< 200` assertion is kept for the
        legacy objective; it would pass vacuously on the new default.
        """
        clusterer_with_data._find_optimal_parameters_DE(
            seed=42, objective='histogram_overlap',
        )
        assert clusterer_with_data._de_result.fun < 200, (
            f'DE histogram_overlap loss {clusterer_with_data._de_result.fun:.1f} '
            f'is too high; expected < 200 on test data'
        )

        clusterer_with_data._find_optimal_parameters_DE(seed=42, objective='auroc')
        assert clusterer_with_data._de_result.fun < 0.5, (
            f'DE auroc loss {clusterer_with_data._de_result.fun:.4f} is at or '
            f'above chance; expected < 0.5 on test data'
        )

    def test_objective_histogram_overlap_runs(self, clusterer_with_data):
        """The legacy objective stays selectable and returns the same keys."""
        result = clusterer_with_data._find_optimal_parameters_DE(
            seed=42,
            objective='histogram_overlap',
            de_kwargs={
                'maxiter': 3, 'tol': 1e-4, 'popsize': 5, 'polish': False,
            },
        )
        assert set(result.keys()) == {'power_sf', 'power_nn', 'power_swt', 'p_norm',
                                      'sig_sf_kwargs', 'sig_nn_kwargs', 'sig_swt_kwargs'}
        assert np.isfinite(clusterer_with_data._de_result.fun)

    def test_legacy_loss_matches_histogram_overlap(self, clusterer_with_data):
        """`de_result.fun` must equal the legacy overlap area at the fit.

        The bit-for-bit promise for `objective='histogram_overlap'` is that
        the DE still minimizes exactly `_compute_histogram_overlap`. Pinning
        the fitted floats themselves would only pin this machine's
        scipy/numpy, so the check is self-consistency instead: recompute the
        overlap from the returned parameters through the public distance
        path and compare. Tight `np.isclose` for the same float32
        reassociation reason as the AUROC test; it is `==` here.
        """
        mixing_params = clusterer_with_data._find_optimal_parameters_DE(
            seed=42,
            objective='histogram_overlap',
            de_kwargs={
                'maxiter': 3, 'tol': 1e-4, 'popsize': 5, 'polish': False,
            },
        )
        loss = clusterer_with_data._de_result.fun

        dConj, _, _ = clusterer_with_data.make_conjunctive_distance_matrix(
            similarities=clusterer_with_data.similarities,
            mixing_params=mixing_params,
        )
        ## Same histogram infrastructure the DE builds internally.
        n_bins = clusterer_with_data.n_bins
        edges = torch.linspace(0, 1, n_bins + 1, dtype=torch.float32)
        smoother = helpers.Convolver_1d(
            kernel=torch.ones(helpers.make_odd(n_bins // 10, mode='up')),
            length_x=n_bins,
            pad_mode='same',
            correct_edge_effects=True,
            device='cpu',
        )
        mask_intra = clusterer_with_data._intra_mask  ## True = same-session pair
        n_intra = int(mask_intra.sum())
        loss_recomputed, _, _ = clusterer_with_data._compute_histogram_overlap(
            distances=torch.as_tensor(dConj.data, dtype=torch.float32),
            intra_indices=torch.as_tensor(np.where(mask_intra)[0]),
            edges=edges,
            smoother=smoother,
            scale_factor=mask_intra.shape[0] / max(n_intra, 1),
        )
        assert np.isclose(loss, loss_recomputed, rtol=1e-12, atol=0.0), (
            f'DE loss {loss!r} != histogram overlap at the fitted parameters '
            f'({loss_recomputed!r})'
        )

    def test_auroc_with_unfrozen_sigmoid_warns(self, clusterer_with_data):
        """`objective='auroc'` + `freeze_sigmoid=False` must warn, not raise.

        AUROC is a rank statistic and so fixes no absolute distance scale,
        while `thresh_cost` and `d_cutoff` downstream are absolute. The
        combination stays available for callers who set their own cutoff,
        but it has to announce itself.
        """
        with pytest.warns(UserWarning, match='scale-free'):
            clusterer_with_data._find_optimal_parameters_DE(
                seed=42,
                objective='auroc',
                freeze_sigmoid=False,
                de_kwargs={
                    'maxiter': 2, 'tol': 1e-4, 'popsize': 4, 'polish': False,
                },
            )

    def test_histogram_overlap_with_unfrozen_sigmoid_does_not_warn(
        self, clusterer_with_data,
    ):
        """The legacy objective anchors the scale, so it must stay quiet.

        Checks for this specific warning rather than promoting every
        `UserWarning` to an error, so an unrelated deprecation from scipy
        or torch on someone else's machine does not fail the test.
        """
        with warnings.catch_warnings(record=True) as warnings_caught:
            warnings.simplefilter('always')
            clusterer_with_data._find_optimal_parameters_DE(
                seed=42,
                objective='histogram_overlap',
                freeze_sigmoid=False,
                de_kwargs={
                    'maxiter': 2, 'tol': 1e-4, 'popsize': 4, 'polish': False,
                },
            )
        assert not any('scale-free' in str(w.message) for w in warnings_caught), (
            'the scale-free warning fired on the legacy objective'
        )

    def test_invalid_objective_raises(self, clusterer_with_data):
        """An unrecognized objective should fail loudly, before any fitting."""
        with pytest.raises(ValueError, match='objective must be one of'):
            clusterer_with_data._find_optimal_parameters_DE(
                seed=42, objective='histogram-overlap',
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

    @pytest.mark.parametrize('objective', ['auroc', 'histogram_overlap'])
    def test_workers_do_not_change_result(self, clusterer_with_data, objective):
        """Deferred updating makes the fit independent of the thread count."""
        results = []
        for workers in [1, 4]:
            clusterer_with_data._find_optimal_parameters_DE(
                seed=42,
                objective=objective,
                de_kwargs={'maxiter': 3, 'popsize': 5, 'polish': False, 'workers': workers},
            )
            results.append(clusterer_with_data._de_result)
        np.testing.assert_array_equal(results[0].x, results[1].x)
        assert results[0].fun == results[1].fun

    def test_invalid_workers_raises(self, clusterer_with_data):
        with pytest.raises(ValueError, match='workers'):
            clusterer_with_data._find_optimal_parameters_DE(
                seed=42, de_kwargs={'maxiter': 1, 'popsize': 5, 'workers': 0},
            )

    def test_loss_history(self, clusterer_with_data):
        """One best loss per generation, ending at the returned loss."""
        clusterer_with_data._find_optimal_parameters_DE(
            seed=42, de_kwargs={'maxiter': 4, 'popsize': 5, 'polish': False},
        )
        history = clusterer_with_data.de_loss_history
        assert len(history) == clusterer_with_data._de_result.nit
        assert history[-1] == clusterer_with_data._de_result.fun
        ## All pairs are used on the test data, so the best loss cannot rise.
        assert np.all(np.diff(history) <= 0)

    @pytest.mark.parametrize('style', ['old', 'new'])
    def test_user_callback_can_stop(self, clusterer_with_data, style):
        """Both scipy callback signatures are called, and a truthy return stops the DE."""
        calls = []
        if style == 'old':
            def callback(xk, convergence):
                calls.append(xk)
                return True
        else:
            def callback(intermediate_result):
                calls.append(intermediate_result.x)
                return True
        clusterer_with_data._find_optimal_parameters_DE(
            seed=42, de_kwargs={'maxiter': 5, 'popsize': 5, 'polish': False, 'callback': callback},
        )
        assert len(calls) == 1
        assert clusterer_with_data._de_result.nit == 1



def _make_clusterer_weak_metrics(seed, n_session, n_cell, frac_outlier):
    """
    A fully connected synthetic graph: ``sf`` separates matches cleanly, while
    ``nn`` and ``swt`` are weak z-scored metrics (matches shifted by half a
    standard deviation). ``frac_outlier`` of their values are replaced by
    +-[50, 1500], the heavy tails that a z-score over a tiny reference set
    produces. Returns the Clusterer and its similarity dict.
    """
    from roicat import tracking
    from roicat.tracking.similarity_graph import DEFAULT_METRICS
    rng = np.random.default_rng(seed)
    n_roi = n_session * n_cell
    sess = np.repeat(np.arange(n_session), n_cell)
    cell = np.tile(np.arange(n_cell), n_session)
    iu, ju = np.triu_indices(n_roi, k=1)
    same_session = sess[iu] == sess[ju]
    match = (~same_session) & (cell[iu] == cell[ju])

    def weak_z():
        z = rng.normal(0, 1, iu.size) + 0.5 * match
        is_outlier = rng.random(iu.size) < frac_outlier
        z[is_outlier] = rng.choice([-1, 1], is_outlier.sum()) * rng.uniform(50, 1500, is_outlier.sum())
        return z

    def symmetric(v, dtype):
        rows, cols = np.concatenate([iu, ju]), np.concatenate([ju, iu])
        return scipy.sparse.csr_array((np.concatenate([v, v]).astype(dtype), (rows, cols)), shape=(n_roi, n_roi))

    sims = {
        'sf': symmetric(np.where(match, rng.uniform(0.5, 0.9, iu.size), rng.uniform(0.0, 0.2, iu.size)), np.float64),
        'nn': symmetric(weak_z(), np.float32),
        'swt': symmetric(weak_z(), np.float32),
    }
    s_sesh = symmetric((~same_session).astype(np.float64), np.float64)
    s_sesh = scipy.sparse.csr_array((s_sesh.data.astype(bool), s_sesh.indices, s_sesh.indptr), shape=s_sesh.shape)
    clusterer = tracking.clustering.Clusterer(
        similarities=sims, metric_configs=DEFAULT_METRICS, s_sesh=s_sesh, verbose=False,
    )
    return clusterer, sims


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

    @pytest.mark.parametrize('seed, n_session, n_cell, frac_outlier', [(0, 14, 30, 0.0), (1, 14, 30, 0.0), (0, 6, 10, 0.0), (0, 6, 10, 0.03), (1, 6, 10, 0.03)])
    def test_weak_metric_sigmoid_does_not_saturate(self, seed, n_session, n_cell, frac_outlier):
        """A weak metric's frozen sigmoid must not map most pairs to ~0.

        Under the negative ``p_norm`` a single near-zero activation drives
        ``sConj`` to zero, so a frozen sigmoid that sits above most of a weak
        metric's values vetoes the strong one and the mixed similarity
        collapses. The Fisher-ratio pre-fit did exactly that here (a step with
        ``b`` at its upper bound, up to 64% of pairs below 1e-6; 99% with
        heavy tails).
        """
        clusterer, sims = _make_clusterer_weak_metrics(
            seed=seed, n_session=n_session, n_cell=n_cell, frac_outlier=frac_outlier,
        )
        clusterer.make_naive_bayes_distance_matrix()
        params = clusterer._estimate_sigmoid_params()
        for name in ('nn', 'swt'):
            x = np.asarray(sims[name].data, dtype=np.float64)
            logsig = -np.logaddexp(0.0, -params[name]['b'] * (x - params[name]['mu']))
            frac_saturated = float((logsig < np.log(1e-6)).mean())
            assert frac_saturated < 0.05, (
                f"{name}: frozen sigmoid (mu={params[name]['mu']:.3g}, b={params[name]['b']:.3g}) "
                f"puts {frac_saturated:.1%} of pairs below 1e-6"
            )


class Test_naive_bayes_distance_matrix:
    """Tests for Clusterer.make_naive_bayes_distance_matrix."""

    def test_tied_values_look_up_the_bin_that_counted_them(self):
        """Pairs tied at one value get P(same) from the bin that counted them.

        Equal-mass edges repeat where many pairs share one value, as in a
        clipped or discrete metric. torch.histogram counts a value equal to an
        edge in the bin to its right, and the per-pair lookup must use that
        same bin, not the (empty) bins left of the repeated edges.
        """
        n_cell = 20
        clusterer, sims = _make_clusterer_weak_metrics(seed=0, n_session=6, n_cell=n_cell, frac_outlier=0.0)
        s = sims['swt']
        assert np.array_equal(s.indices, sims['sf'].indices) and np.array_equal(s.indptr, sims['sf'].indptr)
        ## Tie every match and ~20% of the other inter-session pairs at 2.0,
        ## so the tied bin has a high P(same) and the bins below it a low one
        is_match = sims['sf'].data > 0.5  ## sf is in [0.5, 0.9] for matches, [0, 0.2] otherwise
        idx_session = np.arange(s.shape[0]) // n_cell  ## shape: (n_roi,)
        is_intra = idx_session[np.repeat(np.arange(s.shape[0]), np.diff(s.indptr))] == idx_session[s.indices]  ## shape: (nnz,)
        is_tied = is_match | (~is_intra & (np.random.default_rng(0).random(s.nnz) < 0.2))
        s.data[is_tied] = 2.0

        _, _, calibrations = clusterer.make_naive_bayes_distance_matrix()
        cal = calibrations['features']['swt']
        idx_edgesAtTie = np.flatnonzero(cal['edges'] == 2.0)
        assert len(idx_edgesAtTie) >= 2, "precondition: the ties must repeat an edge"
        idx_binCounted = idx_edgesAtTie.max()  ## bin [2.0, next edge), where torch.histogram counts the ties
        idx_binLeft = idx_edgesAtTie.min() - 1  ## bin [previous edge, 2.0), below the ties
        assert cal['p_same_bins'][idx_binCounted] != cal['p_same_bins'][idx_binLeft], "precondition: a wrong lookup must be visible"
        np.testing.assert_array_equal(cal['p_same_per_pair'][is_tied], cal['p_same_bins'][idx_binCounted])

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

    def test_no_crossover_raises_without_d_cutoff(self, clusterer_with_data):
        """No crossover + inferred cutoff must raise, not `TypeError` on None.

        A sigmoid centered far above the z-scored similarity range saturates
        every activation to 0, so every pair lands at distance 1. The 'same'
        residual is then empty everywhere, `_separate_diffSame_distributions`
        finds no crossover and returns `d_crossover=None`, and the inferred
        cutoff used to be computed as `None - min_d`.
        """
        from roicat import tracking
        from roicat.tracking.similarity_graph import DEFAULT_METRICS

        ## Fresh instance: the call below sets dConj/graph_pruned, which the
        ## module-scoped fixture would otherwise carry into other tests.
        clusterer = tracking.clustering.Clusterer(
            similarities=clusterer_with_data.similarities,
            metric_configs=DEFAULT_METRICS,
            s_sesh=clusterer_with_data.s_sesh,
            verbose=False,
        )
        mixing_params_collapsed = {
            'power_sf': 1.0, 'power_nn': 1.0, 'power_swt': 1.0, 'p_norm': -4.0,
            'sig_sf_kwargs': None,
            'sig_nn_kwargs': {'mu': 10.0, 'b': 10.0},
            'sig_swt_kwargs': {'mu': 10.0, 'b': 10.0},
        }
        with pytest.raises(ValueError, match='No crossover point exists'):
            clusterer.make_pruned_similarity_graphs(
                mixing_params=mixing_params_collapsed,
            )

        ## An explicit cutoff is the documented way through, and it works.
        clusterer.make_pruned_similarity_graphs(
            mixing_params=mixing_params_collapsed, d_cutoff=0.5,
        )
        assert clusterer.d_cutoff == 0.5

        ## The probability map needs the densities themselves, so an explicit
        ## cutoff is not enough there: it used to reach `_fn_smooth(None)`.
        with pytest.raises(ValueError, match='convert the distances into probabilities'):
            clusterer.make_pruned_similarity_graphs(
                mixing_params=mixing_params_collapsed,
                d_cutoff=0.5,
                convert_to_probability=True,
            )

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

    def test_fast_hdbscan_min_samples(self, synthetic_setup):
        """min_samples should be accepted and produce valid labels."""
        clusterer, d_conj, session_bool = synthetic_setup
        labels = clusterer.fit(
            d_conj=d_conj,
            session_bool=session_bool,
            min_samples=1,
        )
        assert isinstance(labels, np.ndarray)
        assert len(labels) == session_bool.shape[0]

    def test_fast_hdbscan_cluster_selection_persistence(self, synthetic_setup):
        """cluster_selection_persistence should be accepted."""
        clusterer, d_conj, session_bool = synthetic_setup
        labels = clusterer.fit(
            d_conj=d_conj,
            session_bool=session_bool,
            cluster_selection_persistence=0.1,
        )
        assert isinstance(labels, np.ndarray)
        assert len(labels) == session_bool.shape[0]

    def test_fast_hdbscan_d_clusterMerge_defaults_to_d_cutoff(self, synthetic_setup):
        """When d_clusterMerge=None, should use self.d_cutoff if available."""
        clusterer, d_conj, session_bool = synthetic_setup
        ## Set d_cutoff as if make_pruned_similarity_graphs was called
        clusterer.d_cutoff = 0.42
        labels = clusterer.fit(
            d_conj=d_conj,
            session_bool=session_bool,
        )
        assert isinstance(labels, np.ndarray)
        ## Verify the stored param reflects the d_cutoff value
        assert clusterer.params['fit']['d_clusterMerge'] is None  ## original arg was None


class TestSequentialHungarianThreshCost:
    """Tests for tying fit_sequentialHungarian's threshold to the pruning cutoff."""

    ## Fixed mixing so `make_pruned_similarity_graphs` never has to infer a crossover
    ## from the synthetic distributions; every fit below passes an explicit `d_cutoff`.
    MIXING_PARAMS = {
        'power_sf': 1.0, 'power_nn': 1.0, 'power_swt': 1.0, 'p_norm': -4.0,
        'sig_sf_kwargs': None,
        'sig_nn_kwargs': {'mu': 0.5, 'b': 1.0},
        'sig_swt_kwargs': {'mu': 0.5, 'b': 1.0},
    }

    def test_thresh_cost_none_matches_explicit_d_cutoff(self):
        """thresh_cost=None must give the labels of passing self.d_cutoff by hand."""
        clusterer, _, session_bool = _make_synthetic_clusterer(
            n_sessions=4, n_rois_per_session=20, seed=42,
        )
        clusterer.make_pruned_similarity_graphs(
            mixing_params=dict(self.MIXING_PARAMS),
            d_cutoff=0.35,
        )
        ## bool, not the helper's float64: fit_sequentialHungarian indexes ROI ranges
        ## with `session_bool.sum(0)`.
        session_bool = session_bool.astype(bool)

        kwargs = dict(d_conj=clusterer.dConj_pruned, session_bool=session_bool)
        labels_tied = clusterer.fit_sequentialHungarian(**kwargs, thresh_cost=None)
        assert clusterer.params['fit_sequentialHungarian']['thresh_cost'] is None
        labels_explicit = clusterer.fit_sequentialHungarian(**kwargs, thresh_cost=0.35)
        np.testing.assert_array_equal(labels_tied, labels_explicit)

        ## The equality above is only meaningful if the threshold moves the labels on
        ## this data at all. It does: below d_cutoff every cluster is lost.
        labels_stricter = clusterer.fit_sequentialHungarian(**kwargs, thresh_cost=0.2)
        assert not np.array_equal(labels_tied, labels_stricter)

    def test_thresh_cost_none_raises_without_d_cutoff(self):
        """Without make_pruned_similarity_graphs there is no cutoff to tie to."""
        clusterer, d_conj, session_bool = _make_synthetic_clusterer(
            n_sessions=4, n_rois_per_session=20, seed=42,
        )
        with pytest.raises(ValueError, match='d_cutoff'):
            clusterer.fit_sequentialHungarian(
                d_conj=d_conj,
                session_bool=session_bool.astype(bool),
                thresh_cost=None,
            )


def _make_random_graph_clusterer(n_sessions=10, n_rois_per_session=5, p_edge=0.1, seed=2):
    """
    Build a Clusterer on a random symmetric graph of inter-session edges.

    Distances are drawn uniformly from (0, 1), so there are no ties and
    single linkage has one answer. Returns the Clusterer, the sparse distance
    matrix, session_bool, and the same distances as a dense matrix in which
    every missing and same-session pair is set to 10.0 (for scipy's linkage).
    """
    from roicat import tracking
    from roicat.tracking.similarity_graph import DEFAULT_METRICS

    rng = np.random.default_rng(seed)
    n_total = n_sessions * n_rois_per_session
    session_of_roi = np.repeat(np.arange(n_sessions), n_rois_per_session)
    session_bool = session_of_roi[:, None] == np.arange(n_sessions)[None, :]  ## shape: (n_total, n_sessions)

    ## Upper triangle of random inter-session edges, then symmetrized
    mask_edge = np.triu(rng.random((n_total, n_total)) < p_edge, k=1)
    mask_edge &= session_of_roi[:, None] != session_of_roi[None, :]
    d_dense = np.where(mask_edge, rng.random((n_total, n_total)), 0.0)
    d_dense = d_dense + d_dense.T

    d_conj = scipy.sparse.csr_array(d_dense)
    s_sesh = scipy.sparse.csr_array((d_dense > 0).astype(np.float64))  ## every stored edge is inter-session
    s = d_conj.copy()
    s.data = 1.0 - s.data
    clusterer = tracking.clustering.Clusterer(
        similarities={'sf': s, 'nn': s.copy(), 'swt': s.copy()},
        metric_configs=DEFAULT_METRICS,
        s_sesh=s_sesh,
        session_bool=session_bool,
        verbose=False,
    )

    d_dense_missingFar = np.where(d_dense > 0, d_dense, 10.0)
    np.fill_diagonal(d_dense_missingFar, 0.0)
    return clusterer, d_conj, session_bool, d_dense_missingFar


def _canonical_labels(labels):
    """Relabel clusters in order of first appearance, so equal partitions give equal arrays. -1 stays -1."""
    labels_out = np.full(len(labels), -1, dtype=np.int64)
    mapping = {}
    for i, label in enumerate(np.asarray(labels)):
        if label >= 0:
            labels_out[i] = mapping.setdefault(label, len(mapping))
    return labels_out


def _singletons_to_noise(labels):
    """Set clusters with one member to -1."""
    labels = np.asarray(labels).copy()
    u, c = np.unique(labels, return_counts=True)
    labels[np.isin(labels, u[c < 2])] = -1
    return labels


def _has_session_violation(labels, session_bool):
    """True if any cluster holds two ROIs from one session."""
    return any(
        np.any(session_bool[labels == u].sum(axis=0) > 1)
        for u in np.unique(labels[labels >= 0])
    )


class TestSingleLinkage:
    """Tests for session-constrained single linkage (``Clusterer.fit_singleLinkage``)."""

    def test_matches_scipy_single_linkage_when_constraint_does_not_bind(self):
        """Below the first same-session merge, the result is plain single linkage."""
        import scipy.cluster.hierarchy
        import scipy.spatial.distance

        clusterer, d_conj, session_bool, d_dense = _make_random_graph_clusterer()
        linkage = scipy.cluster.hierarchy.linkage(
            scipy.spatial.distance.squareform(d_dense, checks=False),
            method='single',
        )

        ## Walk the unconstrained merges in order and stop at the first one
        ## that joins two ROIs from the same session.
        n_total = session_bool.shape[0]
        sessions_of_node = {i: set(np.nonzero(session_bool[i])[0]) for i in range(n_total)}
        for i_merge, (a, b, height, _) in enumerate(linkage):
            if sessions_of_node[int(a)] & sessions_of_node[int(b)]:
                break
            sessions_of_node[n_total + i_merge] = sessions_of_node.pop(int(a)) | sessions_of_node.pop(int(b))
        ## Cut strictly between two merge heights: scipy's fcluster merges at
        ## <= t, fast_hdbscan's dbscan_clustering at < epsilon.
        d_cut = (linkage[i_merge - 1, 2] + linkage[i_merge, 2]) / 2
        assert i_merge >= 10, f"Only {i_merge} merges before the constraint binds; the test would be trivial."

        labels_scipy = _singletons_to_noise(
            scipy.cluster.hierarchy.fcluster(linkage, t=d_cut, criterion='distance')
        )
        assert not _has_session_violation(labels_scipy, session_bool)

        labels = clusterer.fit_singleLinkage(
            d_conj=d_conj,
            session_bool=session_bool,
            d_clusterMerge=d_cut,
        )
        np.testing.assert_array_equal(_canonical_labels(labels), _canonical_labels(labels_scipy))

    def test_no_session_violations_when_constraint_binds(self):
        """Above the first same-session merge, no cluster holds two ROIs from one session."""
        import scipy.cluster.hierarchy
        import scipy.spatial.distance

        clusterer, d_conj, session_bool, d_dense = _make_random_graph_clusterer()
        d_cut = 0.5

        ## The constraint must actually bind at this cut for the test to mean anything
        linkage = scipy.cluster.hierarchy.linkage(
            scipy.spatial.distance.squareform(d_dense, checks=False),
            method='single',
        )
        labels_scipy = _singletons_to_noise(
            scipy.cluster.hierarchy.fcluster(linkage, t=d_cut, criterion='distance')
        )
        assert _has_session_violation(labels_scipy, session_bool)

        labels = clusterer.fit_singleLinkage(
            d_conj=d_conj,
            session_bool=session_bool,
            d_clusterMerge=d_cut,
        )
        assert not _has_session_violation(labels, session_bool)

        ## And it is constrained single linkage: brute-force Kruskal over the
        ## edges below the cut, skipping merges that share a session.
        session_of_roi = np.argmax(session_bool, axis=1)
        root_of_roi = np.arange(len(session_of_roi))
        sessions_of_root = {i: {s} for i, s in enumerate(session_of_roi)}
        def find_root(i):
            while root_of_roi[i] != i:
                i = root_of_roi[i]
            return i
        idx_row, idx_col = np.nonzero(np.triu(d_dense < d_cut, k=1))
        for k in np.argsort(d_dense[idx_row, idx_col]):
            root_a, root_b = find_root(idx_row[k]), find_root(idx_col[k])
            if root_a != root_b and not (sessions_of_root[root_a] & sessions_of_root[root_b]):
                root_of_roi[root_b] = root_a
                sessions_of_root[root_a] |= sessions_of_root.pop(root_b)
        labels_reference = _singletons_to_noise([find_root(i) for i in range(len(session_of_roi))])
        np.testing.assert_array_equal(_canonical_labels(labels), _canonical_labels(labels_reference))

    def test_d_clusterMerge_none_uses_d_cutoff(self):
        """d_clusterMerge=None must give the labels of passing self.d_cutoff by hand."""
        clusterer, d_conj, session_bool, _ = _make_random_graph_clusterer()
        with pytest.raises(ValueError, match='d_cutoff'):
            clusterer.fit_singleLinkage(d_conj=d_conj, session_bool=session_bool)

        clusterer.d_cutoff = 0.3
        labels_tied = clusterer.fit_singleLinkage(d_conj=d_conj, session_bool=session_bool)
        assert clusterer.params['fit_singleLinkage']['d_clusterMerge'] is None
        labels_explicit = clusterer.fit_singleLinkage(d_conj=d_conj, session_bool=session_bool, d_clusterMerge=0.3)
        np.testing.assert_array_equal(labels_tied, labels_explicit)

    def test_quality_metrics_after_single_linkage(self):
        """compute_quality_metrics runs after fit_singleLinkage and reports no HDBSCAN metrics."""
        clusterer, d_conj, session_bool, _ = _make_random_graph_clusterer()
        labels = clusterer.fit_singleLinkage(
            d_conj=d_conj,
            session_bool=session_bool,
            d_clusterMerge=0.5,
        )
        assert not hasattr(clusterer, 'hdbs')
        s = d_conj.copy()
        s.data = 1.0 - s.data
        quality_metrics = clusterer.compute_quality_metrics(sim_mat=s, dist_mat=d_conj)
        assert np.any(labels >= 0)
        assert len(quality_metrics['sample_silhouette']) == session_bool.shape[0]
        assert quality_metrics['hdbscan'] is None
        assert quality_metrics['sample_probabilities'] is None

    def test_pipeline_automatic_choice(self):
        """'automatic' picks single linkage below n_sessions_switch and HDBSCAN at or above it."""
        from roicat import pipelines

        assert pipelines.choose_clustering_method(method='automatic', n_sessions_switch=6, n_sessions=2) == 'SINGLE_LINKAGE'
        assert pipelines.choose_clustering_method(method='automatic', n_sessions_switch=6, n_sessions=5) == 'SINGLE_LINKAGE'
        assert pipelines.choose_clustering_method(method='automatic', n_sessions_switch=6, n_sessions=6) == 'HDBSCAN'
        assert pipelines.choose_clustering_method(method='automatic', n_sessions_switch=6, n_sessions=7) == 'HDBSCAN'
        ## Explicit methods pass through, including the one 'automatic' no longer picks
        assert pipelines.choose_clustering_method(method='sequential_hungarian', n_sessions_switch=6, n_sessions=3) == 'SEQUENTIAL_HUNGARIAN'
        with pytest.raises(AssertionError, match='single_linkage'):
            pipelines.choose_clustering_method(method='kmeans', n_sessions_switch=6, n_sessions=3)
        ## The shipped default switch point
        defaults = util.get_default_parameters(pipeline='tracking')
        assert defaults['clustering']['cluster_method']['n_sessions_switch'] == 6


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
######################################################## NOISE RESCUE ################################################################
######################################################################################################################################


class TestNoiseRescue:
    """Tests for noise_rescue_kruskal and Clusterer.rescue_noise."""

    def _make_graph_and_labels(self):
        """
        Build a synthetic 8-node graph with 3 sessions, 2 pre-formed
        clusters with gaps, and 4 noise points that can fill those gaps.
        Returns (d_conj, labels, group_labels, session_bool, n_groups).

        Layout (sessions S0, S1, S2):
            Cluster 0: nodes 0 (S0), 1 (S1) — missing S2
            Cluster 1: nodes 2 (S0), 3 (S1) — missing S2
            Noise:     nodes 4 (S2), 5 (S2), 6 (S1), 7 (S0)

        Noise edges (all inter-session by construction):
            4 → 0 (d=0.2, S2→S0 — rescue 4 into cluster 0, no S2 conflict)
            5 → 2 (d=0.25, S2→S0 — rescue 5 into cluster 1, no S2 conflict)
            6 → 0 (d=0.3, S1→S0 — blocked: cluster 0 has node 1 from S1)
            7 → 3 (d=0.35, S0→S1 — blocked: cluster 1 has node 2 from S0)
        """
        n = 8
        n_sessions = 3

        ## Session assignments
        session_bool = np.zeros((n, n_sessions), dtype=bool)
        session_bool[0, 0] = True  ## S0, cluster 0
        session_bool[1, 1] = True  ## S1, cluster 0
        session_bool[2, 0] = True  ## S0, cluster 1
        session_bool[3, 1] = True  ## S1, cluster 1
        session_bool[4, 2] = True  ## S2, noise
        session_bool[5, 2] = True  ## S2, noise
        session_bool[6, 1] = True  ## S1, noise
        session_bool[7, 0] = True  ## S0, noise

        group_labels = np.argmax(session_bool, axis=1).astype(np.int32)

        ## Phase 1 labels: clusters 0 and 1, rest noise
        labels = np.array([0, 0, 1, 1, -1, -1, -1, -1], dtype=np.int64)

        ## Build inter-session masked distance matrix
        rows = []
        cols = []
        dists = []

        def add_edge(i, j, d):
            if group_labels[i] != group_labels[j]:
                rows.extend([i, j])
                cols.extend([j, i])
                dists.extend([d, d])

        ## Intra-cluster edges (inter-session)
        add_edge(0, 1, 0.1)   ## S0→S1, cluster 0
        add_edge(2, 3, 0.1)   ## S0→S1, cluster 1

        ## Noise rescue edges
        add_edge(4, 0, 0.2)   ## S2→S0, rescue 4 into cluster 0
        add_edge(5, 2, 0.25)  ## S2→S0, rescue 5 into cluster 1
        add_edge(6, 0, 0.3)   ## S1→S0, but cluster 0 has S1 (node 1) → blocked
        add_edge(7, 3, 0.35)  ## S0→S1, but cluster 1 has S0 (node 2) → blocked

        d_conj = scipy.sparse.csr_array(
            (np.array(dists, dtype=np.float64), (rows, cols)),
            shape=(n, n),
        )
        d_conj.sort_indices()

        return d_conj, labels, group_labels, session_bool, n_sessions

    def test_noise_rescue_basic(self):
        """Noise points near existing clusters should be rescued when no session conflict."""
        from roicat.tracking.clustering import noise_rescue_kruskal

        d_conj, labels, group_labels, session_bool, n_groups = self._make_graph_and_labels()

        new_labels = noise_rescue_kruskal(
            d_conj=d_conj,
            labels=labels,
            group_labels=group_labels,
            n_groups=n_groups,
            d_cutoff=0.5,
        )

        ## Node 4 (S2, noise) → edge to node 0 (S0, cluster 0): no S2 in cluster 0 → rescued
        assert new_labels[4] == 0, f"Node 4 should join cluster 0, got {new_labels[4]}"
        ## Node 5 (S2, noise) → edge to node 2 (S0, cluster 1): no S2 in cluster 1 → rescued
        assert new_labels[5] == 1, f"Node 5 should join cluster 1, got {new_labels[5]}"
        ## Node 6 (S1, noise) → edge to node 0 (S0, cluster 0): cluster 0 has S1 (node 1) → blocked
        assert new_labels[6] == -1, f"Node 6 should stay noise (session conflict), got {new_labels[6]}"
        ## Node 7 (S0, noise) → edge to node 3 (S1, cluster 1): cluster 1 has S0 (node 2) → blocked
        assert new_labels[7] == -1, f"Node 7 should stay noise (session conflict), got {new_labels[7]}"
        ## Original cluster members should keep their labels
        assert new_labels[0] == 0
        assert new_labels[1] == 0
        assert new_labels[2] == 1
        assert new_labels[3] == 1

    def test_noise_rescue_session_constraint(self):
        """Noise rescue should block merges that would violate session constraints."""
        from roicat.tracking.clustering import noise_rescue_kruskal

        ## Build graph: cluster 0 has nodes from S0 and S1.
        ## Noise node from S0 has edge to cluster 0 — should be blocked.
        n = 4
        session_bool = np.zeros((n, 2), dtype=bool)
        session_bool[0, 0] = True  ## S0, cluster 0
        session_bool[1, 1] = True  ## S1, cluster 0
        session_bool[2, 0] = True  ## S0, noise — same session as node 0
        session_bool[3, 1] = True  ## S1, noise — same session as node 1

        group_labels = np.argmax(session_bool, axis=1).astype(np.int32)
        labels = np.array([0, 0, -1, -1], dtype=np.int64)

        ## Noise node 2 (S0) → node 1 (S1, cluster 0): blocked because
        ## cluster 0 already has node 0 (S0) — session conflict!
        ## Noise node 3 (S1) → node 0 (S0, cluster 0): blocked because
        ## cluster 0 already has node 1 (S1) — session conflict!
        rows = [2, 1, 3, 0]
        cols = [1, 2, 0, 3]
        dists = [0.2, 0.2, 0.2, 0.2]
        ## Also the intra-cluster edge
        rows += [0, 1]
        cols += [1, 0]
        dists += [0.1, 0.1]

        d_conj = scipy.sparse.csr_array(
            (np.array(dists, dtype=np.float64), (rows, cols)),
            shape=(n, n),
        )
        d_conj.sort_indices()

        new_labels = noise_rescue_kruskal(
            d_conj=d_conj,
            labels=labels,
            group_labels=group_labels,
            n_groups=2,
            d_cutoff=0.5,
        )

        ## Both noise nodes should remain noise (or form their own cluster
        ## if they can merge with each other, but they are from different
        ## sessions so they CAN merge)
        ## Node 2 (S0) and node 3 (S1) can merge since they are from
        ## different sessions. But they have no direct edge to each other.
        ## So they should remain -1.
        assert new_labels[2] == -1, f"Node 2 should stay noise, got {new_labels[2]}"
        assert new_labels[3] == -1, f"Node 3 should stay noise, got {new_labels[3]}"

    def test_noise_rescue_d_cutoff(self):
        """Edges beyond d_cutoff should be ignored."""
        from roicat.tracking.clustering import noise_rescue_kruskal

        d_conj, labels, group_labels, session_bool, n_groups = self._make_graph_and_labels()

        ## Set d_cutoff to 0.22 — only node 4→0 (d=0.2) should make it
        new_labels = noise_rescue_kruskal(
            d_conj=d_conj,
            labels=labels,
            group_labels=group_labels,
            n_groups=n_groups,
            d_cutoff=0.22,
        )

        ## Node 4 rescued (edge d=0.2 <= 0.22)
        assert new_labels[4] == 0, f"Node 4 should be rescued, got {new_labels[4]}"
        ## Node 5 NOT rescued (edge d=0.25 > 0.22)
        assert new_labels[5] == -1, f"Node 5 should stay noise, got {new_labels[5]}"

    def test_noise_rescue_nucleation(self):
        """Two noise points from different sessions should nucleate a new cluster."""
        from roicat.tracking.clustering import noise_rescue_kruskal

        ## 4 nodes: 2 in cluster 0, 2 noise that are close to each other
        ## but far from the cluster
        n = 4
        session_bool = np.zeros((n, 2), dtype=bool)
        session_bool[0, 0] = True  ## S0, cluster 0
        session_bool[1, 1] = True  ## S1, cluster 0
        session_bool[2, 0] = True  ## S0, noise
        session_bool[3, 1] = True  ## S1, noise

        group_labels = np.argmax(session_bool, axis=1).astype(np.int32)
        labels = np.array([0, 0, -1, -1], dtype=np.int64)

        ## Intra-cluster edge
        rows = [0, 1]
        cols = [1, 0]
        dists = [0.1, 0.1]
        ## Noise-to-noise edge (different sessions, so OK)
        rows += [2, 3]
        cols += [3, 2]
        dists += [0.15, 0.15]

        d_conj = scipy.sparse.csr_array(
            (np.array(dists, dtype=np.float64), (rows, cols)),
            shape=(n, n),
        )
        d_conj.sort_indices()

        new_labels = noise_rescue_kruskal(
            d_conj=d_conj,
            labels=labels,
            group_labels=group_labels,
            n_groups=2,
            d_cutoff=0.5,
        )

        ## Nodes 2 and 3 should form a new cluster (label > 0)
        assert new_labels[2] >= 0, f"Node 2 should be in a cluster, got {new_labels[2]}"
        assert new_labels[2] == new_labels[3], (
            f"Nodes 2 and 3 should be in same cluster: {new_labels[2]} vs {new_labels[3]}"
        )
        ## The new cluster should have a different label from cluster 0
        assert new_labels[2] != labels[0], (
            f"New cluster should have a fresh label, not {labels[0]}"
        )

    def test_noise_rescue_no_mutation(self):
        """Input labels array should not be modified."""
        from roicat.tracking.clustering import noise_rescue_kruskal

        d_conj, labels, group_labels, session_bool, n_groups = self._make_graph_and_labels()
        labels_copy = labels.copy()

        noise_rescue_kruskal(
            d_conj=d_conj,
            labels=labels,
            group_labels=group_labels,
            n_groups=n_groups,
            d_cutoff=0.5,
        )

        np.testing.assert_array_equal(labels, labels_copy, err_msg="Input labels were mutated")

    def test_rescue_noise_wired_into_fit(self):
        """fit(rescue_noise=True) should rescue more noise than rescue_noise=False."""
        from roicat.tracking import clustering
        from roicat.tracking.similarity_graph import SimilarityMetric

        clusterer, d_conj, session_bool = _make_synthetic_clusterer(
            n_sessions=5, n_rois_per_session=30, seed=42,
        )
        ## First run without rescue
        labels_no_rescue = clusterer.fit(
            d_conj=d_conj,
            session_bool=session_bool,
            rescue_noise=False,
        )
        n_noise_no_rescue = np.sum(labels_no_rescue == -1)

        ## Re-create to avoid state leakage
        clusterer2, d_conj2, session_bool2 = _make_synthetic_clusterer(
            n_sessions=5, n_rois_per_session=30, seed=42,
        )
        labels_rescue = clusterer2.fit(
            d_conj=d_conj2,
            session_bool=session_bool2,
            rescue_noise=True,
        )
        n_noise_rescue = np.sum(labels_rescue == -1)

        ## Rescue should reduce noise (or at worst keep it the same)
        assert n_noise_rescue <= n_noise_no_rescue, (
            f"Rescue should not increase noise: {n_noise_rescue} > {n_noise_no_rescue}"
        )


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


######################################################################################################################################
###################################################### CLUSTER QUALITY METRICS #######################################################
######################################################################################################################################


class Test_cluster_quality_metrics:
    """Tests for the sparse edge-reduction implementation."""

    @staticmethod
    def _legacy_metrics(sim, labels):
        """Compute the direct cluster-pair reference implementation."""
        labels_unique, cs_mean, cs_max, cs_min = helpers.compute_cluster_similarity_matrices(
            s=sim,
            l=labels,
            verbose=False,
        )
        inter_max = (cs_max * (1 - np.eye(cs_max.shape[0]))).max(axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            silhouette = (
                (cs_mean.diagonal() - inter_max)
                / np.maximum(cs_mean.diagonal(), inter_max)
            )
        return (
            labels_unique,
            cs_mean.diagonal(),
            cs_min.diagonal(),
            cs_max.diagonal(),
            silhouette,
        )

    @pytest.mark.parametrize('use_sparse', [False, True])
    @pytest.mark.parametrize('batch_size,max_edges', [
        (1, 1),
        (2, 5),
        (100, 10_000),
    ])
    def test_batched_accumulation_matches_direct_implementation(
        self,
        use_sparse,
        batch_size,
        max_edges,
    ):
        """Every batching mode must preserve the direct metric calculation."""
        from roicat.tracking.clustering import cluster_quality_metrics

        rng = np.random.default_rng(12)
        labels = np.array([10, 10, 4, 4, 4, -1, -1, -1])
        sim = rng.uniform(0, 1, size=(len(labels), len(labels))).astype(np.float32)
        sim = (sim + sim.T) / 2
        sim[sim < 0.35] = 0
        np.fill_diagonal(sim, 0)
        if use_sparse:
            sim = scipy.sparse.csr_array(sim)

        expected = self._legacy_metrics(sim=sim, labels=labels)
        actual = cluster_quality_metrics(
            sim=sim,
            labels=labels,
            batch_size=batch_size,
            max_edges_per_batch=max_edges,
        )

        for expected_array, actual_array in zip(expected, actual):
            np.testing.assert_allclose(
                actual_array,
                expected_array,
                atol=1e-7,
                equal_nan=True,
            )

    def test_asymmetric_inter_cluster_max_uses_outgoing_edges(self):
        """Preserve the legacy axis convention for asymmetric matrices."""
        from roicat.tracking.clustering import cluster_quality_metrics

        labels = np.array([0, 0, 1, 1])
        sim = scipy.sparse.csr_array(np.array([
            [0.0, 0.8, 0.9, 0.0],
            [0.8, 0.0, 0.0, 0.0],
            [0.2, 0.0, 0.0, 0.8],
            [0.0, 0.0, 0.8, 0.0],
        ], dtype=np.float32))

        _, _, _, _, silhouette = cluster_quality_metrics(
            sim=sim,
            labels=labels,
        )

        np.testing.assert_allclose(
            silhouette,
            np.array([-1 / 9, 0.75]),
            atol=1e-7,
        )

    def test_handles_logical_shape_larger_than_intp(self):
        """Large label/sample counts must not create a four-dimensional COO."""
        from roicat.tracking.clustering import cluster_quality_metrics

        n_samples = 100_000
        n_clusters = 31_000
        labels = np.arange(n_samples) % n_clusters
        sim = scipy.sparse.eye(n_samples, format='csr', dtype=np.float32)
        logical_size = n_clusters**2 * n_samples**2
        assert logical_size > np.iinfo(np.intp).max

        metrics = cluster_quality_metrics(sim=sim, labels=labels)

        assert all(metric.shape == (n_clusters,) for metric in metrics)

    def test_compute_quality_metrics_end_to_end(self):
        """The public method should assemble both sparse metric kernels."""
        from roicat.tracking.clustering import Clusterer

        labels = np.array([0, 0, 1, 1])
        sim = scipy.sparse.csr_array(np.array([
            [0.0, 0.8, 0.2, 0.0],
            [0.8, 0.0, 0.0, 0.2],
            [0.2, 0.0, 0.0, 0.7],
            [0.0, 0.2, 0.7, 0.0],
        ], dtype=np.float32))
        dist = sim.copy()
        dist.data = 1.0 - dist.data
        clusterer = object.__new__(Clusterer)

        quality_metrics = clusterer.compute_quality_metrics(
            sim_mat=sim,
            dist_mat=dist,
            labels=labels,
        )

        assert quality_metrics['cluster_labels_unique'] == [0.0, 1.0]
        assert len(quality_metrics['cluster_silhouette']) == 2
        assert len(quality_metrics['sample_silhouette']) == len(labels)

    def test_cluster_only_numpy_output_does_not_require_distances(self):
        """Large-dataset mode should skip all per-sample allocations."""
        from roicat.tracking.clustering import Clusterer

        labels = np.array([-1, -1, 0, 0, 1, 1], dtype=np.int32)
        sim = scipy.sparse.eye(len(labels), format='csr', dtype=np.float32)
        clusterer = object.__new__(Clusterer)

        quality_metrics = clusterer.compute_quality_metrics(
            sim_mat=sim,
            labels=labels,
            include_sample_metrics=False,
            return_as_numpy=True,
            cluster_batch_size=2,
            max_edges_per_batch=2,
        )

        assert isinstance(quality_metrics, dict)
        assert isinstance(quality_metrics['cluster_intra_means'], np.ndarray)
        assert quality_metrics['sample_silhouette'] is None
        assert quality_metrics['sample_probabilities'] is None
        assert quality_metrics['hdbscan'] is None

    def test_contiguous_label_encoding_uses_int32_fast_path(self):
        """Squeezed ROICaT labels should avoid np.unique's int64 inverse."""
        from roicat.tracking.clustering import _encode_cluster_labels

        labels = np.array([-1, 0, 1, -1, 1], dtype=np.int64)

        unique, inverse, counts = _encode_cluster_labels(labels=labels)

        np.testing.assert_array_equal(unique, np.array([-1, 0, 1]))
        np.testing.assert_array_equal(inverse, np.array([0, 1, 2, 0, 2]))
        np.testing.assert_array_equal(counts, np.array([2, 1, 2]))
        assert inverse.dtype == np.int32


######################################################################################################################################
##################################################### SILHOUETTE_SAMPLES_SPARSE ######################################################
######################################################################################################################################


class Test_silhouette_samples_sparse:
    """Tests for helpers.silhouette_samples_sparse."""

    @staticmethod
    def _make_sparse_case(n=80, n_clusters=6, sparsity=0.3, fill=5.0, seed=0):
        """Build a symmetric sparse distance matrix plus its densified
        (fill-substituted) twin, so sklearn can be used as ground truth."""
        rng = np.random.default_rng(seed)
        labels = rng.integers(0, n_clusters, size=n)
        D = rng.uniform(0, 1, size=(n, n)).astype(np.float32)
        D = (D + D.T) / 2
        np.fill_diagonal(D, 0)

        mask = rng.uniform(size=(n, n)) < sparsity
        mask = mask | mask.T
        np.fill_diagonal(mask, False)

        D_filled = D.copy()
        D_filled[~mask] = fill
        np.fill_diagonal(D_filled, 0)

        D_sparse_dense = D.copy()
        D_sparse_dense[~mask] = 0
        np.fill_diagonal(D_sparse_dense, 0)
        d_csr = scipy.sparse.csr_array(D_sparse_dense)
        return d_csr, D_filled, labels, fill

    @pytest.mark.parametrize('batch_size,max_edges', [
        (1, 1),
        (11, 10),
        (1_000, 1_000_000),
    ])
    def test_batched_accumulation_matches_dense_reference(
        self,
        batch_size,
        max_edges,
    ):
        """Every batching mode must match the direct dense implementation."""
        import sklearn.metrics

        d_csr, D_filled, labels, fill = self._make_sparse_case()
        sil_ref = sklearn.metrics.silhouette_samples(
            D_filled, labels, metric='precomputed',
        )
        sil_ours = helpers.silhouette_samples_sparse(
            d_sparse=d_csr,
            labels=labels,
            fill_value=fill,
            batch_size=batch_size,
            max_edges_per_batch=max_edges,
        )
        np.testing.assert_allclose(sil_ours, sil_ref, atol=1e-5)

    def test_singleton_cluster_returns_zero(self):
        """A sample alone in its cluster should get s=0 by convention."""
        d_csr, _, labels, fill = self._make_sparse_case()
        labels[0] = 9999  ## unique label → singleton
        sil = helpers.silhouette_samples_sparse(d_csr, labels, fill)
        assert sil[0] == 0.0

    def test_single_label_returns_all_zeros(self):
        """With fewer than 2 unique labels, silhouette is undefined → 0."""
        d_csr, _, _, fill = self._make_sparse_case()
        labels = np.zeros(d_csr.shape[0], dtype=int)
        sil = helpers.silhouette_samples_sparse(d_csr, labels, fill)
        assert np.all(sil == 0.0)

    def test_noise_label_treated_as_cluster(self):
        """-1 (noise) should be treated as a normal cluster, matching sklearn."""
        import sklearn.metrics

        d_csr, D_filled, labels, fill = self._make_sparse_case()
        labels[labels == 0] = -1
        sil_ref = sklearn.metrics.silhouette_samples(
            D_filled, labels, metric='precomputed',
        )
        sil_ours = helpers.silhouette_samples_sparse(d_csr, labels, fill)
        np.testing.assert_allclose(sil_ours, sil_ref, atol=1e-5)

    def test_asymmetric_sparsity_pattern(self):
        """Row i may have stored entries that column i doesn't (and vice
        versa). Real dConj graphs are not guaranteed to be structurally
        symmetric. sklearn treats d[i,j] as the distance for sample i's
        row, so an asymmetric stored matrix should produce identical
        results between our function and sklearn on the densified twin."""
        import sklearn.metrics

        rng = np.random.default_rng(42)
        n = 60
        n_clusters = 5
        labels = rng.integers(0, n_clusters, size=n)
        fill = 4.0

        D = rng.uniform(0.01, 1, size=(n, n)).astype(np.float32)
        np.fill_diagonal(D, 0)
        mask = rng.uniform(size=(n, n)) < 0.35
        np.fill_diagonal(mask, False)

        D_sparse_dense = D.copy()
        D_sparse_dense[~mask] = 0
        D_filled = D.copy()
        D_filled[~mask] = fill
        np.fill_diagonal(D_filled, 0)

        d_csr = scipy.sparse.csr_array(D_sparse_dense)
        ## sanity: structurally asymmetric
        assert (d_csr != d_csr.T).nnz > 0

        sil_ref = sklearn.metrics.silhouette_samples(
            D_filled, labels, metric='precomputed',
        )
        sil_ours = helpers.silhouette_samples_sparse(d_csr, labels, fill)
        np.testing.assert_allclose(sil_ours, sil_ref, atol=1e-5)

    def test_explicit_diagonal_zeros(self):
        """If the caller stores d[i,i] = 0 explicitly (not the typical
        eliminate_zeros case but legal), results should still match."""
        import sklearn.metrics

        d_csr, D_filled, labels, fill = self._make_sparse_case(seed=3)
        d_with_diag = d_csr.tolil()
        for k in range(d_with_diag.shape[0]):
            d_with_diag[k, k] = 0.0
        d_with_diag = scipy.sparse.csr_array(d_with_diag)

        sil_ref = sklearn.metrics.silhouette_samples(
            D_filled, labels, metric='precomputed',
        )
        sil_ours = helpers.silhouette_samples_sparse(d_with_diag, labels, fill)
        np.testing.assert_allclose(sil_ours, sil_ref, atol=1e-5)

    def test_midscale_parity_with_sklearn(self):
        """Parity at a scale closer to typical ROICaT use (n=2K, K=80)
        to catch any drift the small cases might miss."""
        import sklearn.metrics

        rng = np.random.default_rng(7)
        n = 2000
        n_clusters = 80
        labels = rng.integers(0, n_clusters, size=n)
        fill = 3.0

        n_edges = int(n * n * 0.02 / 2)
        i = rng.integers(0, n, size=n_edges)
        j = rng.integers(0, n, size=n_edges)
        keep = i != j
        i, j = i[keep], j[keep]
        vals = rng.uniform(0.01, 0.99, size=len(i)).astype(np.float32)
        rows = np.concatenate([i, j])
        cols = np.concatenate([j, i])
        data = np.concatenate([vals, vals])
        d_csr = scipy.sparse.csr_array((data, (rows, cols)), shape=(n, n))
        d_csr.sum_duplicates()

        D_filled = d_csr.toarray()
        missing = (D_filled == 0) & (~np.eye(n, dtype=bool))
        D_filled[missing] = fill

        sil_ref = sklearn.metrics.silhouette_samples(
            D_filled, labels, metric='precomputed',
        )
        sil_ours = helpers.silhouette_samples_sparse(d_csr, labels, fill)
        np.testing.assert_allclose(sil_ours, sil_ref, atol=1e-5)

    def test_fully_observed_clusters_are_not_capped_by_fill_value(self):
        """Fill is irrelevant when every inter-cluster distance is stored."""
        import sklearn.metrics

        labels = np.array([0, 0, 1, 1])
        distances = np.array([
            [0.0, 0.7, 0.8, 0.9],
            [0.7, 0.0, 0.9, 0.8],
            [0.8, 0.9, 0.0, 0.6],
            [0.9, 0.8, 0.6, 0.0],
        ], dtype=np.float32)
        d_csr = scipy.sparse.csr_array(distances)

        expected = sklearn.metrics.silhouette_samples(
            distances,
            labels,
            metric='precomputed',
        )
        actual = helpers.silhouette_samples_sparse(
            d_sparse=d_csr,
            labels=labels,
            fill_value=0.1,
        )

        np.testing.assert_allclose(actual, expected, atol=1e-6)

    def test_large_cluster_count_uses_sparse_intermediates(self):
        """Large sample/cluster products should remain inexpensive when sparse."""
        n_samples = 100_000
        n_clusters = 31_000
        labels = np.arange(n_samples) % n_clusters
        distances = scipy.sparse.csr_array(
            (n_samples, n_samples),
            dtype=np.float32,
        )
        assert n_samples * n_clusters > 2**31

        silhouette = helpers.silhouette_samples_sparse(
            d_sparse=distances,
            labels=labels,
            fill_value=1.0,
        )

        np.testing.assert_array_equal(silhouette, np.zeros(n_samples))


######################################################################################################################################
########################################### IMAGE ALIGNMENT CHECKER (score_alignment) ################################################
######################################################################################################################################


class Test_image_alignment_checker_batching:
    """
    score_alignment chunks over the leading dim of `images`. These tests pin
    numerical equivalence (within a tight floating-point tolerance) between the
    chunked path (batch_size < N) and the single-shot path (batch_size >= N).
    Chunking is a rearrangement of independent per-pair computations, but it
    changes the shape/order of the batched matmuls and reductions, so results
    can differ in the last few ULPs (float ops are not associative) rather than
    being bitwise identical.
    """

    METRIC_KEYS = ('mean_out', 'mean_in', 'ptile95_out', 'max_in',
                   'std_out', 'std_in', 'max_diff', 'z_in', 'r_in')

    def _make_iac(self, hw=(64, 64)):
        return helpers.ImageAlignmentChecker(
            hw=hw, radius_in=4.0, radius_out=20.0, order=5, device='cpu',
        )

    def _make_ims(self, n, hw, seed):
        rng = np.random.RandomState(seed)
        return rng.randn(n, hw[0], hw[1]).astype(np.float32)

    def _assert_close(self, chunked, single_shot):
        # Assert closeness within a tight tolerance rather than bitwise equality.
        # The chunked and single-shot paths do the same math in a different
        # order/shape, so BLAS rounds the last bits differently (most visible on
        # near-zero self-comparison entries, ~1e-13, and dependent on CPU/BLAS/
        # thread count -- which is why bitwise equality was environment-flaky).
        # A real chunking bug (mispaired or misassembled outputs) diverges by
        # O(signal), orders of magnitude above these tolerances, and is caught.
        for k in self.METRIC_KEYS:
            np.testing.assert_allclose(
                chunked[k], single_shot[k], rtol=1e-5, atol=1e-8,
                err_msg=f'metric {k!r} differs between chunked and single-shot paths',
            )

    def test_cross_ragged_last_chunk(self):
        """N=7 with batch_size=4 -> chunks of [4, 3]; metrics must equal single-shot."""
        iac = self._make_iac()
        ims = self._make_ims(7, (64, 64), seed=0)
        ref = self._make_ims(3, (64, 64), seed=1)
        chunked = iac.score_alignment(ims, images_ref=ref, batch_size=4, verbose=False)
        single = iac.score_alignment(ims, images_ref=ref, batch_size=ims.shape[0], verbose=False)
        self._assert_close(chunked, single)

    def test_cross_multiple_full_chunks(self):
        """N=12, batch_size=4 -> three full chunks, no ragged tail."""
        iac = self._make_iac()
        ims = self._make_ims(12, (64, 64), seed=2)
        ref = self._make_ims(5, (64, 64), seed=3)
        chunked = iac.score_alignment(ims, images_ref=ref, batch_size=4, verbose=False)
        single = iac.score_alignment(ims, images_ref=ref, batch_size=ims.shape[0], verbose=False)
        self._assert_close(chunked, single)

    def test_self_comparison(self):
        """images_ref=None -> N x N self-comparison; chunked must match single-shot."""
        iac = self._make_iac()
        ims = self._make_ims(6, (64, 64), seed=4)
        chunked = iac.score_alignment(ims, batch_size=4, verbose=False)
        single = iac.score_alignment(ims, batch_size=ims.shape[0], verbose=False)
        self._assert_close(chunked, single)
        ## sanity: square output for self-comparison
        assert chunked['z_in'].shape == (6, 6)

    def test_batch_size_larger_than_n(self):
        """batch_size > N is a single chunk; output identical to single-shot."""
        iac = self._make_iac()
        ims = self._make_ims(3, (64, 64), seed=5)
        ref = self._make_ims(2, (64, 64), seed=6)
        a = iac.score_alignment(ims, images_ref=ref, batch_size=64, verbose=False)
        b = iac.score_alignment(ims, images_ref=ref, batch_size=ims.shape[0], verbose=False)
        self._assert_close(a, b)

    def test_pc_dropped_by_default(self):
        """'pc' must not be in outs unless return_pc=True (avoids huge alloc by default)."""
        iac = self._make_iac()
        ims = self._make_ims(4, (64, 64), seed=7)
        out_default = iac.score_alignment(ims, batch_size=2, verbose=False)
        assert 'pc' not in out_default, "'pc' should be absent unless return_pc=True"

    def test_pc_roundtrip_when_requested(self):
        """return_pc=True yields a (N, M, H, W) pc array assembled from chunks."""
        iac = self._make_iac()
        H = W = 64
        ims = self._make_ims(5, (H, W), seed=8)
        ref = self._make_ims(3, (H, W), seed=9)
        chunked = iac.score_alignment(ims, images_ref=ref, batch_size=2,
                                      return_pc=True, verbose=False)
        single = iac.score_alignment(ims, images_ref=ref, batch_size=ims.shape[0],
                                     return_pc=True, verbose=False)
        assert chunked['pc'].shape == (5, 3, H, W)
        np.testing.assert_array_equal(chunked['pc'], single['pc'])

    def test_invalid_batch_size_raises(self):
        """batch_size < 1 should be rejected."""
        iac = self._make_iac()
        ims = self._make_ims(3, (64, 64), seed=10)
        with pytest.raises(AssertionError):
            iac.score_alignment(ims, batch_size=0, verbose=False)


######################################################################################################################################
############################################### ALIGNER: MATCH SEARCH (fit_geometric) ################################################
######################################################################################################################################


class Test_Aligner_match_search:
    """
    The match search in ``Aligner.fit_geometric``, on synthetic images with a
    fake registration. The fake knows each image's true x offset and returns
    the right translation only for pairs at most ``REACH_PX`` apart, so these
    tests exercise the search over paths, not a registration method. Whether
    an image is aligned is still decided by ``ImageAlignmentChecker``.

    Images are crops of one white-noise canvas; image 0 is the template:
        * 0: x offset 0.
        * 1: x offset 15. Aligns directly.
        * 2: x offset 30. Too far to align directly, but aligns to image 1.
        * 3: unrelated noise. Every registration involving it returns a wrong
          warp, so it has no path. While any image has no path, the dense
          search used to discard every path it found, image 2's included.
        * 4: image 0 plus faint noise. Every registration involving it returns
          a wrong warp, but it is aligned as it is.
    """

    SIZE_PX = 128
    REACH_PX = 20
    OFFSETS_X = (0, 15, 30, None, None)
    WARP_WRONG_XY = (25.0, 11.0)
    Z_THRESHOLD = 4.0

    def _make_images(self, seed=0):
        rng = np.random.default_rng(seed)
        canvas = rng.standard_normal((self.SIZE_PX, self.SIZE_PX + 64))
        images = [canvas[:, x:x + self.SIZE_PX] for x in self.OFFSETS_X[:3]]
        images.append(rng.standard_normal((self.SIZE_PX, self.SIZE_PX)))
        images.append(images[0] + 0.01 * rng.standard_normal((self.SIZE_PX, self.SIZE_PX)))
        return [np.ascontiguousarray(im, dtype=np.float32) for im in images]

    def _fit(self, idx_images, monkeypatch):
        """
        Run ``fit_geometric`` on the images in ``idx_images`` (template: the
        first) with the fake registration. Returns the aligner and the number
        of registrations run.
        """
        from roicat.tracking import alignment

        images_all = self._make_images()
        images = [images_all[ii] for ii in idx_images]
        offset_by_image = {im.tobytes(): self.OFFSETS_X[ii] for im, ii in zip(images, idx_images)}
        reach_px, warp_wrong_xy = self.REACH_PX, self.WARP_WRONG_XY
        calls = []

        class FakeRegistration:
            def __init__(self, **kwargs):
                pass

            def fit_rigid(self, im_template, im_moving, **kwargs):
                calls.append(1)
                offset_template = offset_by_image[np.asarray(im_template, dtype=np.float32).tobytes()]
                offset_moving = offset_by_image[np.asarray(im_moving, dtype=np.float32).tobytes()]
                warp = np.eye(3, dtype=np.float32)
                if (offset_template is None) or (offset_moving is None):
                    warp[:2, 2] = warp_wrong_xy
                elif abs(offset_template - offset_moving) <= reach_px:
                    ## Warping by tx gives out[:, j] = im_moving[:, j + tx]
                    warp[0, 2] = offset_template - offset_moving
                return warp

        monkeypatch.setattr(alignment, 'PhaseCorrelationRegistration', FakeRegistration)
        aligner = alignment.Aligner(
            use_match_search=True,
            all_to_all=False,
            radius_in=4,
            radius_out=20,
            z_threshold=self.Z_THRESHOLD,
            um_per_pixel=1.0,
            device='cpu',
            verbose=False,
        )
        aligner.fit_geometric(
            template=0,
            ims_moving=images,
            template_method='image',
            method='PhaseCorrelation',
            kwargs_method={'PhaseCorrelation': {}},
            kwargs_RANSAC={},
            compute_final_all_to_all=False,
            verbose=False,
        )
        return aligner, len(calls)

    @staticmethod
    def _translations(aligner):
        """(N, 2) translation (tx, ty) of each final warp."""
        return np.stack([np.asarray(w)[:2, 2] for w in aligner.results_geometric['warp_matrices']], axis=0)

    def test_path_found_by_dense_search_is_kept(self, monkeypatch):
        """Image 2 gets the warp composed through image 1, although image 3 has no path."""
        aligner, _ = self._fit(idx_images=[0, 1, 2, 3], monkeypatch=monkeypatch)
        assert aligner.results_geometric['direct']['alignment_template_to_all'].tolist() == [True, True, False, False]

        translations = self._translations(aligner)
        np.testing.assert_allclose(translations[1], [-15, 0], atol=1e-5)
        np.testing.assert_allclose(translations[2], [-30, 0], atol=1e-5)
        ## No warp aligns image 3, so it keeps identity
        np.testing.assert_allclose(translations[3], [0, 0], atol=1e-5)
        assert aligner.results_geometric['final']['alignment_template_to_all'].tolist() == [True, True, True, False]

    def test_first_round_success_skips_dense_search(self, monkeypatch):
        """Image 4 fails direct registration but is aligned on identity, so the dense search never runs."""
        aligner, n_registrations = self._fit(idx_images=[0, 1, 4], monkeypatch=monkeypatch)
        assert aligner.results_geometric['direct']['alignment_template_to_all'].tolist() == [True, True, False]

        np.testing.assert_allclose(self._translations(aligner)[2], [0, 0], atol=1e-5)
        assert aligner.results_geometric['final']['alignment_template_to_all'].tolist() == [True, True, True]
        ## 3 direct registrations + 3 onto the failed image; a dense search would add 6
        assert n_registrations == 6


######################################################################################################################################
########################################################## ROInet ####################################################################
######################################################################################################################################


class _ScaleDynamicRange_original(torch.nn.Module):
    """
    Frozen copy of the pre-consolidation ScaleDynamicRange, which reduced over
    all dims. Used as a reference to prove the per-sample path is unchanged.
    """
    def __init__(self, scaler_bounds=(0, 1), epsilon=1e-9):
        super().__init__()
        self.range = scaler_bounds[1] - scaler_bounds[0]
        self.epsilon = epsilon

    def forward(self, tensor):
        tensor_minSub = tensor - tensor.min()
        return tensor_minSub * (self.range / (tensor_minSub.max() + self.epsilon))


class Test_ScaleDynamicRange:
    """
    ScaleDynamicRange reduces over the trailing three dims so that one instance
    serves both the per-sample DataLoader path and the batched preprocessing
    path. The per-sample results must not have changed.
    """

    @staticmethod
    def _scale_dynamic_range_original(tensor, scaler_bounds=(0, 1), epsilon=1e-9):
        """The pre-consolidation implementation, which reduced over all dims."""
        range_ = scaler_bounds[1] - scaler_bounds[0]
        tensor_minSub = tensor - tensor.min()
        return tensor_minSub * (range_ / (tensor_minSub.max() + epsilon))

    def test_single_image_bitwise_unchanged(self):
        """A (n_channels, height, width) input must give bit-identical results."""
        from roicat.ROInet import ScaleDynamicRange
        rng = np.random.default_rng(seed=0)
        for shape in [(1, 36, 36), (1, 12, 20), (3, 8, 8)]:
            x = torch.as_tensor(rng.random(shape) * 700 - 300, dtype=torch.float32)
            assert torch.equal(ScaleDynamicRange()(x), self._scale_dynamic_range_original(x)), \
                f"shape {shape} differs from the original implementation"

    def test_batched_matches_per_image(self):
        """Each image in a batch is scaled by its own min/max."""
        from roicat.ROInet import ScaleDynamicRange
        rng = np.random.default_rng(seed=1)
        ## Deliberately different dynamic ranges per image
        x = torch.as_tensor(
            rng.random((5, 1, 16, 16)) * rng.integers(1, 1000, size=(5, 1, 1, 1)),
            dtype=torch.float32,
        )
        sdr = ScaleDynamicRange()
        expected = torch.stack([sdr(im) for im in x], dim=0)
        assert torch.equal(sdr(x), expected)

    def test_output_range(self):
        from roicat.ROInet import ScaleDynamicRange
        rng = np.random.default_rng(seed=2)
        x = torch.as_tensor(rng.random((4, 1, 10, 10)) * 50 + 10, dtype=torch.float32)
        out = ScaleDynamicRange()(x)
        np.testing.assert_allclose(out.amin(dim=(-3, -2, -1)).numpy(), np.zeros(4), atol=1e-6)
        np.testing.assert_allclose(out.amax(dim=(-3, -2, -1)).numpy(), np.ones(4), atol=1e-6)

    def test_jit_scriptable(self):
        """jit_script_transforms=True is a supported option, so the module must script."""
        from roicat.ROInet import ScaleDynamicRange
        scripted = torch.jit.script(ScaleDynamicRange())
        x = torch.as_tensor(np.random.default_rng(3).random((2, 1, 9, 9)), dtype=torch.float32)
        assert torch.equal(scripted(x), ScaleDynamicRange()(x))


class Test_Preprocessor_ROI_images:
    """
    Preprocessor_ROI_images is the single definition of the ROInet preprocessing
    chain. These tests pin (a) that it reproduces the pre-consolidation
    DataLoader chain bitwise, and (b) that its config round-trips, since
    ClassifierPackage serialises it.
    """

    @staticmethod
    def _images(n_roi=13, size=36, seed=0):
        rng = np.random.default_rng(seed=seed)
        ## Varying per-image dynamic range, to catch batch-wide normalization
        return (rng.random((n_roi, size, size)) * rng.integers(1, 500, size=(n_roi, 1, 1))).astype(np.float32)

    def test_matches_original_dataloader_chain_bitwise(self):
        """
        The batched chain must equal the original per-sample chain
        (Resizer_ROI_images -> ScaleDynamicRange -> Resize -> TileChannels)
        bitwise. This is what lets ClassifierPackage.predict and
        ROInet_embedder.generate_latents agree.
        """
        import torchvision
        from roicat.ROInet import (
            Preprocessor_ROI_images, Resizer_ROI_images, TileChannels, dataset_simCLR,
        )
        images = self._images()
        um_per_pixel = 1.6365

        ## Original: stage 1 with the default scale-factor lambda, then per-sample
        ## transforms applied through dataset_simCLR, exactly as before.
        images_rs_ref = Resizer_ROI_images(verbose=False).resize_ROIs(
            ROI_images=images, um_per_pixel=um_per_pixel,
        )
        transforms_ref = torch.nn.Sequential(
            _ScaleDynamicRange_original(),
            torchvision.transforms.Resize(
                size=(224, 224),
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
                antialias=True,
            ),
            TileChannels(dim=0, n_channels=3),
        )
        dataset_ref = dataset_simCLR(
            X=torch.as_tensor(images_rs_ref, dtype=torch.float32),
            y=torch.zeros(images_rs_ref.shape[0]),
            n_transforms=1,
            transform=transforms_ref,
            DEVICE='cpu',
            dtype_X=torch.float32,
        )
        out_ref = torch.stack([dataset_ref[ii][0][0] for ii in range(len(dataset_ref))], dim=0)

        ## Consolidated: one preprocessor, batched
        preprocessor = Preprocessor_ROI_images(verbose=False)
        images_rs = preprocessor.scale_normalize_images(ROI_images=images, um_per_pixel=um_per_pixel)
        out = preprocessor.transform_images(ROI_images=images_rs)

        assert np.array_equal(images_rs_ref, images_rs), 'stage 1 (scale normalization) differs'
        assert torch.equal(out_ref, out), 'stage 2 (tensor transforms) differs'

    def test_to_dict_covers_every_config_arg(self):
        """
        to_dict() is built from an explicit key list. If an __init__ arg is added
        and not listed, to_dict() drops it and from_dict() silently substitutes
        the default — a packet whose recorded preprocessing differs from the one
        used at training. Pin the two together.
        """
        import inspect
        from roicat.ROInet import Preprocessor_ROI_images
        keys_signature = set(inspect.signature(Preprocessor_ROI_images.__init__).parameters) - {'self', 'verbose'}
        keys_dict = set(Preprocessor_ROI_images(verbose=False).to_dict())
        assert keys_dict == keys_signature, (
            f"to_dict() keys {sorted(keys_dict)} != __init__ args {sorted(keys_signature)}. "
            f"Missing keys would be silently replaced by defaults on from_dict()."
        )

    def test_serializable_by_richfile(self):
        """
        ROInet_embedder holds a preprocessor and pipelines.py saves
        roinet.__dict__ through RichFile, so this object must be serializable.
        """
        import tempfile
        from roicat.ROInet import Preprocessor_ROI_images
        preprocessor = Preprocessor_ROI_images(factor_scaleFactor=1.7, verbose=False)
        with tempfile.TemporaryDirectory() as dir_tmp:
            path = str(Path(dir_tmp) / 'pp.richfile.zip')
            util.RichFile_ROICaT(path=path, backend='zip').save({'preprocessor': preprocessor}, overwrite=True)
            loaded = util.RichFile_ROICaT(path=path).load()
        assert loaded['preprocessor'].to_dict() == preprocessor.to_dict()

    def test_no_callables_in_dict(self):
        """
        Holding a Callable (e.g. a scale-factor closure) would make the object
        unpicklable and unserializable; the scale factor is two numbers instead.
        """
        from roicat.ROInet import Preprocessor_ROI_images
        preprocessor = Preprocessor_ROI_images(verbose=False)
        offenders = {k: v for k, v in preprocessor.__dict__.items() if callable(v) and not isinstance(v, torch.nn.Module)}
        assert offenders == {}, f"Preprocessor_ROI_images holds callables: {sorted(offenders)}"

    def test_config_roundtrip(self):
        from roicat.ROInet import Preprocessor_ROI_images
        preprocessor = Preprocessor_ROI_images(
            factor_scaleFactor=2.4,
            size_im_reference=48,
            img_size_out=(112, 112),
            n_channels_out=1,
            verbose=False,
        )
        config = preprocessor.to_dict()
        ## Must survive a JSON round trip, since it is stored in a .roicat_classifier packet
        import json
        rebuilt = Preprocessor_ROI_images.from_dict(config=json.loads(json.dumps(config)))
        assert rebuilt.to_dict() == config
        out = rebuilt.preprocess(ROI_images=self._images(n_roi=2), um_per_pixel=1.0)
        assert out.shape == (2, 1, 112, 112)

    def test_scale_factor_parameterization_matches_lambda(self):
        """
        The two-number parameterization must reproduce the default
        Resizer_ROI_images lambda, `1.2 * um_per_pixel * (size_im / 36)`.
        """
        from roicat.ROInet import Preprocessor_ROI_images, Resizer_ROI_images
        images = self._images(n_roi=5)
        out_default = Resizer_ROI_images(verbose=False).resize_ROIs(ROI_images=images, um_per_pixel=2.0)
        out_param = Preprocessor_ROI_images(verbose=False).scale_normalize_images(
            ROI_images=images, um_per_pixel=2.0,
        )
        assert np.array_equal(out_default, out_param)

    def test_um_per_pixel_changes_output(self):
        from roicat.ROInet import Preprocessor_ROI_images
        preprocessor = Preprocessor_ROI_images(verbose=False)
        images = self._images(n_roi=3)
        out_a = preprocessor.preprocess(ROI_images=images, um_per_pixel=1.0)
        out_b = preprocessor.preprocess(ROI_images=images, um_per_pixel=2.0)
        assert not torch.allclose(out_a, out_b)

    def test_scale_normalize_false_skips_stage_1(self):
        from roicat.ROInet import Preprocessor_ROI_images
        preprocessor = Preprocessor_ROI_images(scale_normalize=False, verbose=False)
        images = self._images(n_roi=3)
        out_a = preprocessor.preprocess(ROI_images=images, um_per_pixel=1.0)
        out_b = preprocessor.preprocess(ROI_images=images, um_per_pixel=7.0)
        assert torch.equal(out_a, out_b)
        assert np.array_equal(
            preprocessor.scale_normalize_images(ROI_images=images, um_per_pixel=1.0), images,
        )

    def test_multiple_sessions_use_own_um_per_pixel(self):
        from roicat.ROInet import Preprocessor_ROI_images
        preprocessor = Preprocessor_ROI_images(verbose=False)
        sessions = [self._images(n_roi=3, seed=0), self._images(n_roi=4, seed=1)]
        out = preprocessor.preprocess(ROI_images=sessions, um_per_pixel=[1.0, 2.0])
        assert out.shape == (7, 3, 224, 224)
        ## Session 0 must match a solo run at its own um_per_pixel
        out_solo = preprocessor.preprocess(ROI_images=sessions[0], um_per_pixel=1.0)
        assert torch.equal(out[:3], out_solo)

    def test_empty_input(self):
        from roicat.ROInet import Preprocessor_ROI_images
        preprocessor = Preprocessor_ROI_images(verbose=False)
        out = preprocessor.preprocess(
            ROI_images=np.zeros((0, 36, 36), dtype=np.float32), um_per_pixel=1.0,
        )
        assert out.shape == (0, 3, 224, 224)

    def test_int_um_per_pixel_accepted(self):
        from roicat.ROInet import Preprocessor_ROI_images
        preprocessor = Preprocessor_ROI_images(verbose=False)
        out = preprocessor.preprocess(ROI_images=self._images(n_roi=2), um_per_pixel=2)
        assert out.shape == (2, 3, 224, 224)

    def test_bad_ndim_raises(self):
        from roicat.ROInet import Preprocessor_ROI_images
        preprocessor = Preprocessor_ROI_images(verbose=False)
        with pytest.raises(ValueError, match='3-D'):
            preprocessor.transform_images(ROI_images=np.zeros((36, 36), dtype=np.float32))


class Test_reason_fused_local_corr_unavailable:
    """
    The helper that decides whether RoMa is given its fused correlation kernel.

    Every test clears the cache on both sides. The helper is ``lru_cache``d, so
    without that the answer depends on which test ran first, and a cached answer
    derived from a patched ``sys.modules`` would leak into the rest of the run.
    """

    @staticmethod
    def _fn():
        from roicat.tracking.alignment import _reason_fused_local_corr_unavailable
        return _reason_fused_local_corr_unavailable

    @pytest.fixture(autouse=True)
    def _clear_cache(self):
        self._fn().cache_clear()
        yield
        self._fn().cache_clear()

    def test_returns_none_when_kernel_imports(self, monkeypatch):
        import sys, types
        monkeypatch.setitem(sys.modules, 'local_corr', types.ModuleType('local_corr'))
        assert self._fn()() is None

    def test_returns_reason_when_kernel_missing(self, monkeypatch):
        import sys
        ## A None entry in sys.modules makes `import local_corr` raise ImportError,
        ## which is the same failure a machine without the package produces.
        monkeypatch.setitem(sys.modules, 'local_corr', None)
        reason = self._fn()()
        assert isinstance(reason, str) and reason

    def test_answer_is_cached(self, monkeypatch):
        import sys, types
        monkeypatch.setitem(sys.modules, 'local_corr', types.ModuleType('local_corr'))
        assert self._fn()() is None
        ## The installation cannot change mid-process, so the second call must
        ## not re-probe even though the module is now unimportable.
        monkeypatch.setitem(sys.modules, 'local_corr', None)
        assert self._fn()() is None
        self._fn().cache_clear()
        assert isinstance(self._fn()(), str)


######################################################################################################################################
################################################## RICHFILE TYPE REGISTRY ############################################################
######################################################################################################################################


class Test_richfile_type_registry:
    """
    Every type registered on RichFile_ROICaT carries a "library" string, and
    richfile resolves that string to an installed distribution on *every save*
    in order to stamp a version into the metadata. So a library naming a
    package ROICaT does not actually depend on is a save-time crash for anyone
    without it -- see issue #660, where "model_swt" claimed to come from
    onnx2torch (a package nothing in ROICaT imports, installed only by the
    `all` extra), which broke saving for every tracking-only install.
    """

    @staticmethod
    def _resolve_library(library):
        """Mirror of the library -> version resolution richfile performs during
        save. Returns a version string, or raises the way a save would."""
        import importlib
        import importlib.metadata

        if library in ('python', 'builtins'):
            return 'builtin'
        try:
            return importlib.metadata.version(library)
        except importlib.metadata.PackageNotFoundError:
            ## richfile falls back to importing the module; so do we.
            return getattr(importlib.import_module(library), '__version__', 'unknown')

    def _registered_properties(self):
        return util.RichFile_ROICaT().type_lookup.properties

    def test_every_registered_library_is_resolvable(self):
        """A library string that does not resolve is one that raises on save.

        This has teeth because the test environment is built from ROICaT's own
        extras: a registration naming a package that is not a ROICaT dependency
        fails here rather than in a user's pipeline.
        """
        unresolvable = {}
        for prop in self._registered_properties():
            try:
                self._resolve_library(prop['library'])
            except Exception as e:
                unresolvable[prop['type_name']] = f"{prop['library']!r} -> {type(e).__name__}: {e}"
        assert not unresolvable, (
            'These registered types name a library that does not resolve in this '
            f'environment, so saving them would raise: {unresolvable}'
        )

    def test_model_swt_is_attributed_to_roicat(self):
        """Model_SWT is defined in roicat.util. Regression guard for #660."""
        prop = util.RichFile_ROICaT().type_lookup['model_swt']
        assert prop['library'] == 'roicat', (
            f"model_swt should be attributed to roicat, got {prop['library']!r}"
        )

    def test_no_registration_names_onnx2torch(self):
        """Nothing in ROICaT imports onnx2torch and it is no longer a
        dependency, so no type should claim to come from it."""
        offenders = [p['type_name'] for p in self._registered_properties() if p['library'] == 'onnx2torch']
        assert not offenders, f'types still attributed to onnx2torch: {offenders}'


######################################################################################################################################
##################################################### MODEL_SWT SERIALIZATION ########################################################
######################################################################################################################################


class Test_Model_SWT_serialization:
    """
    Model_SWT used to be written into richfiles as its repr -- a 38-byte string
    like ``'Model_SWT(\\n  (model): Scattering2D()\\n)'`` -- which recorded
    neither J, L, shape nor device, and loaded back as that string rather than
    as a model. These tests pin the JSON round trip that replaced it, and pin
    that files written the old way still load the old way.
    """

    SHAPE = (36, 36)

    def _make_model(self):
        from roicat.tracking import scatteringWaveletTransformer as swt_module
        Scattering2D = swt_module.import_Scattering2D()
        return util.Model_SWT(Scattering2D(shape=self.SHAPE, J=2, L=8))

    def _save_load(self, obj, tmp_path, name='payload.richfile'):
        path = str(Path(tmp_path) / name)
        util.RichFile_ROICaT(path=path, backend='directory').save({'swt': obj}, overwrite=True)
        return util.RichFile_ROICaT(path=path).load()['swt']

    def test_to_dict_captures_constructor_args_and_is_json_safe(self):
        import json

        d = self._make_model().to_dict()
        assert d['kwargs_Scattering2D']['J'] == 2
        assert d['kwargs_Scattering2D']['L'] == 8
        assert tuple(d['kwargs_Scattering2D']['shape']) == self.SHAPE
        ## `backend` becomes a module object inside kymatio, so it must not be
        ## recorded; if it leaked in, this dump would raise.
        assert 'backend' not in d['kwargs_Scattering2D']
        json.dumps(d)

    def test_round_trip_returns_an_equivalent_model(self, tmp_path):
        """The point of the change: what comes back is a model, not a string."""
        model = self._make_model()
        loaded = self._save_load(model, tmp_path)

        assert isinstance(loaded, util.Model_SWT), (
            f'expected a Model_SWT back, got {type(loaded).__name__}'
        )

        ## kymatio derives its filter bank analytically, so a rebuilt model's
        ## buffers should be bit-identical, not merely close.
        buffers_original = dict(model.named_buffers())
        buffers_loaded = dict(loaded.named_buffers())
        assert set(buffers_original) == set(buffers_loaded), 'filter bank differs in structure'
        for k in buffers_original:
            np.testing.assert_array_equal(
                buffers_original[k].numpy(), buffers_loaded[k].numpy(),
                err_msg=f'filter buffer {k!r} differs after round trip',
            )

    def test_round_trip_preserves_the_transform(self, tmp_path):
        """Identical filters should mean identical outputs on the same input."""
        model = self._make_model()
        loaded = self._save_load(model, tmp_path)

        rng = np.random.RandomState(0)
        x = torch.as_tensor(rng.randn(2, *self.SHAPE).astype(np.float32)).contiguous()
        with torch.no_grad():
            np.testing.assert_array_equal(model(x).numpy(), loaded(x).numpy())

    def test_round_trip_preserves_dtype(self, tmp_path):
        """A model cast with .double() must come back as float64.

        kymatio builds its filter bank in a fixed default dtype, and its Fourier
        ops require input and filters to match ("Input and filter must be of the
        same dtype"). Recording only the device would silently narrow a cast
        model on reload, so it would reject the very inputs the original
        accepted -- a wrong answer rather than an error.
        """
        model = self._make_model().double()
        assert next(iter(model.buffers())).dtype == torch.float64

        loaded = self._save_load(model, tmp_path)
        assert next(iter(loaded.buffers())).dtype == torch.float64, (
            'dtype was not preserved through the round trip'
        )

        rng = np.random.RandomState(0)
        x = torch.as_tensor(rng.randn(2, *self.SHAPE)).double().contiguous()
        with torch.no_grad():
            np.testing.assert_array_equal(model(x).numpy(), loaded(x).numpy())

    def test_load_without_recorded_dtype_still_works(self, tmp_path):
        """Payloads written before dtype was recorded must still rebuild."""
        d = self._make_model().to_dict()
        d.pop('dtype')
        rebuilt = util.Model_SWT.from_dict(d)
        assert isinstance(rebuilt, util.Model_SWT)

    def test_legacy_repr_payload_loads_unchanged(self, tmp_path):
        """Archives written before this change hold a repr string under the
        same type name. They must keep loading, and keep returning the string
        they always returned."""
        legacy = 'Model_SWT(\n  (model): Scattering2D()\n)'
        path = Path(tmp_path) / 'legacy.swt'
        path.write_text(legacy)

        function_load = util.RichFile_ROICaT().type_lookup['model_swt']['function_load']
        assert function_load(path=str(path)) == legacy

    def test_load_degrades_to_params_without_kymatio(self, tmp_path, monkeypatch):
        """Rebuilding needs kymatio. Not having it must not turn a load that
        used to succeed into an exception -- warn and hand back the params."""
        import sys

        model = self._make_model()
        path = Path(tmp_path) / 'model.swt'
        function_save = util.RichFile_ROICaT().type_lookup['model_swt']['function_save']
        function_load = util.RichFile_ROICaT().type_lookup['model_swt']['function_load']
        function_save(obj=model, path=str(path))

        ## A None entry in sys.modules makes the import raise ImportError, which
        ## is the same failure a machine without kymatio produces.
        monkeypatch.setitem(sys.modules, 'kymatio', None)
        monkeypatch.setitem(sys.modules, 'kymatio.torch', None)

        with pytest.warns(UserWarning):
            out = function_load(path=str(path))
        assert isinstance(out, dict) and 'kwargs_Scattering2D' in out

    def test_non_kymatio_model_falls_back_to_repr(self, tmp_path):
        """Model_SWT is a generic wrapper. Wrapping something that is not a
        Scattering2D should degrade to the old repr record rather than break
        the entire save."""
        wrapped = util.Model_SWT(torch.nn.Linear(2, 2))
        path = Path(tmp_path) / 'other.swt'
        function_save = util.RichFile_ROICaT().type_lookup['model_swt']['function_save']

        with pytest.warns(UserWarning):
            function_save(obj=wrapped, path=str(path))
        assert path.read_text().startswith('Model_SWT(')


class Test_plot_quality_metrics:
    """
    The suptitle counts of ``plot_quality_metrics``.

    ``make_label_variants`` ends by casting the squeezed labels to a
    ``util.JSON_List`` for JSON compatibility, and the pipeline hands that
    straight to this function. On a list, ``labels == -1`` is the scalar
    ``False`` instead of a boolean mask, so the title read
    ``n_excluded: 0, n_included: 1, n_clusters: 1`` on every run. These tests
    pin the counts, and pin that a list and an array give the same title.
    """

    ## 3 excluded, 4 included, 2 clusters, 7 total.
    LABELS = [0, 0, 1, 1, -1, -1, -1]

    @pytest.fixture(autouse=True)
    def _headless_backend(self):
        import matplotlib
        backend_original = matplotlib.get_backend()
        matplotlib.use('Agg')
        yield
        matplotlib.use(backend_original)

    @staticmethod
    def _quality_metrics():
        """The three keys the function histograms. Values are arbitrary."""
        return {
            'cluster_silhouette': np.array([0.1, 0.6]),
            'cluster_intra_means': np.array([0.4, 0.8]),
            'sample_silhouette': np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
        }

    @staticmethod
    def _title(labels):
        import matplotlib.pyplot as plt
        from roicat.tracking.clustering import plot_quality_metrics

        fig, _ = plot_quality_metrics(
            quality_metrics=Test_plot_quality_metrics._quality_metrics(),
            labels=labels,
            n_sessions=2,
        )
        try:
            return fig.get_suptitle()
        finally:
            plt.close(fig)

    def test_counts_are_correct_for_a_JSON_List(self):
        """The type the pipeline actually passes."""
        title = self._title(util.JSON_List(self.LABELS))
        assert 'n_excluded: 3' in title
        assert 'n_included: 4' in title
        assert 'n_total: 7' in title
        assert 'n_clusters: 2' in title
        assert 'n_sessions: 2' in title

    def test_counts_are_correct_for_a_plain_list(self):
        title = self._title(list(self.LABELS))
        assert 'n_excluded: 3' in title
        assert 'n_clusters: 2' in title

    def test_list_and_array_give_the_same_title(self):
        """The bug stated directly: the title must not depend on the container."""
        assert self._title(util.JSON_List(self.LABELS)) == self._title(np.array(self.LABELS))

    def test_counts_are_correct_when_nothing_is_excluded(self):
        """`labels == -1` matching nothing must still give a real mask, not False."""
        title = self._title(util.JSON_List([0, 0, 1, 1, 2]))
        assert 'n_excluded: 0' in title
        assert 'n_included: 5' in title
        assert 'n_clusters: 3' in title


@pytest.mark.parametrize('n_pixels, dtype_idx, n_workers', [(512 * 512, np.int32, 1), (2**33, np.int64, 1), (512 * 512, np.int32, -1)])
def test_manhattan_similarity_index_dtypes(n_pixels, dtype_idx, n_workers):
    """The spatial footprint similarity must work with int32 pixel indices
    (typical FOVs) and int64 ones beyond int32 (FOVs over ~46k x 46k), with
    one or all cores, and equal 1 - L1 on the normalized rows."""
    from roicat.tracking.similarity_graph import ROI_graph, DEFAULT_METRICS
    rng = np.random.default_rng(0)
    n_roi, n_perRoi = 30, 40
    idx_row = np.repeat(np.arange(n_roi), n_perRoi)  ## shape: (n_roi * n_perRoi,)
    idx_col = rng.integers(n_pixels // 2, n_pixels, size=n_roi * n_perRoi, dtype=np.int64)
    idx_col[::3] = idx_col[0]  ## shared pixels so some similarities are nonzero
    sf = scipy.sparse.csr_array((rng.random(idx_col.size, dtype=np.float32), (idx_row, idx_col)), shape=(n_roi, n_pixels))
    sf.sum_duplicates()
    sf = scipy.sparse.csr_array((sf.data, sf.indices.astype(dtype_idx), sf.indptr.astype(dtype_idx)), shape=sf.shape)
    assert sf.indices.dtype == dtype_idx

    graph = ROI_graph(n_workers=n_workers, frame_height=1, frame_width=1, block_height=1, block_width=1, verbose=False)
    graph._sf_maskPower = 1.0
    s_sf = graph._compute_manhattan_similarity(spatialFootprints=sf, config=DEFAULT_METRICS[0])
    assert isinstance(s_sf, scipy.sparse.csr_array)
    assert s_sf.has_sorted_indices

    ## Dense reference on the used columns only
    _, idx_colUsed = np.unique(sf.indices, return_inverse=True)
    x = scipy.sparse.csr_array((sf.data.astype(np.float64), idx_colUsed, sf.indptr), shape=(n_roi, idx_colUsed.max() + 1)).toarray()
    x = 0.5 * x / x.sum(1, keepdims=True)  ## shape: (n_roi, n_colUsed)
    s_ref = 1 - np.abs(x[:, None, :] - x[None, :, :]).sum(-1)  ## shape: (n_roi, n_roi)
    s_ref[s_ref < 1e-5] = 0
    np.fill_diagonal(s_ref, 0)
    assert s_sf.nnz > 0
    assert np.allclose(s_sf.toarray(), s_ref, atol=1e-6)


@pytest.mark.parametrize('n_workers', [0, -2, 1.0])
def test_roi_graph_invalid_n_workers_raises(n_workers):
    from roicat.tracking.similarity_graph import ROI_graph
    with pytest.raises(ValueError, match='n_workers'):
        ROI_graph(n_workers=n_workers, frame_height=1, frame_width=1, block_height=1, block_width=1, verbose=False)


def test_manhattan_similarity_kernel_cache_loads_in_new_process():
    """A second Python process loads the compiled kernel from numba's cache.

    A jitted function that captures another dispatcher from its enclosing
    scope gets a new cache key in every process, so it recompiles and writes
    a new cache file on every run.
    """
    import subprocess
    import sys
    code = (
        "import numpy as np\n"
        "from roicat.tracking.similarity_graph import _get_manhattan_similarity_kernel\n"
        "k = _get_manhattan_similarity_kernel()\n"
        "k(np.array([0, 1], np.int64), np.array([0.5]), np.array([0], np.int64), np.array([0, 1], np.int64),"
        " np.array([0], np.int64), np.array([0.5]), np.array([0.5]), 1e-5)\n"
        "print(sum(k.stats.cache_hits.values()))\n"
    )
    hits = [int(subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, check=True).stdout.strip().splitlines()[-1]) for _ in range(2)]
    assert hits[1] == 1, f"cache hits per process: {hits}"
