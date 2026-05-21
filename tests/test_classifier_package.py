"""
Round-trip tests for roicat.ClassifierPackage.

Constructs a minimal Simclr_Model-shaped embedder using a stub `model.py`
that defines `Simclr_Model` with `.model` (an nn.Module), `.embed()`, and
`.arch_kwargs`. Round-trips through save/load and exercises the contract
guarantees (byte-determinism, SHA mismatch, namespace isolation, etc.).

Runtime: well under 30 s on CPU.
"""

import io
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

import roicat
from roicat.classification.classifier import Auto_LogisticRegression
from roicat.classification.package import (
    ClassifierPackage,
    PackageIntegrityError,
    _SCHEMA_VERSION,
    _MEMBER_ORDER,
)


_LATENT_DIM = 8
_N_CLASSES = 2
_N_SAMPLES = 40
_N_ROI = 6
_IMG_H, _IMG_W = 16, 16


# ---------------------------------------------------------------------------
# Stub model.py — self-contained, defines Simclr_Model with .model + .embed
# Built as a parametrized template so we can produce TWO distinct bundled
# model.py bytes (for namespace-collision testing).
# ---------------------------------------------------------------------------

def _stub_model_py(tag: str = "A", latent_dim: int = _LATENT_DIM) -> str:
    return f'''"""Stub model.py — tag={tag}."""
import torch
import torch.nn as nn
import numpy as np
import torchvision

_LATENT_DIM = {latent_dim}
_TAG = "{tag}"


class _TinyNet(nn.Module):
    def __init__(self, latent_dim=_LATENT_DIM):
        super().__init__()
        self.fc = nn.Linear(3 * 224 * 224, latent_dim, bias=False)

    def forward(self, x):
        return self.fc(x.flatten(1))


class Simclr_Model:
    def __init__(self, latent_dim=_LATENT_DIM, **extra):
        self.model = _TinyNet(latent_dim=latent_dim)
        self.arch_kwargs = {{"latent_dim": latent_dim, **extra}}
        self._tag = _TAG

    def embed(self, patches_np, device="cpu"):
        _eps = 1e-9
        x = torch.as_tensor(patches_np, dtype=torch.float32)
        if x.shape[0] == 0:
            x = x.reshape(0, 3, 224, 224)
        else:
            x_min = x.flatten(1).min(dim=1).values[:, None, None]
            x_max = x.flatten(1).max(dim=1).values[:, None, None]
            x = (x - x_min) / (x_max - x_min + _eps)
            resize = torchvision.transforms.Resize(
                size=(224, 224),
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
                antialias=True,
            )
            x = torch.stack([resize(img[None, ...]) for img in x], dim=0)
            x = x.expand(-1, 3, -1, -1)
        self.model.eval()
        self.model.to(device)
        with torch.no_grad():
            return self.model(x.to(device)).cpu().numpy().astype(np.float32, copy=False)
'''


def _import_stub_module(stub_src: str, module_name: str = "_test_stub_model"):
    """Import the stub model.py from a string under a custom module name."""
    import importlib.util
    import tempfile
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)
    tmp.write(stub_src)
    tmp.close()
    spec = importlib.util.spec_from_file_location(module_name, tmp.name)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod, tmp.name


def _make_stub_embedder(tag: str = "A"):
    """Build a stub Simclr_Model from an in-memory model.py."""
    src = _stub_model_py(tag=tag)
    mod, _path = _import_stub_module(src, module_name=f"_stub_{tag}")
    return mod.Simclr_Model(latent_dim=_LATENT_DIM)


def _make_fitted_classifier() -> Auto_LogisticRegression:
    """Build an Auto_LogisticRegression with a manually-fitted sklearn model."""
    import sklearn.linear_model
    rng = np.random.default_rng(seed=0)
    X = rng.standard_normal((_N_SAMPLES, _LATENT_DIM)).astype(np.float32)
    y = np.array([0] * (_N_SAMPLES // 2) + [1] * (_N_SAMPLES // 2), dtype=np.int64)

    clf = Auto_LogisticRegression(
        X=X, y=y,
        params_LogisticRegression={"C": 1.0, "penalty": "l2", "solver": "lbfgs", "max_iter": 200},
        label_names=["class_0", "class_1"],
        verbose=False,
    )
    sklearn_lr = sklearn.linear_model.LogisticRegression(
        C=1.0, penalty="l2", solver="lbfgs", max_iter=200,
        class_weight=clf.class_weight,
    )
    sklearn_lr.fit(X, y)
    clf.model_best = sklearn_lr
    clf.best_params = {"C": 1.0}
    return clf


# ---------------------------------------------------------------------------
# Test fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def pkg_factory(tmp_path):
    """Factory yielding (pkg, filepath) for fresh stubs per call."""
    def _factory(tag: str = "A", label_names=None, preprocessing=None):
        embedder = _make_stub_embedder(tag=tag)
        clf = _make_fitted_classifier()
        ln = label_names if label_names is not None else ["class_0", "class_1"]
        pp = preprocessing if preprocessing is not None else {
            "image_out_size": [_IMG_H, _IMG_W],
            "um_per_pixel": 1.0,
            "normalization": "per_roi_max",
        }
        pkg = ClassifierPackage(classifier=clf, embedder=embedder, label_names=ln, preprocessing=pp)
        return pkg, str(tmp_path / f"test_{tag}.roicat")
    return _factory


# ---------------------------------------------------------------------------
# Tests — save / load / round-trip
# ---------------------------------------------------------------------------

class TestRoundTrip:
    def test_save_creates_file(self, pkg_factory):
        pkg, filepath = pkg_factory()
        pkg.save(filepath)
        assert Path(filepath).exists()
        assert Path(filepath).stat().st_size > 0

    def test_zip_contains_expected_members(self, pkg_factory):
        pkg, filepath = pkg_factory()
        pkg.save(filepath)
        with zipfile.ZipFile(filepath) as zf:
            names = zf.namelist()
        for expected in _MEMBER_ORDER:
            assert expected in names

    def test_member_order_matches_contract(self, pkg_factory):
        pkg, filepath = pkg_factory()
        pkg.save(filepath)
        with zipfile.ZipFile(filepath) as zf:
            names = zf.namelist()
        assert names == list(_MEMBER_ORDER), f"Member order {names} != contract {list(_MEMBER_ORDER)}"

    def test_metadata_fields(self, pkg_factory):
        pkg, filepath = pkg_factory()
        pkg.save(filepath)
        with zipfile.ZipFile(filepath) as zf:
            metadata = json.loads(zf.read("metadata.json"))
        assert metadata["schema_version"] == _SCHEMA_VERSION
        assert metadata["roicat_version"] == roicat.__version__
        assert metadata["label_names"] == ["class_0", "class_1"]
        assert "embedder_model_py_sha256" in metadata
        assert "embedder_weights_sha256" in metadata
        assert "classifier_sha256" in metadata
        assert "embedder_forward_pass_version" in metadata
        assert "created_at_utc" in metadata
        ## ISO-8601 UTC ends with 'Z'
        assert metadata["created_at_utc"].endswith("Z")

    def test_load_roundtrip(self, pkg_factory):
        pkg, filepath = pkg_factory()
        pkg.save(filepath)
        loaded = ClassifierPackage.load(filepath)
        assert loaded.label_names == ["class_0", "class_1"]
        assert tuple(loaded.preprocessing["image_out_size"]) == (_IMG_H, _IMG_W)

    def test_predict_output_shapes(self, pkg_factory):
        pkg, filepath = pkg_factory()
        pkg.save(filepath)
        loaded = ClassifierPackage.load(filepath)
        rng = np.random.default_rng(seed=1)
        roi_images = rng.standard_normal((_N_ROI, _IMG_H, _IMG_W)).astype(np.float32)
        label_ids, probs = loaded.predict(roi_images=roi_images)
        assert label_ids.shape == (_N_ROI,)
        assert probs.shape == (_N_ROI, _N_CLASSES)
        assert probs.dtype == np.float32
        np.testing.assert_allclose(probs.sum(axis=1), np.ones(_N_ROI), atol=1e-5)

    def test_save_raises_if_file_exists(self, pkg_factory):
        pkg, filepath = pkg_factory()
        pkg.save(filepath)
        with pytest.raises(FileExistsError):
            pkg.save(filepath)

    def test_save_overwrite_succeeds(self, pkg_factory):
        pkg, filepath = pkg_factory()
        pkg.save(filepath)
        pkg.save(filepath, overwrite=True)
        assert Path(filepath).exists()

    def test_save_creates_parent_dirs(self, tmp_path, pkg_factory):
        pkg, _ = pkg_factory()
        nested = tmp_path / "deep" / "nested" / "pkt.roicat"
        pkg.save(str(nested))
        assert nested.exists()


# ---------------------------------------------------------------------------
# Tests — integrity & error model
# ---------------------------------------------------------------------------

class TestIntegrity:
    def test_load_raises_on_tampered_sha256(self, pkg_factory, tmp_path):
        pkg, filepath = pkg_factory()
        pkg.save(filepath)
        with zipfile.ZipFile(filepath, "r") as zf_in:
            members = {n: zf_in.read(n) for n in zf_in.namelist()}
        metadata = json.loads(members["metadata.json"])
        metadata["classifier_sha256"] = "deadbeef" * 8
        members["metadata.json"] = json.dumps(metadata).encode()
        tampered = str(tmp_path / "tampered.roicat")
        with zipfile.ZipFile(tampered, "w") as zf_out:
            for name, data in members.items():
                zf_out.writestr(name, data)
        with pytest.raises(PackageIntegrityError, match="SHA-256 mismatch"):
            ClassifierPackage.load(tampered)

    def test_load_raises_on_schema_version_mismatch(self, pkg_factory, tmp_path):
        pkg, filepath = pkg_factory()
        pkg.save(filepath)
        with zipfile.ZipFile(filepath, "r") as zf_in:
            members = {n: zf_in.read(n) for n in zf_in.namelist()}
        metadata = json.loads(members["metadata.json"])
        metadata["schema_version"] = 999
        members["metadata.json"] = json.dumps(metadata).encode()
        bogus = str(tmp_path / "bogus.roicat")
        with zipfile.ZipFile(bogus, "w") as zf_out:
            for name, data in members.items():
                zf_out.writestr(name, data)
        with pytest.raises(PackageIntegrityError, match="schema_version"):
            ClassifierPackage.load(bogus)

    def test_load_raises_on_missing_member(self, pkg_factory, tmp_path):
        pkg, filepath = pkg_factory()
        pkg.save(filepath)
        with zipfile.ZipFile(filepath, "r") as zf_in:
            members = {n: zf_in.read(n) for n in zf_in.namelist() if n != "preprocessing.json"}
        stripped = str(tmp_path / "stripped.roicat")
        with zipfile.ZipFile(stripped, "w") as zf_out:
            for name, data in members.items():
                zf_out.writestr(name, data)
        with pytest.raises(KeyError, match="preprocessing.json"):
            ClassifierPackage.load(stripped)

    def test_load_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ClassifierPackage.load(str(tmp_path / "does_not_exist.roicat"))


# ---------------------------------------------------------------------------
# Tests — byte-determinism
# ---------------------------------------------------------------------------

class TestByteDeterminism:
    def test_two_saves_byte_identical(self, pkg_factory, tmp_path):
        """
        Same inputs produce byte-identical packets for the deterministic members.
        metadata.json contains a timestamp; classifier.onnx is generated by skl2onnx,
        which embeds a wall-clock timestamp in the protobuf 'doc_string'. We assert
        determinism on preprocessing/model.py/params/weights, and assert ZIP-level
        determinism (zeroed mtimes, ZIP_DEFLATED) on every member.
        """
        pkg, _ = pkg_factory(tag="DETA")
        p1 = str(tmp_path / "a.roicat")
        p2 = str(tmp_path / "b.roicat")
        pkg.save(p1)
        pkg.save(p2)
        deterministic = {
            "preprocessing.json",
            "embedder/model.py",
            "embedder/params.json",
            "embedder/weights.pt",
        }
        with zipfile.ZipFile(p1) as zf1, zipfile.ZipFile(p2) as zf2:
            for member in _MEMBER_ORDER:
                zi1 = zf1.getinfo(member)
                assert zi1.date_time == (1980, 1, 1, 0, 0, 0), f"{member}: non-zero mtime"
                assert zi1.compress_type == zipfile.ZIP_DEFLATED, f"{member}: not DEFLATE"
                if member in deterministic:
                    assert zf1.read(member) == zf2.read(member), f"{member} bytes differ"

    def test_external_attr_is_0644(self, pkg_factory, tmp_path):
        pkg, filepath = pkg_factory()
        pkg.save(filepath)
        with zipfile.ZipFile(filepath) as zf:
            for info in zf.infolist():
                assert info.external_attr == (0o644 << 16), f"{info.filename}: bad external_attr"


# ---------------------------------------------------------------------------
# Tests — validation
# ---------------------------------------------------------------------------

class TestInitValidation:
    def test_label_names_length_mismatch(self):
        clf = _make_fitted_classifier()
        emb = _make_stub_embedder()
        with pytest.raises(ValueError, match="label_names"):
            ClassifierPackage(
                classifier=clf, embedder=emb,
                label_names=["a", "b", "c"],
                preprocessing={"image_out_size": [16, 16], "um_per_pixel": 1.0, "normalization": "per_roi_max"},
            )

    def test_label_names_non_str(self):
        clf = _make_fitted_classifier()
        emb = _make_stub_embedder()
        with pytest.raises(TypeError, match="str"):
            ClassifierPackage(
                classifier=clf, embedder=emb,
                label_names=[0, 1],
                preprocessing={"image_out_size": [16, 16], "um_per_pixel": 1.0, "normalization": "per_roi_max"},
            )

    def test_preprocessing_missing_key(self):
        clf = _make_fitted_classifier()
        emb = _make_stub_embedder()
        with pytest.raises(KeyError, match="um_per_pixel"):
            ClassifierPackage(
                classifier=clf, embedder=emb,
                label_names=["class_0", "class_1"],
                preprocessing={"image_out_size": [16, 16], "normalization": "per_roi_max"},
            )

    def test_classifier_label_names_mismatch(self):
        clf = _make_fitted_classifier()  # has label_names = ["class_0","class_1"]
        emb = _make_stub_embedder()
        with pytest.raises(ValueError, match="label_names"):
            ClassifierPackage(
                classifier=clf, embedder=emb,
                label_names=["a", "b"],
                preprocessing={"image_out_size": [16, 16], "um_per_pixel": 1.0, "normalization": "per_roi_max"},
            )


# ---------------------------------------------------------------------------
# Tests — predict edge cases
# ---------------------------------------------------------------------------

class TestPredict:
    def test_predict_empty(self, pkg_factory):
        pkg, filepath = pkg_factory()
        pkg.save(filepath)
        loaded = ClassifierPackage.load(filepath)
        empty = np.zeros((0, _IMG_H, _IMG_W), dtype=np.float32)
        labels, probs = loaded.predict(empty)
        assert labels.shape == (0,)
        assert probs.shape == (0, _N_CLASSES)
        assert labels.dtype == np.int64
        assert probs.dtype == np.float32

    def test_predict_wrong_ndim_raises(self, pkg_factory):
        pkg, filepath = pkg_factory()
        pkg.save(filepath)
        loaded = ClassifierPackage.load(filepath)
        with pytest.raises(ValueError, match="3-D"):
            loaded.predict(np.zeros((_IMG_H, _IMG_W), dtype=np.float32))

    def test_predict_wrong_shape_raises(self, pkg_factory):
        pkg, filepath = pkg_factory()
        pkg.save(filepath)
        loaded = ClassifierPackage.load(filepath)
        with pytest.raises(ValueError, match="image_out_size"):
            loaded.predict(np.zeros((3, 32, 32), dtype=np.float32))


# ---------------------------------------------------------------------------
# Tests — Auto_LogisticRegression label_names
# ---------------------------------------------------------------------------

class TestAutoLR:
    def test_label_names_stored(self):
        clf = _make_fitted_classifier()
        assert clf.label_names == ["class_0", "class_1"]

    def test_label_names_default_str_classes(self):
        """Contract: default label_names = [str(c) for c in self.classes]."""
        rng = np.random.default_rng(seed=42)
        X = rng.standard_normal((20, 4)).astype(np.float32)
        y = np.array([0] * 10 + [1] * 10, dtype=np.int64)
        clf = Auto_LogisticRegression(
            X=X, y=y,
            params_LogisticRegression={"C": 1.0, "solver": "lbfgs"},
            verbose=False,
        )
        assert clf.label_names == ["0", "1"]

    def test_label_names_length_mismatch_raises(self):
        rng = np.random.default_rng(seed=0)
        X = rng.standard_normal((10, 3)).astype(np.float32)
        y = np.array([0, 1] * 5, dtype=np.int64)
        with pytest.raises(ValueError, match="label_names"):
            Auto_LogisticRegression(
                X=X, y=y,
                params_LogisticRegression={"C": 1.0, "solver": "lbfgs"},
                label_names=["a", "b", "c"],
                verbose=False,
            )

    def test_label_names_non_str_raises(self):
        rng = np.random.default_rng(seed=0)
        X = rng.standard_normal((10, 3)).astype(np.float32)
        y = np.array([0, 1] * 5, dtype=np.int64)
        with pytest.raises(TypeError, match="str"):
            Auto_LogisticRegression(
                X=X, y=y,
                params_LogisticRegression={"C": 1.0, "solver": "lbfgs"},
                label_names=[0, 1],
                verbose=False,
            )


# ---------------------------------------------------------------------------
# Tests — namespace isolation across multiple packets in one process
# ---------------------------------------------------------------------------

class TestNamespaceIsolation:
    def test_two_packets_different_model_py(self, tmp_path):
        """Two packets with different model.py bytes coexist; both work independently."""
        ## Build packet A
        emb_a = _make_stub_embedder(tag="A1")
        clf = _make_fitted_classifier()
        pkg_a = ClassifierPackage(
            classifier=clf, embedder=emb_a, label_names=["class_0", "class_1"],
            preprocessing={"image_out_size": [_IMG_H, _IMG_W], "um_per_pixel": 1.0, "normalization": "per_roi_max"},
        )
        path_a = str(tmp_path / "a.roicat")
        pkg_a.save(path_a)

        ## Build packet B with a different model.py (different _TAG → different bytes/SHA)
        emb_b = _make_stub_embedder(tag="B2")
        pkg_b = ClassifierPackage(
            classifier=clf, embedder=emb_b, label_names=["class_0", "class_1"],
            preprocessing={"image_out_size": [_IMG_H, _IMG_W], "um_per_pixel": 1.0, "normalization": "per_roi_max"},
        )
        path_b = str(tmp_path / "b.roicat")
        pkg_b.save(path_b)

        ## Load both in the same process.
        loaded_a = ClassifierPackage.load(path_a)
        loaded_b = ClassifierPackage.load(path_b)

        assert loaded_a._module_name != loaded_b._module_name
        assert loaded_a._module_name in sys.modules
        assert loaded_b._module_name in sys.modules

        ## Both packets predict cleanly after the other was loaded.
        rng = np.random.default_rng(seed=7)
        roi_images = rng.standard_normal((4, _IMG_H, _IMG_W)).astype(np.float32)
        la, _ = loaded_a.predict(roi_images)
        lb, _ = loaded_b.predict(roi_images)
        assert la.shape == (4,) and lb.shape == (4,)

        ## The tags survived the separate namespaces.
        assert getattr(loaded_a.embedder, "_tag", None) == "A1"
        assert getattr(loaded_b.embedder, "_tag", None) == "B2"


# ---------------------------------------------------------------------------
# Tests — real model.py self-containment (the production case)
# ---------------------------------------------------------------------------

class TestRealModelPy:
    def test_real_model_py_loads_in_clean_subprocess(self, tmp_path):
        """
        Snapshot the real `roicat/model_training/model.py` to a tempdir and
        load it via importlib.util in a SUBPROCESS where `roicat` is NOT on
        sys.path / sys.modules. Verifies that the bundled model.py is genuinely
        self-contained (no `from ..` imports leak). numpy/torch/torchvision
        must remain importable.
        """
        import inspect
        from roicat.model_training import model as real_model
        src_path = inspect.getfile(real_model)
        snap = tmp_path / "model.py"
        snap.write_bytes(Path(src_path).read_bytes())

        roicat_root = str(Path(inspect.getfile(roicat)).parent.parent)

        script = f'''
import sys, importlib.abc, importlib.util

## Hard-block any `roicat` import (editable installs add it via a .pth finder).
class _BlockRoicat(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == "roicat" or name.startswith("roicat."):
            raise ImportError(f"roicat blocked by test isolation: {{name}}")
        return None
sys.meta_path.insert(0, _BlockRoicat())

## Load the bundled model.py via spec_from_file_location.
spec = importlib.util.spec_from_file_location("standalone_model", {str(snap)!r})
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

## Confirm roicat is NOT importable.
try:
    import roicat  # type: ignore
    raise AssertionError("roicat should not be importable here")
except ImportError:
    pass

assert hasattr(m, "Simclr_Model"), "Simclr_Model missing"
assert hasattr(m, "make_model"), "make_model missing"
assert hasattr(m, "ModelTackOn"), "ModelTackOn missing"

## Also sanity-check that a Simclr_Model can be constructed (validates that
## inlined helpers like get_nums_from_string resolve).
sm = m.Simclr_Model(
    torchvision_model="convnext_tiny",
    head_pool_method="AdaptiveAvgPool2d",
    head_pool_method_kwargs={{"output_size": 1}},
    pre_head_fc_sizes=[256],
    post_head_fc_sizes=[128],
    head_nonlinearity="GELU",
    head_nonlinearity_kwargs={{}},
    block_to_unfreeze="5.6",
    n_block_toInclude=7,
    image_out_size=[3, 224, 224],
    forward_version="forward_head",
)
print("OK")
'''
        ## Run with PYTHONPATH cleared so the subprocess does not inherit roicat src dir.
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=60, env=env,
        )
        assert result.returncode == 0, f"stdout={result.stdout!r}\nstderr={result.stderr!r}"
        assert "OK" in result.stdout
