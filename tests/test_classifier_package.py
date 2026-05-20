"""
Round-trip test for roicat.ClassifierPackage.

Constructs a minimal Simclr_Model_with_PCA from a tiny convnet (NOT a full
ConvNeXt/ResNet), fits a dummy Auto_LogisticRegression on synthetic latents,
packs to a .roicat zip, loads it back, and calls predict().

Runtime: well under 30 s on CPU.
"""

import io
import json
import os
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn

import roicat
from roicat.classification.classifier import Auto_LogisticRegression, ONNX_model_sklearnLogisticRegression
from roicat.classification.package import ClassifierPackage, _SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Minimal stub embedder — a small nn.Module that implements .embed()
# and carries .arch_kwargs.  We skip instantiating a full ConvNeXt/ResNet
# to keep the test fast.  The PackageClassifier pack/unpack path is fully
# exercised because _pack_embedder uses inspect.getfile(type(embedder)),
# which points at this test file — that is deliberate, since the bundled
# model.py only needs to define make_model() or Simclr_Model_with_PCA.
# Instead we patch _pack_embedder / _unpack_embedder to use a real small net.
# ---------------------------------------------------------------------------

_LATENT_DIM = 8   # tiny latent space for speed
_N_CLASSES = 2
_N_SAMPLES = 40
_N_ROI = 6
_IMG_H, _IMG_W = 16, 16


class _TinyEmbedder(nn.Module):
    """
    Stand-in for Simclr_Model_with_PCA: a single linear layer that maps
    (N, 3, 224, 224) inputs → (N, _LATENT_DIM) outputs.
    The module is registered as an nn.Module so state_dict() works, and
    we attach arch_kwargs manually.
    """

    def __init__(self, latent_dim: int = _LATENT_DIM):
        super().__init__()
        self.fc = nn.Linear(3 * 224 * 224, latent_dim, bias=False)
        nn.init.normal_(self.fc.weight, std=0.01)
        self.arch_kwargs = {"latent_dim": latent_dim}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.flatten(1))

    def embed(self, patches_np: np.ndarray, device: str = "cpu") -> np.ndarray:
        """Mirrors Simclr_Model_with_PCA.embed signature."""
        x = torch.as_tensor(patches_np, dtype=torch.float32)  # (N, H, W)
        ## minimal preprocessing: scale + resize + tile channels (same as real embed)
        import torchvision
        _eps = 1e-9
        x_min = x.flatten(1).min(dim=1).values[:, None, None]
        x_max = x.flatten(1).max(dim=1).values[:, None, None]
        x = (x - x_min) / (x_max - x_min + _eps)
        _resize = torchvision.transforms.Resize(
            size=(224, 224),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
            antialias=True,
        )
        x = torch.stack([_resize(img[None, ...]) for img in x], dim=0)  # (N, 1, 224, 224)
        x = x.expand(-1, 3, -1, -1)  # (N, 3, 224, 224)
        self.eval()
        with torch.no_grad():
            return self(x).cpu().numpy()


def _make_tiny_embedder() -> _TinyEmbedder:
    return _TinyEmbedder(latent_dim=_LATENT_DIM)


def _make_fitted_classifier(embedder: _TinyEmbedder) -> Auto_LogisticRegression:
    """
    Build an Auto_LogisticRegression and inject a pre-fitted sklearn model
    directly, bypassing optuna.  This avoids the pre-existing bug where
    ``LossFunction_CrossEntropy_CV`` is constructed with ``labels=y`` (full
    array) instead of ``labels=np.unique(y)``, which causes a shape mismatch
    inside sklearn's ``log_loss`` in newer sklearn versions.
    """
    import sklearn.linear_model

    rng = np.random.default_rng(seed=0)
    X = rng.standard_normal((_N_SAMPLES, _LATENT_DIM)).astype(np.float32)
    y = np.array([0] * (_N_SAMPLES // 2) + [1] * (_N_SAMPLES // 2), dtype=np.int64)

    clf = Auto_LogisticRegression(
        X=X,
        y=y,
        params_LogisticRegression={
            "C": 1.0,
            "penalty": "l2",
            "solver": "lbfgs",
            "max_iter": 200,
        },
        label_names=["class_0", "class_1"],
        verbose=False,
    )
    ## Manually fit and assign model_best, bypassing the buggy optuna path.
    sklearn_lr = sklearn.linear_model.LogisticRegression(
        C=1.0, penalty="l2", solver="lbfgs", max_iter=200,
        class_weight=clf.class_weight,
    )
    sklearn_lr.fit(X, y)
    clf.model_best = sklearn_lr
    clf.best_params = {"C": 1.0}
    return clf


# ---------------------------------------------------------------------------
# Patch _pack_embedder / _unpack_embedder to work with _TinyEmbedder.
# The real methods use inspect.getfile(type(embedder)) to snapshot model.py;
# _TinyEmbedder lives in this test file, which is not a valid standalone
# importable module for the unpack path.  We instead write a tiny self-
# contained model.py stub that defines make_model() returning a _TinyEmbedder-
# equivalent module.
# ---------------------------------------------------------------------------

_STUB_MODEL_PY = '''
"""Stub model.py bundled inside test packet."""
import torch
import torch.nn as nn
import numpy as np
import torchvision

_LATENT_DIM = {latent_dim}


class _TinyNet(nn.Module):
    def __init__(self, latent_dim=_LATENT_DIM):
        super().__init__()
        self.fc = nn.Linear(3 * 224 * 224, latent_dim, bias=False)

    def forward(self, x):
        return self.fc(x.flatten(1))

    def embed(self, patches_np, device="cpu"):
        _eps = 1e-9
        x = torch.as_tensor(patches_np, dtype=torch.float32)
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
        self.eval()
        with torch.no_grad():
            return self(x).cpu().numpy()


def make_model(fwd_version="head", **params):
    net = _TinyNet(latent_dim=params.get("latent_dim", _LATENT_DIM))
    net.arch_kwargs = dict(params)
    return net
'''


class _PatchedClassifierPackage(ClassifierPackage):
    """Subclass that overrides _pack_embedder to use the stub model.py."""

    def _pack_embedder(self):
        bytes_model_py = _STUB_MODEL_PY.format(latent_dim=_LATENT_DIM).encode()
        bytes_params = json.dumps(self.embedder.arch_kwargs, indent=2).encode()
        buf = io.BytesIO()
        torch.save(self.embedder.state_dict(), buf)
        bytes_weights = buf.getvalue()
        return bytes_model_py, bytes_params, bytes_weights


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestClassifierPackageRoundTrip:
    """Round-trip: save → load → predict, asserting output shapes and types."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        self.dir_tmp = tmp_path
        self.embedder = _make_tiny_embedder()
        self.clf = _make_fitted_classifier(self.embedder)
        self.label_names = ["class_0", "class_1"]
        self.preprocessing = {
            "image_out_size": [_IMG_H, _IMG_W],
            "um_per_pixel": 1.0,
            "normalization": "per_roi_max",
        }
        self.filepath_packet = str(tmp_path / "test_model.roicat")

    def _make_pkg(self) -> _PatchedClassifierPackage:
        return _PatchedClassifierPackage(
            classifier=self.clf,
            embedder=self.embedder,
            label_names=self.label_names,
            preprocessing=self.preprocessing,
        )

    def test_save_creates_file(self):
        pkg = self._make_pkg()
        pkg.save(self.filepath_packet)
        assert Path(self.filepath_packet).exists()
        assert Path(self.filepath_packet).stat().st_size > 0

    def test_zip_contains_expected_members(self):
        pkg = self._make_pkg()
        pkg.save(self.filepath_packet)
        with zipfile.ZipFile(self.filepath_packet) as zf:
            names = set(zf.namelist())
        assert "classifier.onnx" in names
        assert "embedder/model.py" in names
        assert "embedder/params.json" in names
        assert "embedder/weights.pt" in names
        assert "preprocessing.json" in names
        assert "metadata.json" in names

    def test_metadata_fields(self):
        pkg = self._make_pkg()
        pkg.save(self.filepath_packet)
        with zipfile.ZipFile(self.filepath_packet) as zf:
            metadata = json.loads(zf.read("metadata.json"))
        assert metadata["schema_version"] == _SCHEMA_VERSION
        assert metadata["roicat_version"] == roicat.__version__
        assert metadata["label_names"] == self.label_names
        assert "embedder_model_py_sha256" in metadata
        assert "embedder_weights_sha256" in metadata
        assert "classifier_sha256" in metadata

    def test_load_roundtrip(self):
        """Full round-trip: save, load via base class, predict."""
        pkg = self._make_pkg()
        pkg.save(self.filepath_packet)

        ## Load via the base ClassifierPackage.load (uses _unpack_embedder)
        pkg_loaded = ClassifierPackage.load(self.filepath_packet)

        assert pkg_loaded.label_names == self.label_names
        assert pkg_loaded.preprocessing == self.preprocessing

    def test_predict_output_shapes(self):
        """predict() must return (N,) label_ids and (N, K) probs."""
        pkg = self._make_pkg()
        pkg.save(self.filepath_packet)
        pkg_loaded = ClassifierPackage.load(self.filepath_packet)

        rng = np.random.default_rng(seed=1)
        roi_images = rng.standard_normal((_N_ROI, _IMG_H, _IMG_W)).astype(np.float32)

        label_ids, probs = pkg_loaded.predict(roi_images=roi_images)

        assert label_ids.shape == (_N_ROI,), f"Expected ({_N_ROI},), got {label_ids.shape}"
        assert probs.shape == (_N_ROI, _N_CLASSES), \
            f"Expected ({_N_ROI}, {_N_CLASSES}), got {probs.shape}"
        assert probs.dtype == np.float32
        np.testing.assert_allclose(probs.sum(axis=1), np.ones(_N_ROI), atol=1e-5)

    def test_save_raises_if_file_exists(self):
        pkg = self._make_pkg()
        pkg.save(self.filepath_packet)
        with pytest.raises(FileExistsError):
            pkg.save(self.filepath_packet)

    def test_load_raises_on_tampered_sha256(self):
        pkg = self._make_pkg()
        pkg.save(self.filepath_packet)

        ## Tamper with metadata sha256
        with zipfile.ZipFile(self.filepath_packet, "r") as zf_in:
            members = {name: zf_in.read(name) for name in zf_in.namelist()}

        metadata = json.loads(members["metadata.json"])
        metadata["classifier_sha256"] = "deadbeef" * 8
        members["metadata.json"] = json.dumps(metadata).encode()

        filepath_tampered = str(self.dir_tmp / "tampered.roicat")
        with zipfile.ZipFile(filepath_tampered, "w") as zf_out:
            for name, data in members.items():
                zf_out.writestr(name, data)

        with pytest.raises(ValueError, match="SHA-256 mismatch"):
            ClassifierPackage.load(filepath_tampered)

    def test_auto_logistic_regression_label_names(self):
        """Auto_LogisticRegression stores label_names when provided."""
        assert self.clf.label_names == ["class_0", "class_1"]

    def test_auto_logistic_regression_label_names_default_none(self):
        """Auto_LogisticRegression defaults label_names to None."""
        rng = np.random.default_rng(seed=42)
        X = rng.standard_normal((20, 4)).astype(np.float32)
        y = np.array([0] * 10 + [1] * 10, dtype=np.int64)
        clf = Auto_LogisticRegression(
            X=X, y=y,
            params_LogisticRegression={"C": 1.0, "solver": "lbfgs"},
            verbose=False,
        )
        assert clf.label_names is None
