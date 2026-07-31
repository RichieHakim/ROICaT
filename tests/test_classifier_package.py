"""
Round-trip tests for roicat.ClassifierPackage.

The only packable embedder kind, ``ROInet_embedder``, is exercised with a stub
``model.py`` bundle in the shape of the distributed ROInet zips (exposes
``make_model``).

The stub does not implement preprocessing: the packet owns that, via
``roicat.ROInet.Preprocessor_ROI_images``. Classes are dispatched by *name*, so
the stub is named exactly as the real class is.

Runtime: ~17 min on CPU, dominated by one skl2onnx conversion per packet and by
the clean-subprocess tests against the real bundled ``model.py``. Run it in the
background rather than waiting on it.
"""

from typing import Optional, Sequence  ## typing

import json
import os
import subprocess
import sys
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

import roicat
from roicat.ROInet import Preprocessor_ROI_images
from roicat.classification.classifier import Auto_LogisticRegression
from roicat.classification.package import (
    ClassifierPackage,
    PackageIntegrityError,
    _SCHEMA_VERSION,
    _MEMBER_ORDER,
    _SUFFIX,
    _classes_json_safe,
    _class_values_to_label_ids,
    _check_transforms_packable,
)


_LATENT_DIM = 8
_N_CLASSES = 2
_N_SAMPLES = 40
_N_ROI = 6
_IMG_H, _IMG_W = 16, 16
_UM_PER_PIXEL = 1.5
## Stub networks take a small input so the tests stay fast; the real networks use
## (224, 224). Every Preprocessor_ROI_images built here must agree with this.
_IMG_SIZE_OUT = (32, 32)
_N_FEATURES_IN = 3 * _IMG_SIZE_OUT[0] * _IMG_SIZE_OUT[1]


# ---------------------------------------------------------------------------
# Stub model.py sources — self-contained, no roicat imports, no preprocessing.
# Built as parametrized templates so we can produce distinct bundled model.py
# bytes (for namespace-collision testing).
# ---------------------------------------------------------------------------

_SRC_TINYNET = '''
class _TinyNet(nn.Module):
    def __init__(self, latent_dim=_LATENT_DIM):
        super().__init__()
        self.fc = nn.Linear(_N_FEATURES_IN, latent_dim, bias=False)

    def forward(self, x):
        return self.fc(x.flatten(1))
'''


def _stub_model_py_makemodel(tag: str = "M", latent_dim: int = _LATENT_DIM, n_features_in: int = _N_FEATURES_IN) -> str:
    """A bundle in the shape of the distributed ROInet zips: exposes make_model."""
    return f'''"""Stub model.py (make_model kind) - tag={tag}."""
import torch
import torch.nn as nn

_LATENT_DIM = {latent_dim}
_N_FEATURES_IN = {n_features_in}
_TAG = "{tag}"

{_SRC_TINYNET}

def make_model(fwd_version, latent_dim=_LATENT_DIM, **params):
    net = _TinyNet(latent_dim=latent_dim)
    net.fwd_version = fwd_version
    net.tag = _TAG
    return net
'''


def _write_and_import(src: str, module_name: str, tmp_path: Path):
    """Write a stub model.py to disk and import it; returns (module, filepath)."""
    import importlib.util
    ## encoding="utf-8" is required: without it Windows uses cp1252, which
    ## mangles any non-ASCII byte and yields a SyntaxError on import. The stub
    ## source is kept ASCII-only as well, so the two defenses are redundant.
    filepath = tmp_path / f"model_{module_name}.py"
    filepath.write_text(src, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(module_name, str(filepath))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod, str(filepath)


class ROInet_embedder:
    """
    Stand-in for roicat.ROInet.ROInet_embedder. The class **name** is what
    ``_resolve_embedder`` dispatches on, so it must match exactly.
    """
    def __init__(
        self,
        filepath_model_py: str,
        net,
        arch_kwargs: dict,
        forward_pass_version: str = "head",
        path_bundle: Optional[str] = None,
        download_url: Optional[str] = None,
    ):
        self.filepath_model_py = filepath_model_py
        self.net = net
        self.params_model = dict(arch_kwargs)
        self.forward_pass_version = forward_pass_version
        self.preprocessor = Preprocessor_ROI_images(img_size_out=_IMG_SIZE_OUT, verbose=False)
        ## Mirrors the real embedder's private attributes, which _identify_embedder
        ## reads to record which network a packet holds.
        self._download_path_save = path_bundle
        self._download_url = download_url

    @property
    def arch_kwargs(self):
        return dict(self.params_model)


def _make_stub_embedder_roinet(tag: str, tmp_path: Path, with_preprocessor: bool = True) -> ROInet_embedder:
    """
    Build a stub ROInet_embedder over a make_model-style bundle.

    Args:
        with_preprocessor (bool):
            If ``False``, the ``.preprocessor`` attribute is removed, mimicking an
            embedder on which ``generate_dataloader`` / ``embed`` was never called.
    """
    mod, filepath = _write_and_import(
        src=_stub_model_py_makemodel(tag=tag),
        module_name=f"_stub_makemodel_{tag}",
        tmp_path=tmp_path,
    )
    arch_kwargs = {"latent_dim": _LATENT_DIM}
    net = mod.make_model(fwd_version="head", **arch_kwargs)
    embedder = ROInet_embedder(
        filepath_model_py=filepath,
        net=net,
        arch_kwargs=arch_kwargs,
        forward_pass_version="head",
    )
    if not with_preprocessor:
        del embedder.preprocessor
    return embedder


def _make_fitted_classifier(
    latent_dim: int = _LATENT_DIM,
    classes: Sequence = (0, 1),
) -> Auto_LogisticRegression:
    """
    Build an Auto_LogisticRegression with a manually-fitted sklearn model.

    Args:
        classes (Sequence):
            The distinct values to put in ``y``, in ascending order, so that
            ``model_best.classes_`` equals them. Non-contiguous values such as
            ``(0, 2)`` arise whenever a class is absent from the training data.
            ``_N_SAMPLES`` is split evenly, so a count that does not divide it
            (3 classes into 40) trains on the remainder-truncated rows.
            The dtype is left to numpy, so string classes are usable here too.
    """
    import sklearn.linear_model
    rng = np.random.default_rng(seed=0)
    ## float32: ClassifierPackage refuses any other dtype, because skl2onnx exports the
    ## input type from X and its ZipMap operator rejects tensor(double).
    X = rng.standard_normal((_N_SAMPLES, latent_dim)).astype(np.float32)
    n_per = _N_SAMPLES // len(classes)
    y = np.array([c for c in classes for _ in range(n_per)])
    X = X[: y.shape[0]]
    label_names = [f"class_{i}" for i in range(len(classes))]

    clf = Auto_LogisticRegression(
        X=X, y=y,
        params_LogisticRegression={"C": 1.0, "l1_ratio": 0.0, "solver": "lbfgs", "max_iter": 200},
        label_names=list(label_names),
        verbose=False,
    )
    sklearn_lr = sklearn.linear_model.LogisticRegression(
        C=1.0, l1_ratio=0.0, solver="lbfgs", max_iter=200,
        class_weight=clf.class_weight,
    )
    sklearn_lr.fit(X, y)
    clf.model_best = sklearn_lr
    clf.best_params = {"C": 1.0}
    return clf


def _roi_images(n_roi: int = _N_ROI, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed=seed)
    return rng.random((n_roi, _IMG_H, _IMG_W)).astype(np.float32)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def pkg_factory(tmp_path):
    """Factory yielding (pkg, filepath) for fresh stubs per call."""
    def _factory(tag: str = "A", label_names=None, **kwargs_pkg):
        embedder = _make_stub_embedder_roinet(tag=tag, tmp_path=tmp_path)
        pkg = ClassifierPackage(
            classifier=_make_fitted_classifier(),
            embedder=embedder,
            label_names=label_names if label_names is not None else ["class_0", "class_1"],
            size_images_in=(_IMG_H, _IMG_W),
            ## None -> taken from embedder.preprocessor
            preprocessor=kwargs_pkg.pop("preprocessor", None),
            um_per_pixel_training=kwargs_pkg.pop("um_per_pixel_training", _UM_PER_PIXEL),
            **kwargs_pkg,
        )
        return pkg, str(tmp_path / f"test_{tag}{_SUFFIX}")
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
        assert metadata["latent_dim"] == _LATENT_DIM
        ## Every member except metadata.json itself must be hashed. metadata.json
        ## holds these values, so it cannot cover itself.
        assert set(metadata["sha256"]) == {
            "preprocessing.json",
            "classifier.onnx",
            "embedder/model.py",
            "embedder/params.json",
            "embedder/weights.pt",
        }
        assert all(len(h) == 64 for h in metadata["sha256"].values())
        ## Which network this is, readable without unzipping
        for key in ("release", "bundle_md5", "download_url", "forward_pass_version",
                    "torchvision_model", "pre_head_fc_sizes", "post_head_fc_sizes"):
            assert key in metadata["embedder_identity"], key
        assert "created_at_utc" in metadata
        ## ISO-8601 UTC ends with 'Z'
        assert metadata["created_at_utc"].endswith("Z")
        ## The embedder block must fully describe how to rebuild the module
        for key in ("class_name", "factory", "attr_module", "kwarg_fwd_version", "forward_pass_version"):
            assert key in metadata["embedder"], f"metadata['embedder'] missing {key!r}"

    def test_metadata_embedder_block_roinet_kind(self, pkg_factory):
        pkg, filepath = pkg_factory(tag="RN1")
        pkg.save(filepath)
        with zipfile.ZipFile(filepath) as zf:
            metadata = json.loads(zf.read("metadata.json"))
        assert metadata["embedder"]["class_name"] == "ROInet_embedder"
        assert metadata["embedder"]["factory"] == "make_model"
        assert metadata["embedder"]["attr_module"] == ""
        assert metadata["embedder"]["kwarg_fwd_version"] == "fwd_version"
        assert metadata["embedder"]["forward_pass_version"] == "head"

    def test_preprocessing_member_contents(self, pkg_factory):
        pkg, filepath = pkg_factory()
        pkg.save(filepath)
        with zipfile.ZipFile(filepath) as zf:
            preprocessing = json.loads(zf.read("preprocessing.json"))
        assert preprocessing["size_images_in"] == [_IMG_H, _IMG_W]
        assert preprocessing["um_per_pixel_training"] == _UM_PER_PIXEL
        ## The preprocessor config must be sufficient to rebuild the preprocessor
        rebuilt = Preprocessor_ROI_images.from_dict(config=preprocessing["preprocessor"])
        assert rebuilt.to_dict() == Preprocessor_ROI_images(img_size_out=_IMG_SIZE_OUT, verbose=False).to_dict()

    def test_load_roundtrip(self, pkg_factory):
        pkg, filepath = pkg_factory()
        pkg.save(filepath)
        loaded = ClassifierPackage.load(filepath)
        assert loaded.label_names == ["class_0", "class_1"]
        assert tuple(loaded.preprocessing["size_images_in"]) == (_IMG_H, _IMG_W)
        assert loaded.latent_dim == _LATENT_DIM
        assert isinstance(loaded.preprocessor, Preprocessor_ROI_images)

    def test_predict_output_shapes(self, pkg_factory):
        pkg, filepath = pkg_factory(tag="P")
        pkg.save(filepath)
        loaded = ClassifierPackage.load(filepath)
        label_ids, probs = loaded.predict(roi_images=_roi_images(), um_per_pixel=_UM_PER_PIXEL)
        assert label_ids.shape == (_N_ROI,)
        assert probs.shape == (_N_ROI, _N_CLASSES)
        assert probs.dtype == np.float32
        np.testing.assert_allclose(probs.sum(axis=1), np.ones(_N_ROI), atol=1e-5)

    def test_predict_matches_unpacked_embedder(self, pkg_factory):
        """
        The packed pipeline must reproduce the same pipeline run directly on the
        in-memory embedder: preprocessor -> module -> classifier.
        """
        pkg, filepath = pkg_factory(tag="EQ")
        roi_images = _roi_images()

        ## Reference: run the same preprocessor + module the packer resolved
        images = pkg.preprocessor.preprocess(ROI_images=roi_images, um_per_pixel=_UM_PER_PIXEL)
        pkg._module_embedder.eval()
        with torch.no_grad():
            latents_ref = pkg._module_embedder(images).numpy()
        probs_ref = pkg.classifier.model_best.predict_proba(latents_ref.astype(np.float32))

        pkg.save(filepath)
        loaded = ClassifierPackage.load(filepath)
        _, probs = loaded.predict(roi_images=roi_images, um_per_pixel=_UM_PER_PIXEL)
        np.testing.assert_allclose(probs, probs_ref, rtol=1e-4, atol=1e-5)

    def test_predict_is_batch_size_invariant(self, pkg_factory):
        pkg, filepath = pkg_factory(tag="BS")
        pkg.save(filepath)
        loaded = ClassifierPackage.load(filepath)
        roi_images = _roi_images(n_roi=7)
        l_big, p_big = loaded.predict(roi_images=roi_images, um_per_pixel=_UM_PER_PIXEL, batch_size=1024)
        l_small, p_small = loaded.predict(roi_images=roi_images, um_per_pixel=_UM_PER_PIXEL, batch_size=2)
        np.testing.assert_array_equal(l_big, l_small)
        np.testing.assert_allclose(p_big, p_small, rtol=1e-5, atol=1e-6)

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
        nested = tmp_path / "deep" / "nested" / f"pkt{_SUFFIX}"
        pkg.save(str(nested))
        assert nested.exists()

    @pytest.mark.parametrize("name_given", ["pkt", "pkt.onnx"])
    def test_save_appends_suffix_when_missing(self, tmp_path, pkg_factory, name_given):
        """A path without the packet suffix gets it appended, not substituted."""
        pkg, _ = pkg_factory()
        pkg.save(str(tmp_path / name_given))
        assert (tmp_path / (name_given + _SUFFIX)).exists()
        assert not (tmp_path / name_given).exists()

    def test_suffix_is_roicat_classifier(self):
        """Pins the packet suffix; it is part of the user-facing contract."""
        assert _SUFFIX == ".roicat_classifier"


# ---------------------------------------------------------------------------
# Tests — preprocessing is load-bearing (the bug this schema version fixes)
# ---------------------------------------------------------------------------

class TestPreprocessingIsApplied:
    def test_um_per_pixel_changes_predictions(self, pkg_factory):
        """
        um_per_pixel must reach the scale-normalization step. If it were ignored,
        these two calls would return identical probabilities.
        """
        pkg, filepath = pkg_factory(tag="UPP")
        pkg.save(filepath)
        loaded = ClassifierPackage.load(filepath)
        roi_images = _roi_images()
        _, probs_a = loaded.predict(roi_images=roi_images, um_per_pixel=1.0)
        _, probs_b = loaded.predict(roi_images=roi_images, um_per_pixel=3.0)
        assert not np.allclose(probs_a, probs_b), (
            "predictions are invariant to um_per_pixel — scale normalization is not being applied"
        )

    def test_predict_reuses_packed_preprocessor_config(self, pkg_factory):
        """A non-default preprocessor config survives the round trip and is used."""
        preprocessor = Preprocessor_ROI_images(factor_scaleFactor=2.4, img_size_out=_IMG_SIZE_OUT, verbose=False)
        pkg, filepath = pkg_factory(tag="CFG", preprocessor=preprocessor)
        pkg.save(filepath)
        loaded = ClassifierPackage.load(filepath)
        assert loaded.preprocessor.to_dict()["factor_scaleFactor"] == 2.4

        roi_images = _roi_images()
        _, probs = loaded.predict(roi_images=roi_images, um_per_pixel=1.0)

        ## factor_scaleFactor=2.4 at 1.0 um/px gives the same scale factor as the
        ## default 1.2 at 2.0 um/px. Run the *same* loaded embedder both ways.
        images_ref = Preprocessor_ROI_images(img_size_out=_IMG_SIZE_OUT, verbose=False).preprocess(
            ROI_images=roi_images, um_per_pixel=2.0,
        )
        with torch.no_grad():
            latents_ref = loaded._module_embedder(images_ref).numpy().astype(np.float32)
        _, probs_ref = loaded._get_onnx_model()(latents_ref)
        np.testing.assert_allclose(probs, probs_ref, rtol=1e-5, atol=1e-6)

    def test_scale_normalize_false_warns(self, pkg_factory):
        """A packet built without scale normalization ignores um_per_pixel — say so."""
        pkg, filepath = pkg_factory(
            tag="NOSCALE",
            preprocessor=Preprocessor_ROI_images(scale_normalize=False, img_size_out=_IMG_SIZE_OUT, verbose=False),
        )
        pkg.save(filepath)
        loaded = ClassifierPackage.load(filepath)
        with pytest.warns(UserWarning, match="scale_normalize=False"):
            loaded.predict(roi_images=_roi_images(), um_per_pixel=_UM_PER_PIXEL)


# ---------------------------------------------------------------------------
# Tests — integrity & error model
# ---------------------------------------------------------------------------

def _rewrite_packet(
    filepath, tmp_path, name_out, fn_edit_metadata=None, drop_members=(), edit_members=None
):
    """
    Rebuild a packet zip with edited metadata / edited member bytes / dropped members.

    Args:
        edit_members (Optional[Dict[str, bytes]]):
            Member name -> replacement bytes, applied before metadata editing so a
            test can tamper with a member and leave the recorded hash stale.
    """
    with zipfile.ZipFile(filepath, "r") as zf_in:
        members = {n: zf_in.read(n) for n in zf_in.namelist() if n not in drop_members}
    if edit_members:
        members.update(edit_members)
    if fn_edit_metadata is not None and "metadata.json" in members:
        metadata = json.loads(members["metadata.json"])
        members["metadata.json"] = json.dumps(fn_edit_metadata(metadata)).encode()
    path_out = str(tmp_path / name_out)
    with zipfile.ZipFile(path_out, "w") as zf_out:
        for name, data in members.items():
            zf_out.writestr(name, data)
    return path_out


class TestIntegrity:
    def test_load_raises_on_tampered_sha256(self, pkg_factory, tmp_path):
        pkg, filepath = pkg_factory()
        pkg.save(filepath)

        def _edit(metadata):
            metadata["sha256"]["classifier.onnx"] = "deadbeef" * 8
            return metadata

        tampered = _rewrite_packet(filepath, tmp_path, f"tampered{_SUFFIX}", fn_edit_metadata=_edit)
        with pytest.raises(PackageIntegrityError, match="SHA-256 mismatch"):
            ClassifierPackage.load(tampered)

    @pytest.mark.parametrize(
        "name_member",
        ["preprocessing.json", "classifier.onnx", "embedder/model.py", "embedder/params.json"],
    )
    def test_load_raises_when_any_member_is_edited(self, pkg_factory, tmp_path, name_member):
        """
        Editing any hashed member must be caught.

        ``preprocessing.json`` is the one that matters most: it carries the
        scale-normalization configuration, so a change there alters every
        prediction while leaving the file superficially valid.
        """
        pkg, filepath = pkg_factory()
        pkg.save(filepath)

        with zipfile.ZipFile(filepath) as zf:
            data = zf.read(name_member)
        ## Flip one byte, keeping the length identical.
        mutated = bytearray(data)
        mutated[len(mutated) // 2] ^= 0x01
        edited = _rewrite_packet(
            filepath, tmp_path, f"edited{_SUFFIX}", edit_members={name_member: bytes(mutated)}
        )
        with pytest.raises(PackageIntegrityError, match="SHA-256 mismatch"):
            ClassifierPackage.load(edited)

    def test_load_raises_when_a_member_has_no_recorded_hash(self, pkg_factory, tmp_path):
        pkg, filepath = pkg_factory()
        pkg.save(filepath)

        def _edit(metadata):
            del metadata["sha256"]["preprocessing.json"]
            return metadata

        stripped = _rewrite_packet(filepath, tmp_path, f"nohash{_SUFFIX}", fn_edit_metadata=_edit)
        with pytest.raises(PackageIntegrityError, match="no SHA-256 entry"):
            ClassifierPackage.load(stripped)

    def test_load_refuses_version_1_layout_with_a_clear_message(self, pkg_factory, tmp_path):
        """
        A packet in the pre-``preprocessor`` layout must be refused at the version
        check, not crash deeper in load().

        Version-1 packets were never released but were built during development.
        Because the version number alone distinguishes them, the check has to fire
        before anything reads ``preprocessing['preprocessor']``.
        """
        pkg, filepath = pkg_factory()
        pkg.save(filepath)

        with zipfile.ZipFile(filepath) as zf:
            preprocessing = json.loads(zf.read("preprocessing.json"))
        preprocessing.pop("preprocessor", None)  ## as the version-1 writer left it

        def _edit(metadata):
            metadata["schema_version"] = 1
            return metadata

        old = _rewrite_packet(
            filepath,
            tmp_path,
            f"v1{_SUFFIX}",
            fn_edit_metadata=_edit,
            edit_members={"preprocessing.json": json.dumps(preprocessing).encode()},
        )
        with pytest.raises(PackageIntegrityError, match="Rebuild the packet"):
            ClassifierPackage.load(old)

    def test_load_raises_on_schema_version_mismatch(self, pkg_factory, tmp_path):
        pkg, filepath = pkg_factory()
        pkg.save(filepath)

        def _edit(metadata):
            metadata["schema_version"] = 999
            return metadata

        bogus = _rewrite_packet(filepath, tmp_path, f"bogus{_SUFFIX}", fn_edit_metadata=_edit)
        with pytest.raises(PackageIntegrityError, match="schema_version"):
            ClassifierPackage.load(bogus)

    def test_load_raises_on_latent_dim_mismatch(self, pkg_factory, tmp_path):
        """The embedder's output width and the classifier's input width must agree."""
        pkg, filepath = pkg_factory()
        pkg.save(filepath)

        def _edit(metadata):
            metadata["latent_dim"] = _LATENT_DIM + 1
            return metadata

        mismatched = _rewrite_packet(filepath, tmp_path, f"dim{_SUFFIX}", fn_edit_metadata=_edit)
        with pytest.raises(PackageIntegrityError, match="latent_dim mismatch"):
            ClassifierPackage.load(mismatched)

    def test_load_raises_on_missing_factory_in_bundled_model_py(self, pkg_factory, tmp_path):
        """If metadata names a factory the bundled model.py lacks, say which one."""
        pkg, filepath = pkg_factory()
        pkg.save(filepath)

        def _edit(metadata):
            metadata["embedder"]["factory"] = "not_a_real_factory"
            return metadata

        broken = _rewrite_packet(filepath, tmp_path, f"nofactory{_SUFFIX}", fn_edit_metadata=_edit)
        with pytest.raises(PackageIntegrityError, match="not_a_real_factory"):
            ClassifierPackage.load(broken)

    def test_load_raises_on_missing_member(self, pkg_factory, tmp_path):
        pkg, filepath = pkg_factory()
        pkg.save(filepath)
        stripped = _rewrite_packet(
            filepath, tmp_path, f"stripped{_SUFFIX}", drop_members=("preprocessing.json",),
        )
        with pytest.raises(KeyError, match="preprocessing.json"):
            ClassifierPackage.load(stripped)

    def test_load_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ClassifierPackage.load(str(tmp_path / f"does_not_exist{_SUFFIX}"))


# ---------------------------------------------------------------------------
# Tests — byte-determinism
# ---------------------------------------------------------------------------

class TestEmbedderIdentity:
    """
    A packet must record which network it holds, so that reading it answers the
    question the latent-width check only approximates.
    """

    def test_identity_is_recorded_and_survives_a_round_trip(self, pkg_factory):
        pkg, filepath = pkg_factory(tag="ID")
        pkg.save(filepath)
        loaded = ClassifierPackage.load(filepath)
        assert loaded.embedder_identity == pkg.embedder_identity
        assert loaded.embedder_identity["forward_pass_version"] == "head"

    def test_unknown_bundle_is_recorded_as_unknown_not_refused(self, pkg_factory):
        """A self-trained network is not one of ROICaT's releases; that is fine."""
        pkg, _ = pkg_factory(tag="UNK")
        assert pkg.embedder_identity["release"] == "unknown"
        assert pkg.embedder_identity["release_note"] is None

    def test_bundle_md5_is_hashed_from_disk(self, tmp_path):
        """
        ``bundle_md5`` must come from the file, not from a caller-supplied value,
        so that it states what was actually loaded.
        """
        import hashlib

        path_bundle = tmp_path / "ROInet.zip"
        payload = b"not a real bundle, but a real file"
        path_bundle.write_bytes(payload)

        embedder = _make_stub_embedder_roinet(tag="MD5", tmp_path=tmp_path)
        embedder._download_path_save = str(path_bundle)
        embedder._download_url = "https://example.invalid/net.zip"
        pkg = ClassifierPackage(
            classifier=_make_fitted_classifier(),
            embedder=embedder,
            label_names=["class_0", "class_1"],
            size_images_in=(_IMG_H, _IMG_W),
            um_per_pixel_training=_UM_PER_PIXEL,
        )
        assert pkg.embedder_identity["bundle_md5"] == hashlib.md5(payload).hexdigest()
        assert pkg.embedder_identity["download_url"] == "https://example.invalid/net.zip"

    def test_missing_bundle_does_not_block_packing(self, tmp_path):
        """
        The bundle is only needed at embedder construction. If the directory has
        since been cleaned, a valid embedder must still pack.
        """
        embedder = _make_stub_embedder_roinet(tag="GONE", tmp_path=tmp_path)
        embedder._download_path_save = str(tmp_path / "definitely_not_here" / "ROInet.zip")
        pkg = ClassifierPackage(
            classifier=_make_fitted_classifier(),
            embedder=embedder,
            label_names=["class_0", "class_1"],
            size_images_in=(_IMG_H, _IMG_W),
            um_per_pixel_training=_UM_PER_PIXEL,
        )
        assert pkg.embedder_identity["bundle_md5"] is None
        assert pkg.embedder_identity["release"] == "unknown"


class TestLatentsProvenance:
    """
    Catches a wrong network whose latent width happens to match the classifier's,
    which the width check cannot see.
    """

    def test_raises_when_latents_disagree_with_classifier_X(self, tmp_path):
        classifier = _make_fitted_classifier()
        embedder = _make_stub_embedder_roinet(tag="PROV", tmp_path=tmp_path)
        ## Same shape as classifier.X, different values: a different network.
        embedder.latents = np.asarray(classifier.X) + 1.0
        with pytest.raises(ValueError, match="not fit on this embedder's latents"):
            ClassifierPackage(
                classifier=classifier,
                embedder=embedder,
                label_names=["class_0", "class_1"],
                size_images_in=(_IMG_H, _IMG_W),
                um_per_pixel_training=_UM_PER_PIXEL,
            )

    def test_passes_when_latents_match(self, tmp_path):
        classifier = _make_fitted_classifier()
        embedder = _make_stub_embedder_roinet(tag="PROVOK", tmp_path=tmp_path)
        embedder.latents = np.asarray(classifier.X).copy()
        pkg = ClassifierPackage(
            classifier=classifier,
            embedder=embedder,
            label_names=["class_0", "class_1"],
            size_images_in=(_IMG_H, _IMG_W),
            um_per_pixel_training=_UM_PER_PIXEL,
        )
        assert pkg.latent_dim == _LATENT_DIM

    @pytest.mark.parametrize(
        "shape_latents",
        [(3, _LATENT_DIM), (_N_SAMPLES, _LATENT_DIM + 1)],
        ids=["fewer_rows", "different_width"],
    )
    def test_skipped_when_shapes_are_not_comparable(self, tmp_path, shape_latents):
        """
        A different image set or ordering makes an elementwise comparison
        meaningless, so it is skipped rather than reported as a mismatch.

        Neither case raises. Note the second does not trip the width check either:
        that check measures the embedder by running an image through it, so a
        stale ``.latents`` of the wrong width says nothing about the network. The
        two checks are independent, which is the point — the width check is about
        the network, this one is about which latents the classifier saw.
        """
        classifier = _make_fitted_classifier()
        embedder = _make_stub_embedder_roinet(tag=f"SKIP{shape_latents[1]}", tmp_path=tmp_path)
        embedder.latents = np.zeros(shape_latents, dtype=np.float32)
        pkg = ClassifierPackage(
            classifier=classifier,
            embedder=embedder,
            label_names=["class_0", "class_1"],
            size_images_in=(_IMG_H, _IMG_W),
            um_per_pixel_training=_UM_PER_PIXEL,
        )
        assert pkg.latent_dim == _LATENT_DIM


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
        p1 = str(tmp_path / f"a{_SUFFIX}")
        p2 = str(tmp_path / f"b{_SUFFIX}")
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

    def test_external_attr_is_0644(self, pkg_factory):
        pkg, filepath = pkg_factory()
        pkg.save(filepath)
        with zipfile.ZipFile(filepath) as zf:
            for info in zf.infolist():
                assert info.external_attr == (0o644 << 16), f"{info.filename}: bad external_attr"


# ---------------------------------------------------------------------------
# Tests — validation
# ---------------------------------------------------------------------------

class TestInitValidation:
    def _kwargs(self, tmp_path, **overrides):
        kwargs = dict(
            classifier=_make_fitted_classifier(),
            ## No .preprocessor, so `preprocessor` must be supplied explicitly here.
            embedder=_make_stub_embedder_roinet(tag="V", tmp_path=tmp_path, with_preprocessor=False),
            label_names=["class_0", "class_1"],
            size_images_in=(_IMG_H, _IMG_W),
            preprocessor=Preprocessor_ROI_images(img_size_out=_IMG_SIZE_OUT, verbose=False),
        )
        kwargs.update(overrides)
        return kwargs

    def test_label_names_length_mismatch(self, tmp_path):
        with pytest.raises(ValueError, match="label_names"):
            ClassifierPackage(**self._kwargs(tmp_path, label_names=["a", "b", "c"]))

    def test_label_names_non_str(self, tmp_path):
        with pytest.raises(TypeError, match="str"):
            ClassifierPackage(**self._kwargs(tmp_path, label_names=[0, 1]))

    def test_classifier_label_names_mismatch(self, tmp_path):
        with pytest.raises(ValueError, match="label_names"):
            ClassifierPackage(**self._kwargs(tmp_path, label_names=["a", "b"]))

    def test_preprocessor_wrong_type(self, tmp_path):
        with pytest.raises(TypeError, match="Preprocessor_ROI_images"):
            ClassifierPackage(**self._kwargs(tmp_path, preprocessor={"um_per_pixel": 1.0}))

    def test_preprocessor_missing_and_not_on_embedder(self, tmp_path):
        """
        An embedder that never ran generate_dataloader / embed has no
        .preprocessor, so one must be passed explicitly.
        """
        with pytest.raises(ValueError, match="preprocessor is None"):
            ClassifierPackage(**self._kwargs(tmp_path, preprocessor=None))

    def test_preprocessor_taken_from_embedder(self, tmp_path):
        """An ROInet_embedder carries its preprocessor; no need to pass one."""
        embedder = _make_stub_embedder_roinet(tag="FROMEMB", tmp_path=tmp_path)
        pkg = ClassifierPackage(**self._kwargs(tmp_path, embedder=embedder, preprocessor=None))
        assert pkg.preprocessor is embedder.preprocessor

    def test_size_images_in_bad_length(self, tmp_path):
        with pytest.raises(ValueError, match="size_images_in"):
            ClassifierPackage(**self._kwargs(tmp_path, size_images_in=(16, 16, 3)))

    def test_unsupported_embedder_kind(self, tmp_path):
        class SomeOtherModel:
            arch_kwargs = {}
            net = nn.Linear(2, 2)

        with pytest.raises(TypeError, match="cannot be packed"):
            ClassifierPackage(**self._kwargs(tmp_path, embedder=SomeOtherModel()))

    @pytest.mark.parametrize("name_class", ["Simclr_Model", "Simclr_Model_with_PCA"])
    def test_training_containers_specifically_rejected(self, tmp_path, name_class):
        """
        The training-time containers must be refused with a message that names
        ROInet_embedder, not with the generic 'unsupported kind' text. Dispatch is
        by class name, so a stand-in named the same is enough.
        """
        embedder = type(name_class, (), {"arch_kwargs": {}, "net": nn.Linear(2, 2)})()
        with pytest.raises(TypeError, match="not inference-facing"):
            ClassifierPackage(**self._kwargs(tmp_path, embedder=embedder))
        with pytest.raises(TypeError, match="ROInet_embedder"):
            ClassifierPackage(**self._kwargs(tmp_path, embedder=embedder))

    def test_embedder_classifier_latent_dim_mismatch(self, tmp_path):
        """
        The original silent-wrongness case: a classifier fit on one network's
        latents packed with a different network. Caught by a forward pass at pack
        time — reading the ONNX input width would compare the classifier to
        itself, since both derive from classifier.X.
        """
        clf = _make_fitted_classifier(latent_dim=_LATENT_DIM + 4)
        with pytest.raises(ValueError, match="features but the classifier was fit on"):
            ClassifierPackage(**self._kwargs(tmp_path, classifier=clf))

    def test_classifier_X_float64_refused(self, tmp_path):
        """
        ``save_model`` derives the ONNX input type from ``classifier.X``, so a float64
        X exports a ``tensor(double)`` graph and skl2onnx's ZipMap refuses it. Nothing
        constructs a session at pack time, so without this check the packet saves on
        the training machine and can never be loaded on the inference machine.
        """
        clf = _make_fitted_classifier()
        clf.X = np.asarray(clf.X, dtype=np.float64)
        with pytest.raises(TypeError, match="float32-only"):
            ClassifierPackage(**self._kwargs(tmp_path, classifier=clf))

    def test_preprocessor_output_shape_incompatible_with_embedder(self, tmp_path):
        """A preprocessor emitting a shape the network cannot accept fails at pack time."""
        preprocessor = Preprocessor_ROI_images(img_size_out=(64, 64), verbose=False)
        with pytest.raises(RuntimeError, match="could not accept the preprocessor's output shape"):
            ClassifierPackage(**self._kwargs(tmp_path, preprocessor=preprocessor))


# ---------------------------------------------------------------------------
# Tests — predict edge cases
# ---------------------------------------------------------------------------

class TestPredict:
    def test_predict_empty(self, pkg_factory):
        pkg, filepath = pkg_factory()
        pkg.save(filepath)
        loaded = ClassifierPackage.load(filepath)
        empty = np.zeros((0, _IMG_H, _IMG_W), dtype=np.float32)
        labels, probs = loaded.predict(roi_images=empty, um_per_pixel=_UM_PER_PIXEL)
        assert labels.shape == (0,)
        assert probs.shape == (0, _N_CLASSES)
        assert labels.dtype == np.int64
        assert probs.dtype == np.float32

    def test_predict_wrong_ndim_raises(self, pkg_factory):
        pkg, filepath = pkg_factory()
        pkg.save(filepath)
        loaded = ClassifierPackage.load(filepath)
        with pytest.raises(ValueError, match="3-D"):
            loaded.predict(roi_images=np.zeros((_IMG_H, _IMG_W), dtype=np.float32), um_per_pixel=_UM_PER_PIXEL)

    def test_predict_wrong_shape_raises(self, pkg_factory):
        pkg, filepath = pkg_factory()
        pkg.save(filepath)
        loaded = ClassifierPackage.load(filepath)
        with pytest.raises(ValueError, match="size_images_in"):
            loaded.predict(roi_images=np.zeros((3, 32, 32), dtype=np.float32), um_per_pixel=_UM_PER_PIXEL)

    def test_predict_bad_batch_size_raises(self, pkg_factory):
        pkg, filepath = pkg_factory()
        pkg.save(filepath)
        loaded = ClassifierPackage.load(filepath)
        with pytest.raises(ValueError, match="batch_size"):
            loaded.predict(roi_images=_roi_images(), um_per_pixel=_UM_PER_PIXEL, batch_size=0)

    def test_predict_requires_um_per_pixel(self, pkg_factory):
        """um_per_pixel is a property of the data, so it has no default."""
        pkg, filepath = pkg_factory()
        pkg.save(filepath)
        loaded = ClassifierPackage.load(filepath)
        with pytest.raises(TypeError):
            loaded.predict(_roi_images())


# ---------------------------------------------------------------------------
# Tests — class values vs label_names positions
# ---------------------------------------------------------------------------

def _pkg_with_classes(classes, tag, tmp_path):
    """Pack a classifier fit on the given class values."""
    pkg = ClassifierPackage(
        classifier=_make_fitted_classifier(classes=classes),
        embedder=_make_stub_embedder_roinet(tag=tag, tmp_path=tmp_path),
        label_names=[f"class_{i}" for i in range(len(classes))],
        size_images_in=(_IMG_H, _IMG_W),
        um_per_pixel_training=_UM_PER_PIXEL,
    )
    return pkg, str(tmp_path / f"cls_{tag}{_SUFFIX}")


class TestClassOrdering:
    """
    The ONNX classifier emits class *values*; ``label_names`` is indexed by
    *position*. They coincide only when the values are ``0..n_classes-1``.
    """

    def test_classes_recorded_and_round_tripped(self, tmp_path):
        pkg, filepath = _pkg_with_classes(classes=(0, 2), tag="cA", tmp_path=tmp_path)
        assert pkg.classes == [0, 2]
        pkg.save(filepath)
        with zipfile.ZipFile(filepath) as zf:
            metadata = json.loads(zf.read("metadata.json"))
        assert metadata["classes"] == [0, 2]
        assert ClassifierPackage.load(filepath).classes == [0, 2]

    @pytest.mark.parametrize(
        "classes", [(0, 1), (0, 2), (1, 2), (-1, 5), (0, 2, 5), ("bad", "good", "ugly")],
    )
    def test_predict_returns_positions_not_class_values(self, classes, tmp_path):
        """
        With classes (0, 2) the ONNX label output is 0 or 2, so the old cast made
        ``label_names[2]`` an IndexError. label_ids must stay in range(n_classes).

        ``(0, 2, 5)`` is the only non-binary numeric case. With two classes
        ``probs.argmax(1)`` has two outcomes and can only detect a column
        transposition, so the cross-check below barely constrains the mapping; a
        middle class is where a translation bug hides.

        ``("bad", "good", "ugly")`` is the string case. skl2onnx exports a
        ``tensor(string)`` label output rather than int64, and the dict lookup in
        ``_class_values_to_label_ids`` only hits because onnxruntime materializes it
        as Python ``str`` and not ``bytes`` — measured, and pinned here. ASCII labels
        also keep ``np.unique``'s lexicographic order equal to the ascending order
        ``load()`` checks for.
        """
        tag = "cB" + "_".join(str(c) for c in classes)
        pkg, filepath = _pkg_with_classes(classes=classes, tag=tag, tmp_path=tmp_path)
        pkg.save(filepath)
        loaded = ClassifierPackage.load(filepath)
        assert loaded.classes == list(classes)
        label_ids, probs = loaded.predict(roi_images=_roi_images(), um_per_pixel=_UM_PER_PIXEL)

        assert label_ids.dtype == np.int64
        assert label_ids.min() >= 0 and label_ids.max() < len(classes)
        ## Every id must name something.
        assert all(loaded.label_names[int(i)] for i in label_ids)
        ## probs columns are in class order, so argmax is an independent derivation
        ## of the same positions.
        np.testing.assert_array_equal(label_ids, probs.argmax(axis=1))

    def test_unmapped_class_value_raises(self):
        with pytest.raises(ValueError, match="not among the packet's classes"):
            _class_values_to_label_ids(values=np.array([0, 7]), classes=[0, 2])

    def test_classes_json_safe_rejects_unstorable(self):
        with pytest.raises(TypeError, match="cannot be stored"):
            _classes_json_safe(np.array([{"a": 1}, {"b": 2}], dtype=object))

    def test_classes_json_safe_converts_numpy_scalars(self):
        out = _classes_json_safe(np.array([0, 2], dtype=np.int64))
        assert out == [0, 2]
        assert all(type(c) is int for c in out)

    def test_load_raises_when_classes_and_label_names_disagree(self, pkg_factory, tmp_path):
        pkg, filepath = pkg_factory()
        pkg.save(filepath)

        def _edit(metadata):
            metadata["classes"] = [0, 1, 2]
            return metadata

        bad = _rewrite_packet(filepath, tmp_path, f"nclass{_SUFFIX}", fn_edit_metadata=_edit)
        with pytest.raises(PackageIntegrityError, match="label_names"):
            ClassifierPackage.load(bad)

    def test_load_refuses_permuted_classes(self, pkg_factory, tmp_path):
        """
        A permutation is the dangerous edit: right length, and ``probs`` stays
        bit-identical while every ``label_id`` moves, so the packet contradicts
        itself with no other symptom. ``metadata.json`` holds the hashes and so has
        none of its own, which is what makes the edit reachable. sklearn's
        ``classes_`` is ``np.unique(y)`` and is always ascending, so this is
        checkable at load.
        """
        pkg, filepath = pkg_factory()
        pkg.save(filepath)

        def _edit(metadata):
            metadata["classes"] = list(reversed(metadata["classes"]))
            return metadata

        bad = _rewrite_packet(filepath, tmp_path, f"perm{_SUFFIX}", fn_edit_metadata=_edit)
        with pytest.raises(PackageIntegrityError, match="not ascending"):
            ClassifierPackage.load(bad)

    def test_load_refuses_duplicate_classes(self, pkg_factory, tmp_path):
        """
        Duplicates collapse the lookup, so one position becomes unreachable and the
        other class value raises from inside ``predict`` — an error that arrives at
        inference time blaming the classifier for a metadata defect.
        """
        pkg, filepath = pkg_factory()
        pkg.save(filepath)

        def _edit(metadata):
            metadata["classes"] = [metadata["classes"][0]] * len(metadata["classes"])
            return metadata

        bad = _rewrite_packet(filepath, tmp_path, f"dupe{_SUFFIX}", fn_edit_metadata=_edit)
        with pytest.raises(PackageIntegrityError, match="contain duplicates"):
            ClassifierPackage.load(bad)

    def test_load_refuses_empty_classes(self, pkg_factory, tmp_path):
        """``label_names`` is emptied too, so the length check cannot fire first."""
        pkg, filepath = pkg_factory()
        pkg.save(filepath)

        def _edit(metadata):
            metadata["classes"] = []
            metadata["label_names"] = []
            return metadata

        bad = _rewrite_packet(filepath, tmp_path, f"noclass{_SUFFIX}", fn_edit_metadata=_edit)
        with pytest.raises(PackageIntegrityError, match="records no classes"):
            ClassifierPackage.load(bad)

    def test_load_refuses_version_2_packet(self, pkg_factory, tmp_path):
        """
        A packet in the pre-``classes`` layout must be refused at the version
        check, not crash with a KeyError deeper in load(). Version-2 packets were
        never released but were built during development.
        """
        pkg, filepath = pkg_factory()
        pkg.save(filepath)

        def _edit(metadata):
            metadata["schema_version"] = 2
            metadata.pop("classes", None)  ## as the version-2 writer left it
            return metadata

        old = _rewrite_packet(filepath, tmp_path, f"v2{_SUFFIX}", fn_edit_metadata=_edit)
        with pytest.raises(PackageIntegrityError, match="Rebuild the packet"):
            ClassifierPackage.load(old)


# ---------------------------------------------------------------------------
# Tests — custom transforms bypass the preprocessor
# ---------------------------------------------------------------------------

class TestCustomTransforms:
    """
    ``generate_dataloader(transforms=...)`` feeds the DataLoader a chain that
    ``embedder.preprocessor`` does not describe. A packet can only store the
    preprocessor, so packing such an embedder must be refused.
    """

    def test_pack_refuses_custom_transforms(self, tmp_path):
        embedder = _make_stub_embedder_roinet(tag="ctA", tmp_path=tmp_path)
        embedder._transforms_custom = True
        with pytest.raises(ValueError, match="generate_dataloader\\(transforms="):
            ClassifierPackage(
                classifier=_make_fitted_classifier(),
                embedder=embedder,
                label_names=["class_0", "class_1"],
                size_images_in=(_IMG_H, _IMG_W),
            )

    def test_pack_refuses_even_with_explicit_preprocessor(self, tmp_path):
        """
        Passing a preprocessor by hand does not help: Preprocessor_ROI_images is a
        fixed chain, so it cannot represent an arbitrary callable either way.
        """
        embedder = _make_stub_embedder_roinet(tag="ctB", tmp_path=tmp_path)
        embedder._transforms_custom = True
        with pytest.raises(ValueError, match="generate_dataloader\\(transforms="):
            ClassifierPackage(
                classifier=_make_fitted_classifier(),
                embedder=embedder,
                label_names=["class_0", "class_1"],
                size_images_in=(_IMG_H, _IMG_W),
                preprocessor=Preprocessor_ROI_images(img_size_out=_IMG_SIZE_OUT, verbose=False),
            )

    @pytest.mark.parametrize("value", [False, None])
    def test_pack_allows_default_transforms(self, value, tmp_path):
        """An absent flag (older embedder objects) and False both mean 'no bypass'."""
        embedder = _make_stub_embedder_roinet(tag=f"ctC{value}", tmp_path=tmp_path)
        if value is not None:
            embedder._transforms_custom = value
        pkg = ClassifierPackage(
            classifier=_make_fitted_classifier(),
            embedder=embedder,
            label_names=["class_0", "class_1"],
            size_images_in=(_IMG_H, _IMG_W),
        )
        assert pkg.label_names == ["class_0", "class_1"]

    def test_generate_dataloader_sets_and_clears_the_flag(self):
        """
        The real producer joined to the packet-side consumer, both directions.

        Every test above sets ``_transforms_custom`` by hand on a stub, so nothing
        would notice if the real ``ROInet_embedder`` stopped setting it.
        ``__init__`` is bypassed with ``object.__new__`` because it downloads a
        ~100 MB network bundle; ``generate_dataloader`` never touches ``self.net``
        and needs only ``self.params`` and ``self._verbose``, so it runs offline
        and in CI.
        """
        embedder = object.__new__(roicat.ROInet.ROInet_embedder)
        embedder.params = {}
        embedder._verbose = False
        kwargs = dict(
            ROI_images=[_roi_images()],
            um_per_pixel=_UM_PER_PIXEL,
            batchSize_dataloader=4,
            numWorkers_dataloader=0,
            persistentWorkers_dataloader=False,
            img_size_out=_IMG_SIZE_OUT,
        )

        transforms_custom = torch.nn.Sequential(torch.nn.Identity())
        embedder.generate_dataloader(transforms=transforms_custom, **kwargs)
        assert embedder._transforms_custom is True
        assert embedder.transforms is transforms_custom
        with pytest.raises(ValueError, match="generate_dataloader\\(transforms="):
            _check_transforms_packable(embedder)

        ## Re-running without the argument is the first of the three documented steps
        ## out of the refusal. Same object, so a flag left stale from the call above
        ## shows up here.
        embedder.generate_dataloader(**kwargs)
        assert embedder._transforms_custom is False
        assert embedder.transforms is embedder.preprocessor.transforms
        _check_transforms_packable(embedder)


# ---------------------------------------------------------------------------
# Tests — Auto_LogisticRegression: label_names, default params, string labels
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

    def test_default_params_emit_no_sklearn_warnings(self):
        """
        The default ``params_LogisticRegression`` must not use anything scikit-learn
        has deprecated. ``penalty`` is removed in 1.10 and ``l1_ratio=None`` becomes
        an error there, so a surviving FutureWarning on the default path is a
        countdown to a hard break, not cosmetic noise.

        ``simplefilter("always")`` is required: Python's per-module
        ``__warningregistry__`` deduplicates a FutureWarning after its first
        emission, so a later test run would read as falsely clean. Warnings are
        filtered to sklearn-originating ones so third-party noise cannot make this
        flaky.
        """
        rng = np.random.default_rng(seed=7)
        X = rng.standard_normal((30, 4)).astype(np.float32)
        y = np.array([0] * 15 + [1] * 15, dtype=np.int64)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ## params_LogisticRegression deliberately left at its default — that is the
            ## thing under test. Only the convergence budget is shrunk.
            clf = Auto_LogisticRegression(
                X=X, y=y,
                kwargs_convergence={"n_patience": 3, "tol_frac": 0.05, "max_trials": 4, "max_duration": 60},
                verbose=False,
            )
            clf.fit()
        offenders = [
            f"{w.category.__name__}: {w.message}"
            for w in caught
            if "sklearn" in str(w.filename) and issubclass(w.category, (FutureWarning, UserWarning))
        ]
        assert offenders == [], "scikit-learn warned on the default params:\n" + "\n".join(offenders)

    def test_string_y_fits(self):
        """
        Regression test for the ``sample_weight`` dtype fix in
        ``Autotuner_regression.__init__``; the mechanism is documented there.

        This has to call ``fit()`` — that is where the old dtype blew up, and
        ``_make_fitted_classifier`` hand-fits its sklearn model rather than going
        through ``fit()``, so the packet tests do not reach this path.
        """
        rng = np.random.default_rng(seed=3)
        classes = ["bad", "good", "ugly"]
        n_per = 20
        X = rng.standard_normal((n_per * len(classes), 6)).astype(np.float32)
        y = np.array([c for c in classes for _ in range(n_per)])
        ## Offset one feature per class so the problem is separable and the fit is
        ## not degenerate.
        for ii, c in enumerate(classes):
            X[y == c, ii] += 3.0

        clf = Auto_LogisticRegression(
            X=X, y=y,
            kwargs_convergence={"n_patience": 5, "tol_frac": 0.05, "max_trials": 8, "max_duration": 60},
            verbose=False,
        )
        assert np.asarray(clf.sample_weight).dtype == np.float64
        assert clf.label_names == classes
        model_best, _ = clf.fit()
        assert model_best.classes_.tolist() == classes

    def test_sample_weight_round_trips(self):
        """
        ``Auto_LogisticRegression`` used to compute ``self.sample_weight`` and then let
        ``super().__init__()`` overwrite it with uniform ones, so a caller's weights
        silently never reached the optuna loss or ``evaluate_model``.

        A Python ``list`` is passed on purpose. ``_objective`` does
        ``self.sample_weight[idx_train]`` with an integer array, which a bare list cannot
        do, so calling ``fit()`` pins storage, float64 conversion, and indexability at
        once. The weights are non-uniform because uniform ones are indistinguishable
        from the bug.
        """
        rng = np.random.default_rng(seed=11)
        n_samples = 40
        X = rng.standard_normal((n_samples, 5)).astype(np.float32)
        y = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2), dtype=np.int64)
        sample_weight = [0.25 if ii % 2 == 0 else 4.0 for ii in range(n_samples)]

        clf = Auto_LogisticRegression(
            X=X, y=y,
            sample_weight=sample_weight,
            kwargs_convergence={"n_patience": 3, "tol_frac": 0.05, "max_trials": 4, "max_duration": 60},
            verbose=False,
        )
        assert isinstance(clf.sample_weight, np.ndarray)
        assert clf.sample_weight.dtype == np.float64
        assert np.array_equal(clf.sample_weight, np.asarray(sample_weight))
        ## fit() is where _objective indexes sample_weight with the CV split's integer
        ## array, and where the weights reach the loss. A stored list would raise there.
        clf.fit()
        assert len(clf.loss_running) > 0

    @pytest.mark.parametrize(
        "sample_weight, error, match",
        [
            (np.ones(19, dtype=np.float64), ValueError, "sample_weight"),
            ("balanced", TypeError, "class_weight"),
            ({0: 1.0, 1: 2.0}, TypeError, "class_weight"),
        ],
    )
    def test_sample_weight_bad_input_raises(self, sample_weight, error, match):
        """
        A class-weight spec (``'balanced'`` or a dict) is a different argument, and the
        error has to name it — otherwise ``np.asarray(..., dtype=np.float64)`` raises a
        numpy message that mentions neither the argument nor the fix.
        """
        rng = np.random.default_rng(seed=12)
        X = rng.standard_normal((20, 3)).astype(np.float32)
        y = np.array([0, 1] * 10, dtype=np.int64)
        with pytest.raises(error, match=match):
            Auto_LogisticRegression(
                X=X, y=y,
                sample_weight=sample_weight,
                verbose=False,
            )

    def test_tuning_fit_intercept_completes(self):
        """
        ``'fit_intercept'`` was typed ``'real'`` with a ``{'bool': True}`` kwarg, which
        ``trial.suggest_float`` rejects, so tuning it died on trial 0.

        The default ``params_LogisticRegression`` is read off the signature and only
        ``'fit_intercept'`` is changed: overriding the dict wholesale would silently drop
        ROICaT's ``'max_iter': 1000`` and ``'l1_ratio': 0.0`` back to sklearn's
        deprecated defaults, which is a different failure.
        """
        import inspect

        params = dict(inspect.signature(Auto_LogisticRegression).parameters["params_LogisticRegression"].default)
        params["fit_intercept"] = [True, False]

        rng = np.random.default_rng(seed=14)
        X = rng.standard_normal((30, 4)).astype(np.float32)
        y = np.array([0] * 15 + [1] * 15, dtype=np.int64)
        clf = Auto_LogisticRegression(
            X=X, y=y,
            params_LogisticRegression=params,
            kwargs_convergence={"n_patience": 3, "tol_frac": 0.05, "max_trials": 6, "max_duration": 60},
            verbose=False,
        )
        model_best, params_best = clf.fit()
        assert isinstance(params_best["fit_intercept"], bool)
        assert model_best.fit_intercept == params_best["fit_intercept"]


# ---------------------------------------------------------------------------
# Tests — namespace isolation across multiple packets in one process
# ---------------------------------------------------------------------------

class TestNamespaceIsolation:
    def test_two_packets_different_model_py(self, pkg_factory):
        """Two packets with different model.py bytes coexist; both work independently."""
        pkg_a, path_a = pkg_factory(tag="A1")
        pkg_a.save(path_a)
        pkg_b, path_b = pkg_factory(tag="B2")
        pkg_b.save(path_b)

        ## Load both in the same process.
        loaded_a = ClassifierPackage.load(path_a)
        loaded_b = ClassifierPackage.load(path_b)

        assert loaded_a._module_name != loaded_b._module_name
        assert loaded_a._module_name in sys.modules
        assert loaded_b._module_name in sys.modules

        ## Both packets predict cleanly after the other was loaded.
        roi_images = _roi_images(n_roi=4, seed=7)
        la, _ = loaded_a.predict(roi_images=roi_images, um_per_pixel=_UM_PER_PIXEL)
        lb, _ = loaded_b.predict(roi_images=roi_images, um_per_pixel=_UM_PER_PIXEL)
        assert la.shape == (4,) and lb.shape == (4,)

        ## The tags survived the separate namespaces.
        assert sys.modules[loaded_a._module_name]._TAG == "A1"
        assert sys.modules[loaded_b._module_name]._TAG == "B2"


class TestBundleModuleIsolation:
    def test_two_bundles_get_different_modules(self, tmp_path):
        """
        Two ROInet bundles in one process must not share a ``model.py`` module.

        This tests ``roicat.ROInet.import_model_py_from_bundle`` rather than
        ``ROInet_embedder`` itself: the class's ``__init__`` downloads a ~100 MB
        bundle, and this helper is the whole of the import it performs. The old code
        was ``sys.path.append`` + ``import model``, so the second embedder built in a
        process got the *first* bundle's ``make_model`` back out of
        ``sys.modules['model']``, built that network, and loaded its own weights into
        it — no error, and the packet then shipped one bundle's ``model.py`` beside
        another's forward pass.
        """
        _, filepath_a = _write_and_import(
            src=_stub_model_py_makemodel(tag="bundleA"),
            module_name="_bundle_a",
            tmp_path=tmp_path,
        )
        _, filepath_b = _write_and_import(
            src=_stub_model_py_makemodel(tag="bundleB"),
            module_name="_bundle_b",
            tmp_path=tmp_path,
        )

        imported_a = roicat.ROInet.import_model_py_from_bundle(filepath_model_py=filepath_a)
        imported_b = roicat.ROInet.import_model_py_from_bundle(filepath_model_py=filepath_b)

        assert imported_a is not imported_b
        assert imported_a.__name__ != imported_b.__name__
        assert imported_a._TAG == "bundleA"
        assert imported_b._TAG == "bundleB"
        ## The tag is carried by the network each bundle builds, which is what an
        ## embedder holds and a packet is built from.
        assert imported_a.make_model(fwd_version="head").tag == "bundleA"
        assert imported_b.make_model(fwd_version="head").tag == "bundleB"

        ## The module name is derived from the file's bytes, so the same bundle
        ## imported twice is the same object — one embedder in a process behaves
        ## exactly as it did under the old bare import.
        assert roicat.ROInet.import_model_py_from_bundle(filepath_model_py=filepath_a) is imported_a
        assert sys.modules[imported_a.__name__]._TAG == "bundleA"
        assert sys.modules[imported_b.__name__]._TAG == "bundleB"


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
            capture_output=True, text=True, timeout=120, env=env,
        )
        assert result.returncode == 0, f"stdout={result.stdout!r}\nstderr={result.stderr!r}"
        assert "OK" in result.stdout
