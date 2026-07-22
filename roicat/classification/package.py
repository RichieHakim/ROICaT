"""
Self-contained classifier packet for ROICaT inference.

A ``.roicat`` file is a zip archive containing a trained ONNX classifier, a
snapshot of the embedder's module definition + weights, preprocessing config,
and metadata (label names, SHA-256 hashes, ROICaT version). The packet
survives ROICaT version drift because the embedder's ``.py`` definition travels
with the weights.

Embedder type is ``roicat.model_training.model.Simclr_Model`` — a container
holding an inner ``nn.Module`` at ``self.model``. The packet stores
``embedder.model.state_dict()`` and rebuilds the container on load via
``Simclr_Model(**arch_kwargs)`` then ``embedder.model.load_state_dict(...)``.
PCA is intentionally not included in this workflow.
"""

import datetime
import hashlib
import importlib.util
import inspect
import io
import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

import roicat
from roicat.classification.classifier import Auto_LogisticRegression, ONNX_model_sklearnLogisticRegression


_MEMBER_METADATA = "metadata.json"
_MEMBER_PREPROCESSING = "preprocessing.json"
_MEMBER_CLASSIFIER = "classifier.onnx"
_MEMBER_EMBEDDER_MODEL_PY = "embedder/model.py"
_MEMBER_EMBEDDER_PARAMS = "embedder/params.json"
_MEMBER_EMBEDDER_WEIGHTS = "embedder/weights.pt"

## Canonical member write order (matches API contract §8).
_MEMBER_ORDER = (
    _MEMBER_METADATA,
    _MEMBER_PREPROCESSING,
    _MEMBER_CLASSIFIER,
    _MEMBER_EMBEDDER_MODEL_PY,
    _MEMBER_EMBEDDER_PARAMS,
    _MEMBER_EMBEDDER_WEIGHTS,
)

_SCHEMA_VERSION = 1
_DEFAULT_BATCH_SIZE = 1024
_DEFAULT_FWD_VERSION = "head"

## Deterministic ZIP defaults: epoch-zero mtime (zip-safe min is 1980), 0o644 perms,
## DEFLATE level 6.
_ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)
_ZIP_EXTERNAL_ATTR = 0o644 << 16
_ZIP_COMPRESS_LEVEL = 6


class PackageIntegrityError(RuntimeError):
    """Raised when a ``.roicat`` packet fails SHA-256 or schema validation on load."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _validate_classifier_package_inputs(classifier, embedder, label_names, preprocessing):
    """Cross-check init inputs against the contract; raise loudly on any violation."""
    if not isinstance(classifier, Auto_LogisticRegression):
        raise TypeError(f"classifier must be Auto_LogisticRegression, got {type(classifier).__name__}.")
    if getattr(classifier, "model_best", None) is None:
        raise RuntimeError("classifier.model_best is None — call classifier.fit() before packing.")
    if not hasattr(embedder, "arch_kwargs"):
        raise AttributeError("embedder is missing 'arch_kwargs'.")
    if not hasattr(embedder, "model") or not hasattr(embedder.model, "state_dict"):
        raise AttributeError("embedder must expose 'model' (an nn.Module with state_dict()).")
    if not hasattr(embedder, "embed"):
        raise AttributeError("embedder is missing 'embed' method.")

    if not isinstance(label_names, (list, tuple)):
        raise TypeError(f"label_names must be list[str], got {type(label_names).__name__}.")
    if not all(isinstance(n, str) for n in label_names):
        raise TypeError("label_names must all be str.")

    n_classes = len(classifier.model_best.classes_)
    if len(label_names) != n_classes:
        raise ValueError(f"len(label_names)={len(label_names)} must equal n_classes={n_classes}.")
    if classifier.label_names is not None and list(classifier.label_names) != list(label_names):
        raise ValueError(
            f"classifier.label_names={list(classifier.label_names)!r} mismatches "
            f"provided label_names={list(label_names)!r}."
        )

    if not isinstance(preprocessing, dict):
        raise TypeError(f"preprocessing must be dict, got {type(preprocessing).__name__}.")
    for key in ("image_out_size", "um_per_pixel", "normalization"):
        if key not in preprocessing:
            raise KeyError(f"preprocessing missing required key: {key!r}.")


class ClassifierPackage:
    """
    Self-contained inference artifact bundling embedder + ONNX classifier + label schema.

    Pack on the training machine, load and call ``predict`` anywhere — no other
    files needed beyond the ``.roicat`` zip.

    Args:
        classifier (Auto_LogisticRegression): Fitted instance.
        embedder (Simclr_Model): Built with ``Simclr_Model(...)``; must expose
            ``model`` (nn.Module), ``arch_kwargs`` (JSON-safe dict), ``embed``.
        label_names (List[str]): Class names in ``classifier.model_best.classes_`` order.
        preprocessing (dict): Must include ``image_out_size``, ``um_per_pixel``,
            ``normalization``. Round-tripped verbatim.
    """

    def __init__(
        self,
        classifier: Auto_LogisticRegression,
        embedder,
        label_names: List[str],
        preprocessing: dict,
    ):
        _validate_classifier_package_inputs(classifier, embedder, label_names, preprocessing)
        self.classifier = classifier
        self.embedder = embedder
        self.label_names = list(label_names)
        self.preprocessing = dict(preprocessing)
        self._device = "cpu"
        self._onnx_model: Optional[ONNX_model_sklearnLogisticRegression] = None
        self._tmpdir: Optional[tempfile.TemporaryDirectory] = None
        self._module_name: Optional[str] = None

    # ------------------------------------------------------------------
    # save / load
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path], overwrite: bool = False) -> None:
        """
        Serialise the packet to a ``.roicat`` zip file.

        Byte-deterministic: identical inputs produce identical bytes (zeroed
        mtimes, fixed member order, DEFLATE level 6, 0o644 attrs). Atomic write
        via ``{path}.tmp`` + ``os.replace``.

        Args:
            path: Destination. ``.roicat`` auto-appended if missing.
                Parent directories created.
            overwrite: If False, ``FileExistsError`` when ``path`` exists.
        """
        path = Path(path)
        if path.suffix != ".roicat":
            path = path.with_suffix(path.suffix + ".roicat") if path.suffix else path.with_suffix(".roicat")
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not overwrite:
            raise FileExistsError(f"Packet file already exists: {path}")

        ## Gather member bytes
        bytes_classifier = self._pack_classifier()
        bytes_model_py, bytes_params, bytes_weights = self._pack_embedder()
        bytes_preprocessing = json.dumps(self.preprocessing, indent=2, sort_keys=True).encode()

        metadata = {
            "schema_version": _SCHEMA_VERSION,
            "roicat_version": roicat.__version__,
            "label_names": list(self.label_names),
            "embedder_forward_pass_version": _DEFAULT_FWD_VERSION,
            "embedder_model_py_sha256": _sha256_bytes(bytes_model_py),
            "embedder_weights_sha256": _sha256_bytes(bytes_weights),
            "classifier_sha256": _sha256_bytes(bytes_classifier),
            "created_at_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        bytes_metadata = json.dumps(metadata, indent=2, sort_keys=True).encode()

        member_bytes = {
            _MEMBER_METADATA: bytes_metadata,
            _MEMBER_PREPROCESSING: bytes_preprocessing,
            _MEMBER_CLASSIFIER: bytes_classifier,
            _MEMBER_EMBEDDER_MODEL_PY: bytes_model_py,
            _MEMBER_EMBEDDER_PARAMS: bytes_params,
            _MEMBER_EMBEDDER_WEIGHTS: bytes_weights,
        }

        path_tmp = str(path) + ".tmp"
        try:
            with zipfile.ZipFile(path_tmp, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=_ZIP_COMPRESS_LEVEL) as zf:
                for member_name in _MEMBER_ORDER:
                    zi = zipfile.ZipInfo(filename=member_name, date_time=_ZIP_EPOCH)
                    zi.external_attr = _ZIP_EXTERNAL_ATTR
                    zi.compress_type = zipfile.ZIP_DEFLATED
                    zf.writestr(zi, member_bytes[member_name], compresslevel=_ZIP_COMPRESS_LEVEL)
            os.replace(path_tmp, path)
        except Exception:
            if os.path.exists(path_tmp):
                os.remove(path_tmp)
            raise

    @classmethod
    def load(cls, path: Union[str, Path], device: str = "cpu") -> "ClassifierPackage":
        """
        Reconstruct a ``ClassifierPackage`` from a ``.roicat`` zip file.

        Args:
            path: Path to ``.roicat`` file.
            device: Torch device string for embedder weights (default ``"cpu"``).

        Returns:
            Loaded package; embedder is in eval mode on ``device``.

        Raises:
            FileNotFoundError: missing file.
            PackageIntegrityError: schema or SHA-256 mismatch.
            KeyError: missing required zip member.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Packet file not found: {path}")

        with zipfile.ZipFile(path, mode="r") as zf:
            _require_members(zf, list(_MEMBER_ORDER))
            bytes_metadata = zf.read(_MEMBER_METADATA)
            metadata = json.loads(bytes_metadata)
            schema_version = metadata.get("schema_version")
            if schema_version != _SCHEMA_VERSION:
                raise PackageIntegrityError(
                    f"Unsupported schema_version {schema_version!r}; this code handles {_SCHEMA_VERSION}."
                )
            bytes_classifier = zf.read(_MEMBER_CLASSIFIER)
            bytes_model_py = zf.read(_MEMBER_EMBEDDER_MODEL_PY)
            bytes_params = zf.read(_MEMBER_EMBEDDER_PARAMS)
            bytes_weights = zf.read(_MEMBER_EMBEDDER_WEIGHTS)
            bytes_preprocessing = zf.read(_MEMBER_PREPROCESSING)

        ## SHA-256 integrity checks (raise, not warn)
        _check_sha256("embedder_model_py", bytes_model_py, metadata["embedder_model_py_sha256"])
        _check_sha256("embedder_weights", bytes_weights, metadata["embedder_weights_sha256"])
        _check_sha256("classifier", bytes_classifier, metadata["classifier_sha256"])

        label_names = list(metadata["label_names"])
        preprocessing = json.loads(bytes_preprocessing)
        if "image_out_size" in preprocessing and isinstance(preprocessing["image_out_size"], list):
            preprocessing["image_out_size"] = tuple(preprocessing["image_out_size"])
        fwd_version = metadata.get("embedder_forward_pass_version", _DEFAULT_FWD_VERSION)

        embedder, tmpdir, module_name = cls._unpack_embedder(
            bytes_model_py=bytes_model_py,
            bytes_params=bytes_params,
            bytes_weights=bytes_weights,
            device=device,
            fwd_version=fwd_version,
        )
        onnx_model = ONNX_model_sklearnLogisticRegression(path_or_bytes=bytes_classifier)

        obj = object.__new__(cls)
        obj.classifier = None  # not reconstructed from ONNX; use _onnx_model directly
        obj.embedder = embedder
        obj.label_names = label_names
        obj.preprocessing = preprocessing
        obj._device = device
        obj._onnx_model = onnx_model
        obj._tmpdir = tmpdir
        obj._module_name = module_name
        return obj

    def __del__(self):
        ## Best-effort cleanup of bundled-module namespace + tempdir.
        try:
            name = getattr(self, "_module_name", None)
            if name and name in sys.modules:
                del sys.modules[name]
        except Exception:
            pass
        try:
            tmp = getattr(self, "_tmpdir", None)
            if tmp is not None:
                tmp.cleanup()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------

    def predict(
        self,
        roi_images: np.ndarray,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        device: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Embed ROI patches and classify them.

        Args:
            roi_images: ``(N, H, W)`` array; ``(H, W)`` must equal
                ``preprocessing["image_out_size"]``.
            batch_size: Chunk size for embed + ONNX (default 1024).
            device: Torch device override. Defaults to the device set at ``load()``.

        Returns:
            ``(label_ids: (N,) int64, probs: (N, K) float32)`` where
            ``K == len(label_names)``.

        Raises:
            ValueError: bad ndim or HxW mismatch.
        """
        if roi_images.ndim != 3:
            raise ValueError(f"roi_images must be 3-D (N, H, W), got ndim={roi_images.ndim}.")
        expected_hw = tuple(self.preprocessing["image_out_size"])
        if tuple(roi_images.shape[1:]) != expected_hw:
            raise ValueError(
                f"roi_images shape[1:]={tuple(roi_images.shape[1:])} != image_out_size={expected_hw}."
            )

        n_classes = len(self.label_names)
        N = roi_images.shape[0]
        if N == 0:
            return np.empty((0,), dtype=np.int64), np.empty((0, n_classes), dtype=np.float32)

        ## Replace NaN/Inf with 0 (predict-time only; embed assumes finite input).
        if not np.isfinite(roi_images).all():
            roi_images = np.nan_to_num(roi_images, copy=True)

        dev = device if device is not None else self._device
        onnx_model = self._get_onnx_model()

        labels_chunks = []
        probs_chunks = []
        bs = max(int(batch_size), 1)
        for start in range(0, N, bs):
            chunk = roi_images[start:start + bs]
            latents = self.embedder.embed(patches_np=chunk, device=dev)  # (n, D) float32
            label_ids, probs = onnx_model(latents)
            labels_chunks.append(np.asarray(label_ids, dtype=np.int64))
            probs_chunks.append(np.asarray(probs, dtype=np.float32))
        return np.concatenate(labels_chunks, axis=0), np.concatenate(probs_chunks, axis=0)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _get_onnx_model(self) -> ONNX_model_sklearnLogisticRegression:
        if self._onnx_model is None:
            self._onnx_model = ONNX_model_sklearnLogisticRegression(path_or_bytes=self._pack_classifier())
        return self._onnx_model

    def _pack_classifier(self) -> bytes:
        onnx_proto = self.classifier.save_model(filepath=None)
        return onnx_proto.SerializeToString()

    def _pack_embedder(self) -> Tuple[bytes, bytes, bytes]:
        """Snapshot (model_py_bytes, params_json_bytes, weights_pt_bytes)."""
        filepath_model_py = inspect.getfile(type(self.embedder))
        bytes_model_py = Path(filepath_model_py).read_bytes()

        ## arch_kwargs must be JSON-safe; json.dumps raises TypeError otherwise.
        bytes_params = json.dumps(self.embedder.arch_kwargs, indent=2, sort_keys=True).encode()

        ## state_dict from inner nn.Module; tensors on CPU for portability.
        sd = self.embedder.model.state_dict()
        sd_cpu = {k: v.detach().cpu() if hasattr(v, "detach") else v for k, v in sd.items()}
        buf = io.BytesIO()
        torch.save(sd_cpu, buf, _use_new_zipfile_serialization=True)
        bytes_weights = buf.getvalue()

        return bytes_model_py, bytes_params, bytes_weights

    @staticmethod
    def _unpack_embedder(
        bytes_model_py: bytes,
        bytes_params: bytes,
        bytes_weights: bytes,
        device: str = "cpu",
        fwd_version: str = _DEFAULT_FWD_VERSION,
    ) -> Tuple[object, tempfile.TemporaryDirectory, str]:
        """
        Reconstruct the embedder from packed bytes.

        Returns:
            (embedder, tmpdir, module_name) — caller owns the tempdir lifetime.
        """
        arch_kwargs = json.loads(bytes_params)
        sha = hashlib.sha256(bytes_model_py).hexdigest()[:12]
        module_name = f"roicat_pkg_{sha}"

        tmpdir = tempfile.TemporaryDirectory()
        filepath_model_py = os.path.join(tmpdir.name, "model.py")
        Path(filepath_model_py).write_bytes(bytes_model_py)

        spec = importlib.util.spec_from_file_location(module_name, filepath_model_py)
        module_model = importlib.util.module_from_spec(spec)
        ## Insert into sys.modules so any internal pickle/torch references resolve.
        sys.modules[module_name] = module_model
        try:
            spec.loader.exec_module(module_model)
        except Exception:
            sys.modules.pop(module_name, None)
            tmpdir.cleanup()
            raise

        ## Prefer Simclr_Model (this workflow). Fall back to make_model factory or PCA class.
        try:
            if hasattr(module_model, "Simclr_Model"):
                ## Strip None entries so optional kwargs (e.g. torchvision_model) round-trip cleanly.
                kwargs = {k: v for k, v in arch_kwargs.items() if v is not None}
                net = module_model.Simclr_Model(**kwargs)
                buf = io.BytesIO(bytes_weights)
                state_dict = torch.load(buf, map_location=device, weights_only=True)
                net.model.load_state_dict(state_dict)
                net.model.to(device).eval()
            elif hasattr(module_model, "make_model"):
                net = module_model.make_model(fwd_version=fwd_version, **arch_kwargs)
                buf = io.BytesIO(bytes_weights)
                state_dict = torch.load(buf, map_location=device, weights_only=True)
                net.load_state_dict(state_dict)
                net.to(device).eval()
            else:
                raise AttributeError(
                    "Bundled model.py has no Simclr_Model or make_model — packet incompatible."
                )
        except Exception:
            sys.modules.pop(module_name, None)
            tmpdir.cleanup()
            raise

        return net, tmpdir, module_name


# ----------------------------------------------------------------------
# Module-level helpers
# ----------------------------------------------------------------------

def _require_members(zf: zipfile.ZipFile, names: List[str]) -> None:
    zip_names = set(zf.namelist())
    for name in names:
        if name not in zip_names:
            raise KeyError(f"Required member '{name}' not found in packet zip.")


def _check_sha256(label: str, data: bytes, expected: str) -> None:
    actual = _sha256_bytes(data)
    if actual != expected:
        raise PackageIntegrityError(
            f"SHA-256 mismatch for {label}: expected {expected!r}, got {actual!r}."
        )


__all__ = ["ClassifierPackage", "PackageIntegrityError"]
