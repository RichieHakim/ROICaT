"""
Self-contained classifier packet for ROICaT inference.

A ``.roicat`` file is a zip archive containing a trained ONNX classifier, a
snapshot of the embedder's module definition + weights, preprocessing config,
and metadata (label names, SHA-256 hashes, ROICaT version).  The packet
survives ROICaT version drift because the embedder's ``.py`` definition travels
with the weights.

Only ``Simclr_Model_with_PCA`` embedders are supported.  The pack/unpack path
uses the ``make_model`` factory from the bundled ``model.py``, matching exactly
what ``ROInet_embedder`` does when loading a distributed ``.zip`` bundle from OSF.

Spec deviation: the spec refers to the embedder type as ``Simclr_Model``, but
the production-ready class is ``Simclr_Model_with_PCA`` (an ``nn.Module`` with
a proper ``state_dict``).  ``Simclr_Model`` is a plain Python container with no
``state_dict``.  Using ``Simclr_Model_with_PCA`` is consistent with the design
memo's statement "matches what ROInet already does for its OSF download" and is
the only coherent choice for weight serialisation.
"""

from typing import List, Tuple

import hashlib
import importlib.util
import inspect
import io
import json
import os
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import torch

import roicat
from roicat.classification.classifier import Auto_LogisticRegression, ONNX_model_sklearnLogisticRegression


## Names of member files inside the .roicat zip.
_MEMBER_CLASSIFIER = "classifier.onnx"
_MEMBER_EMBEDDER_MODEL_PY = "embedder/model.py"
_MEMBER_EMBEDDER_PARAMS = "embedder/params.json"
_MEMBER_EMBEDDER_WEIGHTS = "embedder/weights.pt"
_MEMBER_PREPROCESSING = "preprocessing.json"
_MEMBER_METADATA = "metadata.json"

_SCHEMA_VERSION = 1


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class ClassifierPackage:
    """
    Self-describing inference artifact bundling embedder + ONNX classifier + label schema.

    Pack on the training machine, load and call ``predict`` anywhere — no other
    files needed beyond the ``.roicat`` zip.

    Args:
        classifier (Auto_LogisticRegression):
            A fitted ``Auto_LogisticRegression`` instance (``fit()`` already called).
        embedder:
            A ``Simclr_Model_with_PCA`` instance with ``arch_kwargs`` set (e.g.
            as returned by ``make_model``).  Must have ``state_dict()``.
        label_names (List[str]):
            Human-readable class names in integer-label order (same ordering
            as ``np.unique(y)`` used during training).
        preprocessing (dict):
            Keys: ``image_out_size`` (Tuple[int,int]), ``um_per_pixel``
            (float), ``normalization`` (str).  Stored verbatim; not used by
            ``predict`` but surfaced for downstream scripts.
    """

    def __init__(
        self,
        classifier: Auto_LogisticRegression,
        embedder,
        label_names: List[str],
        preprocessing: dict,
    ):
        assert classifier.model_best is not None, \
            "classifier.model_best is None — call classifier.fit() before packing."
        assert hasattr(embedder, 'state_dict'), \
            "embedder must be an nn.Module with state_dict() (expected Simclr_Model_with_PCA)."
        assert hasattr(embedder, 'arch_kwargs'), \
            "embedder.arch_kwargs is missing — ensure the embedder was built via make_model()."

        self.classifier = classifier
        self.embedder = embedder
        self.label_names = list(label_names)
        self.preprocessing = dict(preprocessing)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        Serialise the packet to a ``.roicat`` zip file.

        Args:
            path (str):
                Destination path.  The ``.roicat`` extension is not enforced
                but is conventional.

        Raises:
            FileExistsError: if ``path`` already exists.
        """
        if os.path.exists(path):
            raise FileExistsError(f"Packet file already exists: {path}")

        ## --- gather bytes for each member ---
        bytes_classifier = self._pack_classifier()
        bytes_model_py, bytes_params, bytes_weights = self._pack_embedder()

        bytes_preprocessing = json.dumps(self.preprocessing, indent=2).encode()
        metadata = {
            "schema_version": _SCHEMA_VERSION,
            "roicat_version": roicat.__version__,
            "label_names": self.label_names,
            "embedder_model_py_sha256": _sha256_bytes(bytes_model_py),
            "embedder_weights_sha256": _sha256_bytes(bytes_weights),
            "classifier_sha256": _sha256_bytes(bytes_classifier),
        }
        bytes_metadata = json.dumps(metadata, indent=2).encode()

        ## --- write zip atomically: write to .tmp then rename ---
        path_tmp = path + ".tmp"
        try:
            with zipfile.ZipFile(path_tmp, mode="w", compression=zipfile.ZIP_STORED) as zf:
                zf.writestr(_MEMBER_CLASSIFIER, bytes_classifier)
                zf.writestr(_MEMBER_EMBEDDER_MODEL_PY, bytes_model_py)
                zf.writestr(_MEMBER_EMBEDDER_PARAMS, bytes_params)
                zf.writestr(_MEMBER_EMBEDDER_WEIGHTS, bytes_weights)
                zf.writestr(_MEMBER_PREPROCESSING, bytes_preprocessing)
                zf.writestr(_MEMBER_METADATA, bytes_metadata)
            os.replace(path_tmp, path)
        except Exception:
            if os.path.exists(path_tmp):
                os.remove(path_tmp)
            raise

    @classmethod
    def load(cls, path: str) -> "ClassifierPackage":
        """
        Reconstruct a ``ClassifierPackage`` from a ``.roicat`` zip file.

        Args:
            path (str):
                Path to the ``.roicat`` file.

        Returns:
            (ClassifierPackage):
                Loaded package; ``embedder`` and ``classifier`` are
                ready for inference.

        Raises:
            ValueError: on schema version mismatch or SHA-256 mismatch.
            KeyError: if a required member is missing from the zip.
        """
        with zipfile.ZipFile(path, mode="r") as zf:
            _require_members(zf, [
                _MEMBER_CLASSIFIER,
                _MEMBER_EMBEDDER_MODEL_PY,
                _MEMBER_EMBEDDER_PARAMS,
                _MEMBER_EMBEDDER_WEIGHTS,
                _MEMBER_PREPROCESSING,
                _MEMBER_METADATA,
            ])

            bytes_metadata = zf.read(_MEMBER_METADATA)
            metadata = json.loads(bytes_metadata)

            ## Schema version guard
            schema_version = metadata.get("schema_version")
            if schema_version != _SCHEMA_VERSION:
                raise ValueError(
                    f"Unsupported schema_version {schema_version!r}; "
                    f"this code handles version {_SCHEMA_VERSION}."
                )

            bytes_classifier = zf.read(_MEMBER_CLASSIFIER)
            bytes_model_py = zf.read(_MEMBER_EMBEDDER_MODEL_PY)
            bytes_params = zf.read(_MEMBER_EMBEDDER_PARAMS)
            bytes_weights = zf.read(_MEMBER_EMBEDDER_WEIGHTS)
            bytes_preprocessing = zf.read(_MEMBER_PREPROCESSING)

        ## SHA-256 integrity checks
        _check_sha256("embedder_model_py", bytes_model_py, metadata["embedder_model_py_sha256"])
        _check_sha256("embedder_weights", bytes_weights, metadata["embedder_weights_sha256"])
        _check_sha256("classifier", bytes_classifier, metadata["classifier_sha256"])

        label_names = metadata["label_names"]
        preprocessing = json.loads(bytes_preprocessing)
        embedder = cls._unpack_embedder(bytes_model_py=bytes_model_py, bytes_params=bytes_params, bytes_weights=bytes_weights)
        onnx_model = ONNX_model_sklearnLogisticRegression(path_or_bytes=bytes_classifier)

        ## Build a lightweight shell so predict() has .model_best set
        obj = object.__new__(cls)
        obj.label_names = label_names
        obj.preprocessing = preprocessing
        obj.embedder = embedder
        obj._onnx_model = onnx_model
        obj.classifier = None  # not reconstructed from ONNX; use _onnx_model directly
        return obj

    def predict(self, roi_images: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Embed ROI patches and classify them.

        Args:
            roi_images (np.ndarray):
                ROI image patches.  Shape: ``(N, H, W)``.

        Returns:
            (Tuple[np.ndarray, np.ndarray]):
                label_ids (np.ndarray):
                    Integer class predictions.  Shape: ``(N,)``.
                probs (np.ndarray):
                    Class probabilities.  Shape: ``(N, K)`` float32.
        """
        latents = self.embedder.embed(patches_np=roi_images)  # (N, pca_size) float32 np
        onnx_model = self._onnx_model if hasattr(self, '_onnx_model') else ONNX_model_sklearnLogisticRegression(
            path_or_bytes=self._pack_classifier()
        )
        label_ids, probs = onnx_model(latents)
        return label_ids, probs

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pack_classifier(self) -> bytes:
        """Serialise the fitted classifier to ONNX bytes."""
        onnx_proto = self.classifier.save_model(filepath=None)
        return onnx_proto.SerializeToString()

    def _pack_embedder(self) -> Tuple[bytes, bytes, bytes]:
        """
        Snapshot the embedder into (model_py_bytes, params_json_bytes, weights_pt_bytes).

        ``model_py_bytes``: verbatim bytes of the module file that defines
        ``Simclr_Model_with_PCA`` (obtained via ``inspect.getfile``).
        ``params_json_bytes``: ``arch_kwargs`` as compact JSON.
        ``weights_pt_bytes``: ``state_dict`` saved via ``torch.save``.
        """
        ## model.py snapshot — use inspect to locate the actual source file
        filepath_model_py = inspect.getfile(type(self.embedder))
        bytes_model_py = Path(filepath_model_py).read_bytes()

        ## arch params
        bytes_params = json.dumps(self.embedder.arch_kwargs, indent=2).encode()

        ## state dict
        buf = io.BytesIO()
        torch.save(self.embedder.state_dict(), buf)
        bytes_weights = buf.getvalue()

        return bytes_model_py, bytes_params, bytes_weights

    @staticmethod
    def _unpack_embedder(bytes_model_py: bytes, bytes_params: bytes, bytes_weights: bytes):
        """
        Reconstruct a ``Simclr_Model_with_PCA`` from packed bytes.

        Writes ``model.py`` to a temp directory, imports it via
        ``importlib.util.spec_from_file_location``, calls ``make_model``,
        then loads the state dict.

        Args:
            bytes_model_py (bytes): Raw source bytes of the model module.
            bytes_params (bytes): JSON bytes of arch_kwargs.
            bytes_weights (bytes): ``torch.save``-serialised state dict bytes.

        Returns:
            Reconstructed embedder (``nn.Module``), eval mode, on CPU.
        """
        arch_kwargs = json.loads(bytes_params)

        with tempfile.TemporaryDirectory() as dir_tmp:
            filepath_model_py = os.path.join(dir_tmp, "model.py")
            Path(filepath_model_py).write_bytes(bytes_model_py)

            ## Dynamically import the bundled model.py
            spec = importlib.util.spec_from_file_location("_roicat_bundled_model", filepath_model_py)
            module_model = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module_model)

            ## Prefer make_model factory (same as ROInet_embedder); fall back to class
            if hasattr(module_model, 'make_model'):
                net = module_model.make_model(fwd_version='head', **arch_kwargs)
            elif hasattr(module_model, 'Simclr_Model_with_PCA'):
                net = module_model.Simclr_Model_with_PCA(**arch_kwargs)
            else:
                raise AttributeError(
                    "Bundled model.py has neither make_model() nor Simclr_Model_with_PCA. "
                    "The packet may be corrupt or from an incompatible ROICaT version."
                )

            ## Load weights (must happen inside the tempdir context so the file still exists)
            buf = io.BytesIO(bytes_weights)
            state_dict = torch.load(buf, map_location='cpu', weights_only=True)
            net.load_state_dict(state_dict)

        net.eval()
        return net


# ------------------------------------------------------------------
# Module-level helpers (not part of public API)
# ------------------------------------------------------------------

def _require_members(zf: zipfile.ZipFile, names: List[str]) -> None:
    """Raise KeyError if any member name is absent from the zip."""
    zip_names = set(zf.namelist())
    for name in names:
        if name not in zip_names:
            raise KeyError(f"Required member '{name}' not found in packet zip.")


def _check_sha256(label: str, data: bytes, expected: str) -> None:
    """Raise ValueError if the SHA-256 of data does not match expected."""
    actual = _sha256_bytes(data)
    if actual != expected:
        raise ValueError(
            f"SHA-256 mismatch for {label}: "
            f"expected {expected!r}, got {actual!r}. "
            "The packet may be corrupt or tampered."
        )
