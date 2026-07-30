"""
Self-contained classifier packet for ROICaT inference.

**What it is.** One ``.roicat_classifier`` file holding everything needed to
classify ROIs: the trained ONNX classifier, the embedder network's ``model.py``
+ ``params.json`` + weights, the preprocessing configuration, and metadata
(label names, the network's identity, a SHA-256 per member, ROICaT version).
Build it once at the end of training; on any other machine,
``ClassifierPackage.load(path).predict(images, um_per_pixel)`` reproduces the
training-time inference path exactly.

``embedder/params.json`` is the bundle's own ``params.json`` **verbatim**, which
is more than the architecture: it also carries the hyperparameters and paths of
the training run that produced the network. That is kept deliberately — it is the
network's provenance, and the bundled ``make_model`` ignores what it does not
need.

**What it prevents.** Predictions are only meaningful if inference preprocesses
and embeds images the same way training did. Doing that by hand means
re-specifying the network URL, the ``forward_pass_version``, and the
preprocessing constants at inference time, and getting any one of them wrong
produces plausible-looking but wrong labels with no error. The packet removes
those degrees of freedom: they are recorded at pack time and replayed, and the
embedder's ``model.py`` travels with the weights so the packet keeps loading
across ROICaT versions.

Two things are deliberately kept apart:

* **Model properties** — recorded in the packet and replayed verbatim at predict
  time: the scale-factor constants, the network's input size, channel count, the
  architecture, and the weights.
* **Data properties** — supplied by the caller on every ``predict`` call:
  ``um_per_pixel``. The images being classified may come from different optics
  than the training images, and the scale normalization exists precisely to
  cancel that difference. Replaying the training value would reintroduce the bug
  it is there to prevent. ``um_per_pixel`` at training time is recorded only as
  provenance.

Only :class:`roicat.ROInet.ROInet_embedder` can be packed. ``_EMBEDDER_KINDS``
is a table so that future embedders can be added, but training-time containers
such as ``roicat.model_training.model.Simclr_Model`` are deliberately not
inference-facing: they know an architecture, not the preprocessing that must
precede it.

PCA is intentionally not included in this workflow.
"""

import datetime
import hashlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import warnings
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

import roicat
from roicat.ROInet import Preprocessor_ROI_images
from roicat.classification.classifier import Auto_LogisticRegression, ONNX_model_sklearnLogisticRegression


## Filename suffix for a packet. Named for what the file holds, so it is not
## confused with ROICaT's other outputs (``.richfile.zip``, ``.onnx``, ``.json``).
_SUFFIX = ".roicat_classifier"

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

## Written into every packet and required to match on load. There is no migration
## path: a mismatch means the file was written by a different ROICaT, and load()
## refuses it rather than guessing.
##
## Version 1 is the layout that predates ``preprocessing['preprocessor']``,
## per-member SHA-256, and ``embedder_identity``. Its files are NOT loadable here.
## The number must stay ahead of 1 for that refusal to happen at the version check
## rather than as a bare KeyError deeper in load(); dev-built version-1 packets do
## exist even though none were ever released.
_SCHEMA_VERSION = 2
_DEFAULT_BATCH_SIZE = 256

## How to rebuild an embedder's inner nn.Module from the ``model.py`` bundled in
## a packet. The chosen entry is *copied into the packet's metadata*, so a packet
## stays loadable even if this table later changes. A new embedder kind is added
## by giving it an entry here plus ``filepath_model_py`` / ``arch_kwargs`` /
## ``forward_pass_version`` attributes.
##   factory:           attribute of the bundled model.py to call.
##   attr_module:       attribute of the factory's return value holding the
##                      nn.Module whose state_dict was packed. "" means the
##                      returned object itself.
##   kwarg_fwd_version: name of the factory kwarg that selects the forward pass,
##                      or None if the factory takes it inside arch_kwargs.
_EMBEDDER_KINDS = {
    ## roicat.ROInet.ROInet_embedder: a network built from a distributed
    ## (OSF) bundle, whose model.py exposes make_model.
    "ROInet_embedder": {
        "factory": "make_model",
        "attr_module": "",
        "kwarg_fwd_version": "fwd_version",
    },
}

## Training-time containers that must never be packed, mapped to the reason.
## These are rejected with a specific message instead of the generic "unsupported
## kind", because reaching for them is a natural mistake: they hold the right
## weights but know nothing about preprocessing.
_EMBEDDER_KINDS_REJECTED = {
    "Simclr_Model": (
        "Simclr_Model is a training-time container and is not inference-facing. Pack the "
        "roicat.ROInet.ROInet_embedder you used to produce the latents instead; if you "
        "trained a new network, publish it as a bundle (model.py + params.json + weights) "
        "and load it with ROInet_embedder."
    ),
    "Simclr_Model_with_PCA": (
        "Simclr_Model_with_PCA is a training-time container and is not inference-facing. "
        "PCA is intentionally outside the ClassifierPackage workflow; pack the "
        "roicat.ROInet.ROInet_embedder you used to produce the latents instead."
    ),
}

## Known ROInet releases, keyed by the MD5 of the distributed ROInet.zip. Used only
## to put a human-readable name in a packet's metadata, so that reading a packet
## tells you which network it holds without comparing hex strings by hand. An
## unlisted bundle is recorded as 'unknown' with its MD5 — that is not an error,
## since users may train and distribute their own.
_RELEASES_ROINET = {
    "7a5fb8ad94b110037785a46b9463ea94": {
        "name": "tracking",
        "url": "https://osf.io/x3fd2/download",
        "note": "ROInet_tracking_20220527. Latent width: 256 at 'head', 128 at 'latent'.",
    },
    "357a8d9b630ec79f3e015d0056a4c2d5": {
        "name": "classification",
        "url": "https://osf.io/c8m3b/download",
        "note": "ROInet_classification_20220902. Latent width: 128 at either forward_pass_version.",
    },
}

## Deterministic ZIP defaults: epoch-zero mtime (zip-safe min is 1980), 0o644 perms,
## DEFLATE level 6.
_ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)
_ZIP_EXTERNAL_ATTR = 0o644 << 16
_ZIP_COMPRESS_LEVEL = 6


class PackageIntegrityError(RuntimeError):
    """Raised when a ``.roicat_classifier`` packet fails SHA-256 or schema validation on load."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _resolve_embedder(embedder) -> Tuple[torch.nn.Module, Dict[str, Any], str]:
    """
    Works out how to pack an embedder, and how a future ``load`` should rebuild it.

    Dispatch is on the class **name**, not ``isinstance``: an embedder that was
    itself rebuilt from a bundled ``model.py`` belongs to a dynamically imported
    module and is therefore never an instance of roicat's own classes.

    Args:
        embedder:
            An embedder whose class name is a key of ``_EMBEDDER_KINDS``.

    Returns:
        (Tuple[torch.nn.Module, Dict[str, Any], str]):
            module (torch.nn.Module):
                The module whose ``state_dict`` should be packed.
            spec (Dict[str, Any]):
                JSON-safe instructions for rebuilding, written to metadata.
            filepath_model_py (str):
                The ``model.py`` that defines the architecture.

    Raises:
        TypeError: If the embedder's class is not a packable kind.
        ValueError: If the embedder cannot describe its own architecture in a
            JSON-safe way.
    """
    name = type(embedder).__name__
    if name in _EMBEDDER_KINDS_REJECTED:
        raise TypeError(f"embedder of type {name!r} cannot be packed. {_EMBEDDER_KINDS_REJECTED[name]}")
    if name not in _EMBEDDER_KINDS:
        raise TypeError(
            f"embedder of type {name!r} cannot be packed. Supported: {sorted(_EMBEDDER_KINDS)}. "
            f"To add a kind, it must expose a JSON-safe 'arch_kwargs', a 'forward_pass_version', "
            f"and a 'filepath_model_py' whose model.py can rebuild it from those kwargs alone."
        )
    kind = _EMBEDDER_KINDS[name]
    arch_kwargs = embedder.arch_kwargs
    if not isinstance(arch_kwargs, dict):
        raise ValueError(f"embedder.arch_kwargs must be a dict, got {type(arch_kwargs).__name__}.")

    module = embedder.net
    ## The bundle's own model.py, not roicat's: it is what make_model came from.
    filepath_model_py = embedder.filepath_model_py
    fwd_version = embedder.forward_pass_version

    if not isinstance(module, torch.nn.Module):
        raise TypeError(f"Resolved embedder module must be an nn.Module, got {type(module).__name__}.")

    spec = {
        "class_name": name,  ## provenance only
        "factory": kind["factory"],
        "attr_module": kind["attr_module"],
        "kwarg_fwd_version": kind["kwarg_fwd_version"],
        "forward_pass_version": fwd_version,
    }
    return module, spec, str(filepath_model_py)


def _identify_embedder(embedder) -> Dict[str, Any]:
    """
    Records which network this embedder actually holds, for the packet's metadata.

    The identity is the MD5 of the distributed bundle **as found on disk**, not the
    ``download_hash`` the caller passed to ``ROInet_embedder``: that argument is
    optional, may be ``None``, and describes what the caller expected rather than
    what is loaded. Hashing the file is the only statement about the network that
    cannot be wrong.

    A missing bundle is recorded as ``None`` rather than raised on. The bundle is
    read at construction time and is not needed again, so a cleaned-up
    ``dir_networkFiles`` must not stop a valid embedder from being packed.

    Args:
        embedder:
            An ``ROInet_embedder``-shaped object.

    Returns:
        (Dict[str, Any]):
            identity (Dict[str, Any]):
                JSON-safe. ``release`` is the human-readable name when the MD5 is a
                known release, otherwise ``'unknown'``.
    """
    path_bundle = getattr(embedder, "_download_path_save", None)
    md5_bundle = None
    if path_bundle is not None and Path(path_bundle).exists():
        md5_bundle = roicat.helpers.hash_file(path=str(path_bundle), type_hash="MD5")

    release = _RELEASES_ROINET.get(md5_bundle)
    arch_kwargs = embedder.arch_kwargs

    ## Architecture fields that determine the latent width, recorded explicitly so a
    ## width disagreement can be explained without reopening the bundle.
    return {
        "release": release["name"] if release is not None else "unknown",
        "release_note": release["note"] if release is not None else None,
        "bundle_md5": md5_bundle,
        "download_url": getattr(embedder, "_download_url", None),
        "forward_pass_version": embedder.forward_pass_version,
        "torchvision_model": arch_kwargs.get("torchvision_model"),
        "pre_head_fc_sizes": arch_kwargs.get("pre_head_fc_sizes"),
        "post_head_fc_sizes": arch_kwargs.get("post_head_fc_sizes"),
    }


def _check_latents_provenance(embedder, classifier) -> None:
    """
    Checks that the classifier was fit on latents from *this* embedder.

    This is the only check that catches a wrong-release embedder whose latent width
    happens to match the classifier's, which a width comparison cannot see (the
    tracking release at ``forward_pass_version='latent'`` and the classification
    release both emit 128).

    The comparison is elementwise, so it is only meaningful when both arrays cover
    the same ROI images in the same order. That holds when one embedder embedded the
    training images and the classifier was fit on exactly those latents — the
    notebook workflow — and does not hold in general. So the check runs only when the
    shapes line up, and is silent otherwise. It never weakens the width check, which
    always runs.

    Args:
        embedder:
            The embedder being packed. Must expose ``.latents`` for the check to run.
        classifier (Auto_LogisticRegression):
            Must expose ``.X``, the latents it was fit on.

    Raises:
        ValueError: If both arrays are present and comparable but disagree.
    """
    latents = getattr(embedder, "latents", None)
    if latents is None:
        return
    latents = np.asarray(latents)
    X = np.asarray(classifier.X)
    if latents.shape != X.shape:
        return  ## Different images or ordering; the comparison would be meaningless.

    if not np.allclose(latents, X, rtol=1e-5, atol=1e-6):
        raise ValueError(
            f"The classifier was not fit on this embedder's latents. Both are "
            f"{X.shape} but they differ (max abs diff "
            f"{float(np.abs(latents - X).max()):.6g}). This embedder holds a different "
            f"network than the one used for training, or its latents were regenerated "
            f"with different settings. Pack the embedder you called generate_latents() "
            f"on before fitting the classifier."
        )


def _check_latent_dim(module_embedder, preprocessor, n_features_classifier: int) -> int:
    """
    Checks that the embedder's output width equals the classifier's input width,
    by actually running one image through the embedder.

    A forward pass is the only check that discriminates: reading the ONNX input
    width would compare the classifier against itself, since
    ``Auto_LogisticRegression.save_model`` derives the ONNX input type from the
    same ``classifier.X``. This is what catches the case of a classifier fit on
    one network's latents being packed with a different network.

    Args:
        module_embedder (torch.nn.Module):
            The module whose ``state_dict`` will be packed.
        preprocessor (Preprocessor_ROI_images):
            Used only for the shape of the network's input.
        n_features_classifier (int):
            ``classifier.X.shape[1]`` — the width the classifier was fit on.

    Returns:
        (int):
            latent_dim (int):
                The embedder's output width.

    Raises:
        ValueError: If the widths disagree.
        RuntimeError: If the embedder cannot accept the preprocessor's output.
    """
    ## One zeroed image of exactly the shape the preprocessor emits
    images = torch.zeros((1, preprocessor.n_channels_out, *preprocessor.img_size_out), dtype=torch.float32)
    device = next((p.device for p in module_embedder.parameters()), torch.device("cpu"))
    was_training = module_embedder.training
    module_embedder.eval()
    try:
        with torch.no_grad():
            latents = module_embedder(images.to(device))  # (1, latent_dim)
    except Exception as e:
        raise RuntimeError(
            f"The embedder could not accept the preprocessor's output shape "
            f"{tuple(images.shape)}. Either the preprocessor's img_size_out / n_channels_out "
            f"do not match this network, or the network is not the one used for embedding. "
            f"Original error: {e}"
        ) from e
    finally:
        module_embedder.train(was_training)

    latent_dim = int(latents.shape[-1])
    if latent_dim != n_features_classifier:
        raise ValueError(
            f"The embedder outputs {latent_dim} features but the classifier was fit on "
            f"{n_features_classifier}. These latents did not come from this embedder. The width "
            f"depends on BOTH the network release and forward_pass_version: the tracking "
            f"release (osf.io/x3fd2) gives 256 with 'head' and 128 with 'latent'; the "
            f"classification release (osf.io/c8m3b) gives 128 either way. Check that the "
            f"download_url and forward_pass_version being packed are the ones the classifier "
            f"was trained on."
        )
    return latent_dim


def _validate_classifier_package_inputs(classifier, label_names, preprocessor, size_images_in):
    """Cross-check init inputs against the contract; raise loudly on any violation."""
    if not isinstance(classifier, Auto_LogisticRegression):
        raise TypeError(f"classifier must be Auto_LogisticRegression, got {type(classifier).__name__}.")
    if getattr(classifier, "model_best", None) is None:
        raise RuntimeError("classifier.model_best is None — call classifier.fit() before packing.")

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

    if not isinstance(preprocessor, Preprocessor_ROI_images):
        raise TypeError(
            f"preprocessor must be a roicat.ROInet.Preprocessor_ROI_images, got "
            f"{type(preprocessor).__name__}. If you embedded with ROInet_embedder, pass "
            f"embedder=roinet and leave preprocessor=None to reuse roinet.preprocessor."
        )

    if len(tuple(size_images_in)) != 2:
        raise ValueError(f"size_images_in must be (height, width), got {tuple(size_images_in)}.")
    if not all(int(s) > 0 for s in size_images_in):
        raise ValueError(f"size_images_in must be two positive ints, got {tuple(size_images_in)}.")


class ClassifierPackage:
    """
    Self-contained inference artifact bundling embedder + ONNX classifier + label schema.

    Pack on the training machine, load and call ``predict`` anywhere — no other
    files needed beyond the single ``.roicat_classifier`` zip. This exists so that
    inference cannot silently disagree with training: the network, its
    ``forward_pass_version``, and the preprocessing configuration are recorded at
    pack time and replayed, instead of being re-specified by hand at inference
    time where a wrong value produces wrong labels without any error.

    After training a classifier (see the ``B2_classifier_train_interactive``
    notebook)::

        packet = roicat.classification.ClassifierPackage(
            classifier=autoclassifier,           ## fitted Auto_LogisticRegression
            embedder=roinet,                     ## the ROInet_embedder used for the latents
            label_names=['bad', 'good'],         ## in classifier.model_best.classes_ order
            size_images_in=data.ROI_images[0].shape[1:],  ## raw ROI image (height, width)
            um_per_pixel_training=data.um_per_pixel[0],   ## provenance only
        )
        packet.save('/path/to/mouse_1.roicat_classifier')

    Then, on any machine and on any dataset::

        packet = roicat.classification.ClassifierPackage.load('/path/to/mouse_1.roicat_classifier')
        label_ids, probs = packet.predict(
            roi_images=np.concatenate(data.ROI_images, axis=0),
            um_per_pixel=data.um_per_pixel[0],   ## resolution of THESE images
        )
        labels = [packet.label_names[i] for i in label_ids]

    ``um_per_pixel`` is a property of the data, so it is passed on every
    ``predict`` call rather than replayed from training — see the module
    docstring. ``size_images_in``, by contrast, must match training, because the
    scale normalization is defined relative to it.

    Every packet records which network it holds, readable without unzipping it::

        >>> packet.embedder_identity
        {'release': 'classification',
         'release_note': 'ROInet_classification_20220902. Latent width: 128 ...',
         'bundle_md5': '357a8d9b630ec79f3e015d0056a4c2d5',
         'download_url': 'https://osf.io/c8m3b/download',
         'forward_pass_version': 'head',
         'torchvision_model': 'convnext_tiny',
         'pre_head_fc_sizes': [256, 128],
         'post_head_fc_sizes': [128]}

    ``bundle_md5`` is hashed from the bundle on disk, so it states what was loaded
    rather than what the caller expected. ``release`` is ``'unknown'`` for a network
    that is not one of ROICaT's published releases, which is not an error.

    Args:
        classifier (Auto_LogisticRegression):
            Fitted instance.
        embedder:
            The embedder used to produce the latents the classifier was fit on.
            Must be one of the kinds in ``_EMBEDDER_KINDS`` — currently only
            ``roicat.ROInet.ROInet_embedder``.
        label_names (List[str]):
            Class names in ``classifier.model_best.classes_`` order.
        size_images_in (Tuple[int, int]):
            *(height, width)* of the raw ROI images used at training time, before
            any preprocessing. ``predict`` requires the same size.
        preprocessor (Optional[Preprocessor_ROI_images]):
            The preprocessing configuration used at training time. If ``None``,
            taken from ``embedder.preprocessor`` (set by
            ``ROInet_embedder.generate_dataloader`` / ``.embed``).
        um_per_pixel_training (Optional[float]):
            The resolution of the training images. **Recorded as provenance
            only** — never replayed at predict time, since the images being
            classified may have a different resolution.

    Raises:
        TypeError: Wrong type for any input.
        ValueError: Inconsistent label names or image sizes; an embedder whose
            latent width disagrees with the classifier's input width; or an
            embedder whose ``.latents`` are comparable to ``classifier.X`` but
            differ, meaning the classifier was trained on a different network.
        RuntimeError: Classifier not fitted.
    """

    def __init__(
        self,
        classifier: Auto_LogisticRegression,
        embedder,
        label_names: List[str],
        size_images_in: Tuple[int, int],
        preprocessor: Optional[Preprocessor_ROI_images] = None,
        um_per_pixel_training: Optional[float] = None,
    ):
        if preprocessor is None:
            preprocessor = getattr(embedder, "preprocessor", None)
            if preprocessor is None:
                raise ValueError(
                    "preprocessor is None and the embedder has no '.preprocessor' attribute. "
                    "Either call roinet.generate_dataloader(...) / roinet.embed(...) before "
                    "packing, or pass preprocessor=Preprocessor_ROI_images(...) explicitly."
                )
        _validate_classifier_package_inputs(
            classifier=classifier,
            label_names=label_names,
            preprocessor=preprocessor,
            size_images_in=size_images_in,
        )
        ## Fail at pack time, not at load time, if the embedder cannot be rebuilt
        self._module_embedder, self._spec_embedder, self._filepath_model_py = _resolve_embedder(embedder)
        self.latent_dim = _check_latent_dim(
            module_embedder=self._module_embedder,
            preprocessor=preprocessor,
            n_features_classifier=int(np.asarray(classifier.X).shape[1]),
        )
        ## Catches a wrong network whose width happens to match, which the width check
        ## above cannot. Runs only when the latents are comparable; see the docstring.
        _check_latents_provenance(embedder=embedder, classifier=classifier)
        ## Which network this actually is. Hashes the bundle once, here, rather than
        ## per save() call.
        self.embedder_identity = _identify_embedder(embedder)

        self.classifier = classifier
        self.embedder = embedder
        self.label_names = list(label_names)
        self.preprocessor = preprocessor
        self.preprocessing = {
            "preprocessor": preprocessor.to_dict(),
            "size_images_in": [int(s) for s in size_images_in],
            "um_per_pixel_training": None if um_per_pixel_training is None else float(um_per_pixel_training),
        }
        self._device = "cpu"
        self._onnx_model: Optional[ONNX_model_sklearnLogisticRegression] = None
        self._tmpdir: Optional[tempfile.TemporaryDirectory] = None
        self._module_name: Optional[str] = None

    # ------------------------------------------------------------------
    # save / load
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path], overwrite: bool = False) -> None:
        """
        Serialise the packet to a ``.roicat_classifier`` zip file.

        Byte-deterministic: identical inputs produce identical bytes (zeroed
        mtimes, fixed member order, DEFLATE level 6, 0o644 attrs). Atomic write
        via ``{path}.tmp`` + ``os.replace``.

        Args:
            path: Destination. ``.roicat_classifier`` auto-appended if missing.
                Parent directories created.
            overwrite: If False, ``FileExistsError`` when ``path`` exists.
        """
        path = Path(path)
        if path.suffix != _SUFFIX:
            path = path.with_suffix(path.suffix + _SUFFIX) if path.suffix else path.with_suffix(_SUFFIX)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not overwrite:
            raise FileExistsError(f"Packet file already exists: {path}")

        ## Gather member bytes
        bytes_classifier = self._pack_classifier()
        bytes_model_py, bytes_params, bytes_weights = self._pack_embedder()
        bytes_preprocessing = json.dumps(self.preprocessing, indent=2, sort_keys=True).encode()

        member_bytes = {
            _MEMBER_PREPROCESSING: bytes_preprocessing,
            _MEMBER_CLASSIFIER: bytes_classifier,
            _MEMBER_EMBEDDER_MODEL_PY: bytes_model_py,
            _MEMBER_EMBEDDER_PARAMS: bytes_params,
            _MEMBER_EMBEDDER_WEIGHTS: bytes_weights,
        }

        metadata = {
            "schema_version": _SCHEMA_VERSION,
            "roicat_version": roicat.__version__,
            "label_names": list(self.label_names),
            "latent_dim": self.latent_dim,
            "embedder": dict(self._spec_embedder),
            "embedder_identity": dict(self.embedder_identity),
            ## Keyed by member name so a future member is covered without another
            ## schema bump. Covers every member except metadata.json, which holds
            ## these values and so cannot hash itself.
            "sha256": {name: _sha256_bytes(data) for name, data in sorted(member_bytes.items())},
            "created_at_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        member_bytes[_MEMBER_METADATA] = json.dumps(metadata, indent=2, sort_keys=True).encode()

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
        Reconstruct a ``ClassifierPackage`` from a ``.roicat_classifier`` zip file.

        Args:
            path: Path to ``.roicat_classifier`` file.
            device: Torch device string for embedder weights (default ``"cpu"``).

        Returns:
            Loaded package; embedder is in eval mode on ``device``.

        Raises:
            FileNotFoundError: missing file.
            PackageIntegrityError: schema mismatch (including a packet written in
                the version-1 layout), a SHA-256 mismatch on any member, a member
                with no recorded SHA-256, or a latent dimension that disagrees with
                the classifier's input.
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
                    f"Unsupported schema_version {schema_version!r}; this code handles "
                    f"{_SCHEMA_VERSION}. Rebuild the packet with this version of ROICaT."
                )
            bytes_classifier = zf.read(_MEMBER_CLASSIFIER)
            bytes_model_py = zf.read(_MEMBER_EMBEDDER_MODEL_PY)
            bytes_params = zf.read(_MEMBER_EMBEDDER_PARAMS)
            bytes_weights = zf.read(_MEMBER_EMBEDDER_WEIGHTS)
            bytes_preprocessing = zf.read(_MEMBER_PREPROCESSING)

        ## SHA-256 integrity checks (raise, not warn). Every member except
        ## metadata.json is covered, including preprocessing.json — editing the
        ## preprocessing config changes every prediction, so it must be protected.
        bytes_by_member = {
            _MEMBER_PREPROCESSING: bytes_preprocessing,
            _MEMBER_CLASSIFIER: bytes_classifier,
            _MEMBER_EMBEDDER_MODEL_PY: bytes_model_py,
            _MEMBER_EMBEDDER_PARAMS: bytes_params,
            _MEMBER_EMBEDDER_WEIGHTS: bytes_weights,
        }
        sha256_expected = metadata["sha256"]
        missing = sorted(set(bytes_by_member) - set(sha256_expected))
        if missing:
            raise PackageIntegrityError(
                f"Packet metadata has no SHA-256 entry for {missing}. The file is "
                f"incomplete or was not written by this version of ROICaT."
            )
        for member_name, data in sorted(bytes_by_member.items()):
            _check_sha256(member_name, data, sha256_expected[member_name])

        label_names = list(metadata["label_names"])
        preprocessing = json.loads(bytes_preprocessing)
        preprocessor = Preprocessor_ROI_images.from_dict(config=preprocessing["preprocessor"])

        module_embedder, tmpdir, module_name = cls._unpack_embedder(
            bytes_model_py=bytes_model_py,
            bytes_params=bytes_params,
            bytes_weights=bytes_weights,
            spec=metadata["embedder"],
            device=device,
        )
        onnx_model = ONNX_model_sklearnLogisticRegression(path_or_bytes=bytes_classifier)

        ## The embedder/classifier pairing was verified by a forward pass at pack
        ## time (see _check_latent_dim). Here we only confirm that the rebuilt
        ## embedder still produces what the packet claims, and that the recorded
        ## width agrees with the classifier's input.
        latent_dim = metadata["latent_dim"]
        dim_onnx = onnx_model.session.get_inputs()[0].shape[-1]
        if isinstance(dim_onnx, int) and dim_onnx != latent_dim:
            raise PackageIntegrityError(
                f"latent_dim mismatch: metadata says {latent_dim}, but the ONNX classifier "
                f"expects {dim_onnx} features. The packet's embedder and classifier were not "
                f"built together."
            )
        try:
            _check_latent_dim(
                module_embedder=module_embedder,
                preprocessor=preprocessor,
                n_features_classifier=latent_dim,
            )
        except (ValueError, RuntimeError) as e:
            raise PackageIntegrityError(f"Rebuilt embedder does not match the packet: {e}") from e

        obj = object.__new__(cls)
        obj.classifier = None  # not reconstructed from ONNX; use _onnx_model directly
        obj.embedder = None  # only the inner nn.Module is packed, not the container
        obj._module_embedder = module_embedder
        obj._spec_embedder = dict(metadata["embedder"])
        obj._filepath_model_py = None
        obj.label_names = label_names
        obj.preprocessor = preprocessor
        obj.preprocessing = preprocessing
        obj.latent_dim = latent_dim
        ## Which network is inside. Present on both packed and loaded packets so it
        ## can be printed or logged at inference time.
        obj.embedder_identity = dict(metadata["embedder_identity"])
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
        um_per_pixel: float,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        device: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess ROI patches, embed them, and classify them.

        Preprocessing uses the configuration recorded in the packet, so it is by
        construction the same as at training time.

        Args:
            roi_images (np.ndarray):
                ``(n_rois, height, width)`` array of **raw** ROI images;
                ``(height, width)`` must equal the recorded
                ``size_images_in``.
            um_per_pixel (float):
                Micrometers per pixel of ``roi_images``. A property of *these*
                images — pass the resolution of the data being classified, which
                need not match the training data's resolution.
            batch_size (int):
                Number of ROIs preprocessed, embedded, and classified at a time.
            device (Optional[str]):
                Torch device override. Defaults to the device set at ``load()``.

        Returns:
            (Tuple[np.ndarray, np.ndarray]):
                label_ids (np.ndarray):
                    Predicted class indices into ``label_names``. Shape:
                    *(n_rois,)*. dtype: ``int64``.
                probs (np.ndarray):
                    Class probabilities. Shape: *(n_rois, n_classes)*. dtype:
                    ``float32``.

        Raises:
            ValueError: Bad ndim, height/width mismatch, or bad batch_size.
        """
        if roi_images.ndim != 3:
            raise ValueError(f"roi_images must be 3-D (n_rois, height, width), got ndim={roi_images.ndim}.")
        expected_hw = tuple(self.preprocessing["size_images_in"])
        if tuple(roi_images.shape[1:]) != expected_hw:
            raise ValueError(
                f"roi_images shape[1:]={tuple(roi_images.shape[1:])} != size_images_in={expected_hw}. "
                f"The packet was trained on {expected_hw} images; crop to the same size."
            )
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}.")
        if not self.preprocessor.scale_normalize:
            warnings.warn(
                "ROICaT WARNING: this packet was built with scale_normalize=False, so "
                "um_per_pixel has no effect and the images are used at their native scale."
            )

        n_classes = len(self.label_names)
        n_rois = roi_images.shape[0]
        if n_rois == 0:
            return np.empty((0,), dtype=np.int64), np.empty((0, n_classes), dtype=np.float32)

        dev = device if device is not None else self._device
        onnx_model = self._get_onnx_model()
        self._module_embedder.to(dev).eval()

        labels_chunks = []
        probs_chunks = []
        with torch.no_grad():
            for start in range(0, n_rois, batch_size):
                chunk = roi_images[start : start + batch_size]
                images = self.preprocessor.preprocess(
                    ROI_images=chunk,
                    um_per_pixel=um_per_pixel,
                )  # (n_chunk, n_channels_out, *img_size_out)
                latents = self._module_embedder(images.to(dev)).cpu().numpy().astype(np.float32, copy=False)  # (n_chunk, latent_dim)
                label_ids, probs = onnx_model(latents)
                labels_chunks.append(np.asarray(label_ids, dtype=np.int64))
                probs_chunks.append(np.asarray(probs, dtype=np.float32))
                del images, latents
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
        bytes_model_py = Path(self._filepath_model_py).read_bytes()

        ## arch_kwargs must be JSON-safe; json.dumps raises TypeError otherwise.
        ## Packed verbatim: the bundled model.py's factory is called with exactly
        ## these kwargs, so filtering keys could break reconstruction.
        bytes_params = json.dumps(self.embedder.arch_kwargs, indent=2, sort_keys=True).encode()

        ## state_dict from the inner nn.Module; tensors on CPU for portability.
        sd = self._module_embedder.state_dict()
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
        spec: Dict[str, Any],
        device: str = "cpu",
    ) -> Tuple[torch.nn.Module, tempfile.TemporaryDirectory, str]:
        """
        Reconstruct the embedder's nn.Module from packed bytes, following the
        recorded ``spec`` exactly rather than sniffing the bundled module.

        Args:
            bytes_model_py: The bundled ``model.py`` source.
            bytes_params: JSON ``arch_kwargs``.
            bytes_weights: ``torch.save``d state_dict.
            spec: The ``metadata["embedder"]`` block: ``factory``,
                ``attr_module``, ``kwarg_fwd_version``, ``forward_pass_version``.
            device: Torch device string.

        Returns:
            (Tuple[torch.nn.Module, tempfile.TemporaryDirectory, str]):
                (module, tmpdir, module_name) — caller owns the tempdir lifetime.

        Raises:
            PackageIntegrityError: If the bundled model.py lacks the recorded factory.
        """
        arch_kwargs = json.loads(bytes_params)
        sha = hashlib.sha256(bytes_model_py).hexdigest()[:12]
        module_name = f"roicat_pkg_{sha}"

        tmpdir = tempfile.TemporaryDirectory()
        filepath_model_py = os.path.join(tmpdir.name, "model.py")
        Path(filepath_model_py).write_bytes(bytes_model_py)

        spec_import = importlib.util.spec_from_file_location(module_name, filepath_model_py)
        module_model = importlib.util.module_from_spec(spec_import)
        ## Insert into sys.modules so any internal pickle/torch references resolve.
        sys.modules[module_name] = module_model
        try:
            spec_import.loader.exec_module(module_model)

            name_factory = spec["factory"]
            if not hasattr(module_model, name_factory):
                raise PackageIntegrityError(
                    f"Bundled model.py has no {name_factory!r}, which this packet's metadata "
                    f"says was used to build its embedder — packet incompatible."
                )
            kwargs = dict(arch_kwargs)
            if spec["kwarg_fwd_version"] is not None:
                kwargs[spec["kwarg_fwd_version"]] = spec["forward_pass_version"]
            obj = getattr(module_model, name_factory)(**kwargs)

            module = obj if spec["attr_module"] == "" else getattr(obj, spec["attr_module"])
            state_dict = torch.load(io.BytesIO(bytes_weights), map_location=device, weights_only=True)
            module.load_state_dict(state_dict)
            module.to(device).eval()
            for param in module.parameters():
                param.requires_grad = False
        except Exception:
            sys.modules.pop(module_name, None)
            tmpdir.cleanup()
            raise

        return module, tmpdir, module_name


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
