Classifier packets
==================

A **classifier packet** is a single file (``*.roicat_classifier``) holding a
trained ROI classifier together with the embedder and preprocessing settings it
was trained with. Loading a packet reproduces the training-time inference
pipeline on any machine. The class is
:class:`~roicat.classification.package.ClassifierPackage`.

Contents
########

- **classifier:** The trained model, stored as `ONNX <https://onnx.ai/>`__.
- **embedder:** The :mod:`roicat.ROInet` ``model.py``, ``params.json``, and
  weights.
- **preprocessing:** Raw ROI image size, ``forward_pass_version``, and the
  ``um_per_pixel`` of the training data.
- **label_names:** Class names corresponding to the classifier's output indices.
- **metadata:** The network's identity (see below), the classifier's class values
  in ``label_names`` order, a SHA-256 for every other member, a schema version,
  and the ROICaT version that wrote the file.

The embedder's ``model.py`` travels with its weights, so packets keep loading
after ROICaT's own network definitions change. File size is dominated by the
network weights: roughly 100 MB for the current ROInet releases.

``embedder/params.json`` is the network bundle's own ``params.json`` verbatim, so
it also carries the hyperparameters and paths of the training run that produced
the network. That is kept as provenance; the bundled ``make_model`` ignores the
keys it does not need.

Which network is inside
#######################

Every packet records the network it holds:

.. code-block:: python

   >>> packet.embedder_identity
   {'release': 'classification',
    'bundle_md5': '357a8d9b630ec79f3e015d0056a4c2d5',
    'download_url': 'https://osf.io/c8m3b/download',
    'forward_pass_version': 'head',
    'torchvision_model': 'convnext_tiny',
    'pre_head_fc_sizes': [256, 128],
    'post_head_fc_sizes': [128]}

``bundle_md5`` is hashed from the downloaded bundle, so it reports what was
loaded rather than what was requested. ``release`` is ``'unknown'`` for a network
that is not one of ROICaT's published releases, which is not an error.

A classifier is only valid with the network it was trained on, and the number of
features alone does not identify a network: the tracking release emits 256
features at ``forward_pass_version='head'`` but 128 at ``'latent'``, the same as
the classification release.

-------

Where packets fit
#################

Classification splits into work done once and work done repeatedly:

- **Once per classifier:** label ROIs (`A1
  <https://github.com/RichieHakim/ROICaT/blob/main/notebooks/classification/A1_classify_by_drawingSelection.ipynb>`__,
  `B1a
  <https://github.com/RichieHakim/ROICaT/blob/main/notebooks/classification/B1a_labeling_interactive.ipynb>`__,
  `B1b
  <https://github.com/RichieHakim/ROICaT/blob/main/notebooks/classification/B1b_labeling_drawingAndInteractive.ipynb>`__),
  then train (`B2
  <https://github.com/RichieHakim/ROICaT/blob/main/notebooks/classification/B2_classifier_train_interactive.ipynb>`__).
  ``B2`` writes the packet.
- **Once per new dataset:** run inference (`B3
  <https://github.com/RichieHakim/ROICaT/blob/main/notebooks/classification/B3_classifier_inference_interactive.ipynb>`__),
  often on a different machine or by a different person. ``B3`` reads the packet.

The packet is the handoff between the two. Tracking does not use packets.

Predictions are only valid if inference embeds images the same way training did.
Without a packet, that agreement is restated by hand at inference time: the
:class:`~roicat.ROInet.ROInet_embedder` arguments ``download_url``,
``download_hash``, and ``forward_pass_version``, plus the raw image size and the
scale-normalization constants. A mismatch in any of them gives wrong labels and
no error. The packet records these values when it is written and replays them
when it is loaded.

Only :class:`~roicat.ROInet.ROInet_embedder` can be packed. To deploy a network
you trained yourself, publish it as a bundle (``model.py`` + ``params.json`` +
weights) and load it through :class:`~roicat.ROInet.ROInet_embedder`.

An embedder driven with ``generate_dataloader(transforms=...)`` cannot be packed.
A custom chain replaces the packet's own preprocessing without being recorded
anywhere, so packing raises instead of silently recording the wrong configuration.
Configure preprocessing through
:class:`~roicat.ROInet.Preprocessor_ROI_images` arguments instead.

Recovering from the refusal takes three steps: re-run ``generate_dataloader()``
without the argument, then ``generate_latents()``, then refit the classifier.
Clearing the flag alone leaves the classifier fit on latents from the custom
chain, and the packet then builds with the wrong preprocessor recorded.

-------

Writing a packet
################

At the end of training. See the `B2 notebook
<https://github.com/RichieHakim/ROICaT/blob/main/notebooks/classification/B2_classifier_train_interactive.ipynb>`__
for the surrounding workflow, and the API reference for
:class:`~roicat.classification.package.ClassifierPackage`,
:class:`~roicat.classification.classifier.Auto_LogisticRegression`, and
:mod:`roicat.data_importing`.

.. code-block:: python

   import numpy as np
   import roicat

   ## `data` is a Data_roicat / Data_suite2p object with ROI_images and class labels
   roinet = roicat.ROInet.ROInet_embedder(
       device='cpu',
       dir_networkFiles='/tmp',
       download_url='https://osf.io/c8m3b/download',   ## the classification release
       download_hash='357a8d9b630ec79f3e015d0056a4c2d5',
       forward_pass_version='head',
   )
   roinet.generate_dataloader(
       ROI_images=data.ROI_images,
       um_per_pixel=data.um_per_pixel,
   )
   roinet.generate_latents()

   autoclassifier = roicat.classification.classifier.Auto_LogisticRegression(
       X=np.array(roinet.latents).astype(np.float32),
       y=np.concatenate(data.class_labels_index).astype(np.int64),
       params_LogisticRegression={'C': [1e-13, 1e3]},
   )
   autoclassifier.fit()

   packet = roicat.classification.ClassifierPackage(
       classifier=autoclassifier,
       embedder=roinet,
       label_names=[str(l) for l in autoclassifier.model_best.classes_],
       size_images_in=data.ROI_images[0].shape[1:],   ## raw ROI image (height, width)
       um_per_pixel_training=data.um_per_pixel[0],
   )
   packet.save('/path/to/mouse_1.roicat_classifier')

-------

Loading a packet
################

On any machine and any dataset. See the `B3 notebook
<https://github.com/RichieHakim/ROICaT/blob/main/notebooks/classification/B3_classifier_inference_interactive.ipynb>`__.

.. code-block:: python

   import numpy as np
   import roicat

   packet = roicat.classification.ClassifierPackage.load('/path/to/mouse_1.roicat_classifier')

   label_ids, probs = packet.predict(
       roi_images=np.concatenate(data.ROI_images, axis=0),  ## RAW images, (n_rois, height, width)
       um_per_pixel=data.um_per_pixel[0],                   ## resolution of THESE images
   )
   labels = [packet.label_names[i] for i in label_ids]

``label_ids`` are positions in ``label_names``, always in ``range(n_classes)``.
They are not the class values the classifier was fit on, which need not be
``0, 1, 2, ...`` — training data in which one class never appears gives
``classes_ == [0, 2]``. The packet records the class values so ``predict`` can
translate. ``probs`` columns are in the same order.

``um_per_pixel`` describes the data being classified and is required on every
call; it does not have to match the training data's resolution.
:meth:`~roicat.classification.package.ClassifierPackage.predict` takes a single
value, so sessions of differing resolution must be predicted one at a time. The
raw image height and width, however, must equal the packet's
``size_images_in``.

Loading raises :class:`roicat.classification.package.PackageIntegrityError` if
any recorded hash, the schema version, or the classifier's expected input width
disagrees with the file.

-------

Predicted labels can be applied to tracking results with the ``UCID`` helpers in
:ref:`roicat.util <roicat-util-module>`; see :doc:`inputsAndOutputs`.

API reference: :class:`roicat.classification.package.ClassifierPackage`,
:meth:`~roicat.classification.package.ClassifierPackage.save`,
:meth:`~roicat.classification.package.ClassifierPackage.load`,
:meth:`~roicat.classification.package.ClassifierPackage.predict`.
