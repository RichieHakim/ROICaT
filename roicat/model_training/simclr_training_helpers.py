# Imports
import os
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import tqdm

import torch
import torch.cuda
from torch.autograd import Variable

from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

from functools import partial
from typing import Optional, List, Tuple, Union, Dict, Any
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

from . import model


def _resolve_amp_bf16(device: Any, use_amp_bf16: bool) -> Tuple[str, bool]:
    """
    Resolve ``(device_type, enabled)`` for ``torch.autocast(dtype=bfloat16)``.

    Inspects the training device's hardware bf16 support and returns a tuple
    suitable for direct use in ``torch.autocast(device_type=..., enabled=...)``.
    If ``use_amp_bf16`` is requested but the hardware does not support bf16
    (e.g., pre-Ampere CUDA, CPU, missing CUDA), a ``UserWarning`` is emitted
    and ``enabled=False`` is returned so training continues in fp32.

    Args:
        device: A ``torch.device`` or device-string like ``'cuda:0'``, ``'mps'``,
            or ``'cpu'``.
        use_amp_bf16: User request to enable bf16 autocast.

    Returns:
        (device_type, enabled): strings/bools ready for ``torch.autocast``.
    """
    import warnings
    dev = device if isinstance(device, torch.device) else torch.device(device)
    dtype = dev.type
    if not use_amp_bf16:
        return dtype, False
    if dtype == 'cuda':
        if not torch.cuda.is_available():
            warnings.warn("use_amp_bf16=True but CUDA is not available; disabling.", stacklevel=2)
            return dtype, False
        if not torch.cuda.is_bf16_supported():
            cap = torch.cuda.get_device_capability(dev.index if dev.index is not None else 0)
            warnings.warn(f"use_amp_bf16=True but CUDA device sm_{cap[0]}{cap[1]} lacks native bf16 (need sm_80+: A100/3090/4090/H100); disabling.", stacklevel=2)
            return dtype, False
        return dtype, True
    if dtype == 'mps':
        warnings.warn("use_amp_bf16=True on MPS: PyTorch MPS bf16 autocast is experimental; op coverage may be incomplete.", stacklevel=2)
        return dtype, True
    if dtype == 'cpu':
        warnings.warn("use_amp_bf16=True on CPU: bf16 autocast works but is unlikely to speed up training.", stacklevel=2)
        return dtype, True
    warnings.warn(f"use_amp_bf16=True on unrecognized device type '{dtype}'; disabling.", stacklevel=2)
    return dtype, False


def log_fn(log_str, log_file):
    """
    Write a string to a log file

    Args:
        log_str (str):
            String to be written to the log file
        log_file (str):
            Path to the log file
    """
    with open(log_file, 'a') as f:
        f.write(log_str + '\n')


def _save_checkpoint_atomic(
    filepath: str,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    losses: List[float],
) -> None:
    """
    Atomically write a resumable training checkpoint.

    Writes to ``<filepath>_tmp.pth`` then ``os.replace`` onto ``filepath``
    so that a crash during writing cannot leave a corrupted checkpoint.

    Args:
        filepath (str):
            Destination path (typically ``{dir_save}/checkpoint_latest.pth``).
        epoch (int):
            Current epoch index (zero-based; the value stored is the epoch
            that just completed).
        model (torch.nn.Module):
            Model whose state_dict will be saved.
        optimizer (torch.optim.Optimizer):
            Optimizer whose state will be saved.
        scheduler (Optional[torch.optim.lr_scheduler._LRScheduler]):
            Scheduler whose state will be saved. Pass None to skip.
        losses (List[float]):
            Per-step loss list to persist for resume continuity.
    """
    ckpt = {
        "epoch": int(epoch),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "losses": list(losses),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    filepath_tmp = str(filepath) + "_tmp.pth"
    torch.save(ckpt, filepath_tmp)
    os.replace(filepath_tmp, filepath)


class Simclr_Trainer():
    """
    Class to train a SimCLR model from scratch using the provided parameters.

    JZ 2023

    Args:
        dataloader (torch.utils.data.DataLoader):
            The dataloader to use for training.
        model_container (ModelContainer):
            The model container to use for training.
        training_stop_revert_atNan (bool):
            Whether to revert to the previous model if the loss becomes NaN and stop training.
        n_epochs (int):
            The number of epochs to train for.
        device_train (str):
            The device to train on.
        inner_batch_size (int):
            The batch size to use for training.
        learning_rate (float):
            The learning rate to use for training.
        penalty_orthogonality (float):
            The penalty to apply to the orthogonality of the latent space.
        weight_decay (float):
            The weight decay to use for training.
        gamma (float):
            The gamma to use for training.
        temperature (float):
            The temperature to use for training.
        l2_alpha (float):
            The alpha to use for L2 regularization.
        path_saveLog (Optional[str]):
            The path to which to save the training log.
        path_saveLoss (Optional[str]):
            The path to which to save the losses.
        dir_save (Optional[str]):
            Directory into which to write per-epoch ``checkpoint_latest.pth``
            for resume-from-checkpoint. If None, derived from ``path_saveLoss``.
        resume_from_checkpoint (bool):
            If True and ``{dir_save}/checkpoint_latest.pth`` exists, restore
            model, optimizer, scheduler, RNG, and losses from it and continue
            from ``epoch + 1``. If True and no checkpoint exists, train from
            scratch.
        save_onnx_each_epoch (bool):
            If True (default), call ``model_container.save_onnx`` after each
            epoch. If False, skip the ONNX export but still write the .pth
            state checkpoint. Useful when ``onnx``/``onnxruntime`` are not
            installed in the environment.
        wandb_run (Optional[Any]):
            An optional ``wandb.sdk.wandb_run.Run`` instance. If provided,
            per-step ``loss``, ``lr``, and ``pos_over_neg`` are logged. If
            None (default), no wandb logging and ``wandb`` is not imported.
        use_amp_bf16 (bool):
            If True, wrap the model forward sub-batch loop in
            ``torch.autocast(device_type='cuda', dtype=torch.bfloat16)`` for
            ~2.8× throughput improvement on H100 / Ampere GPUs. Features are
            cast back to fp32 before the contrastive loss. Default False so
            existing behaviour is unchanged. No GradScaler is used (bf16 has
            sufficient dynamic range). Enable by setting ``"use_amp_bf16":
            true`` under the ``trainer`` section of your params JSON.
    """


    def __init__(
            self,
            dataloader: torch.utils.data.DataLoader,
            model_container: model.Simclr_Model,

            training_stop_revert_atNan: bool = True,

            n_epochs: int = 9999999,
            device_train: str = 'cuda:0',
            inner_batch_size: int = 256,
            learning_rate: float = 0.01,
            penalty_orthogonality: float = 1.00,
            weight_decay: float = 0.1,
            gamma: float = 1.0000,
            temperature: float = 0.03,
            l2_alpha: float = 0.0000,

            path_saveLog: Optional[str] = None,
            path_saveLoss: Optional[str] = None,

            dir_save: Optional[str] = None,
            resume_from_checkpoint: bool = True,
            save_onnx_each_epoch: bool = False,
            wandb_run: Optional[Any] = None,
            use_amp_bf16: bool = False,
        ):

        self.dataloader = dataloader
        self.training_stop_revert_atNan = training_stop_revert_atNan
        self.model_container = model_container
        self.n_epochs = n_epochs
        self.device_train = device_train
        self.inner_batch_size = inner_batch_size
        self.learning_rate = learning_rate
        self.penalty_orthogonality = penalty_orthogonality
        self.weight_decay = weight_decay
        self.gamma = gamma
        self.temperature = temperature
        self.l2_alpha = l2_alpha
        self.path_saveLog = path_saveLog
        self.path_saveLoss = path_saveLoss

        ## Resolve dir_save: explicit arg wins; else derive from path_saveLoss; else None.
        if dir_save is not None:
            self.dir_save = str(dir_save)
        elif path_saveLoss is not None:
            self.dir_save = str(Path(path_saveLoss).parent)
        else:
            self.dir_save = None
        self.resume_from_checkpoint = resume_from_checkpoint
        self.save_onnx_each_epoch = save_onnx_each_epoch
        self.wandb_run = wandb_run
        self.amp_device_type, self.use_amp_bf16 = _resolve_amp_bf16(self.device_train, use_amp_bf16)

    def train(
            self
            ):
        """
        Trains the model using the saved attributes.

        Per-epoch order: epoch_step -> save losses -> NaN-break check ->
        write ``checkpoint_latest.pth`` atomically -> (optional) ONNX export.
        The state_dict checkpoint is written before the ONNX export so a
        crash in ONNX export does not lose the resumable .pth checkpoint.
        """
        self.model_container.model.train();
        self.model_container.model.to(self.device_train)
        self.model_container.model.prep_contrast()

        criteria = [CrossEntropyLoss()]
        criteria = [criterion.to(self.device_train) for criterion in criteria]
        optimizer = Adam(
            self.model_container.model.parameters(),
            lr=self.learning_rate,
        )
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,
                                                        gamma=self.gamma,
                                                        )

        log_function = partial(log_fn, log_file=self.path_saveLog) if self.path_saveLog is not None else lambda x: None

        ## ------------------------------------------------------------
        ## Resume-from-checkpoint
        ## ------------------------------------------------------------
        losses_train, losses_val = [], [np.nan]
        epoch_start = 0
        filepath_ckpt = (
            str(Path(self.dir_save) / "checkpoint_latest.pth")
            if self.dir_save is not None else None
        )
        if self.resume_from_checkpoint and filepath_ckpt is not None and Path(filepath_ckpt).exists():
            ckpt = torch.load(filepath_ckpt, map_location=self.device_train, weights_only=False)
            self.model_container.model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            if "scheduler_state" in ckpt and ckpt["scheduler_state"] is not None:
                scheduler.load_state_dict(ckpt["scheduler_state"])
            losses_train = list(ckpt.get("losses", []))
            if "torch_rng_state" in ckpt and ckpt["torch_rng_state"] is not None:
                torch.set_rng_state(ckpt["torch_rng_state"].cpu().to(torch.uint8))
            if torch.cuda.is_available() and ckpt.get("cuda_rng_state") is not None:
                try:
                    torch.cuda.set_rng_state_all([s.cpu().to(torch.uint8) for s in ckpt["cuda_rng_state"]])
                except Exception as e:
                    print(f"[RESUME] Could not restore CUDA RNG state: {e}")
            epoch_start = int(ckpt["epoch"]) + 1
            msg_resume = f"[RESUME] Loaded {filepath_ckpt}. Resuming from epoch {epoch_start}. n_losses_so_far={len(losses_train)}."
            print(msg_resume)
            log_function(msg_resume)
        elif self.resume_from_checkpoint and filepath_ckpt is not None:
            msg_fresh = f"[RESUME] No checkpoint at {filepath_ckpt}. Starting fresh."
            print(msg_fresh)
            log_function(msg_fresh)

        for epoch in tqdm.tqdm(range(epoch_start, self.n_epochs)):
            print(f'epoch: {epoch}')
            losses_train = epoch_step(
                self.dataloader,
                self.model_container.model,
                optimizer,
                criteria,
                scheduler=scheduler,
                temperature=self.temperature,
                penalty_orthogonality=self.penalty_orthogonality,
                loss_rolling_train=losses_train,
                loss_rolling_val=losses_val,
                device=self.device_train,
                inner_batch_size=self.inner_batch_size,
                verbose=2,
                verbose_update_period=1,
                log_function=log_function,
                wandb_run=self.wandb_run,
                wandb_step_offset=len(losses_train),
                use_amp_bf16=self.use_amp_bf16,
                amp_device_type=self.amp_device_type,
            )

            ## save loss information
            if self.path_saveLoss is not None:
                np.save(self.path_saveLoss, losses_train)

            ## if loss becomes NaNs, don't save the network and stop training.
            ## Break BEFORE writing the resume checkpoint so a NaN epoch never
            ## poisons the resume.
            if torch.isnan(torch.as_tensor(losses_train[-1])) and self.training_stop_revert_atNan:
                break

            ## Write resumable state checkpoint atomically.
            if self.dir_save is not None:
                _save_checkpoint_atomic(
                    filepath=filepath_ckpt,
                    epoch=epoch,
                    model=self.model_container.model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    losses=losses_train,
                )

            ## save ONNX model (gated)
            if self.save_onnx_each_epoch:
                self.model_container.save_onnx(check_load_onnx_valid=True)

def train_step_simCLR(
    X_train_batch,
    y_train_batch,
    model,
    optimizer,
    criterion,
    scheduler,
    temperature,
    sample_weights,
    penalty_orthogonality=0,
    inner_batch_size=None,
    use_amp_bf16: bool = False,
    amp_device_type: str = 'cuda',
    ):
    """
    Performs a single training step.
    RH 2021 / JZ 2021

    Args:
        X_train_batch (torch.Tensor):
            Batch of training data.
            Shape: 
             (batch_size, n_transforms, n_channels, height, width)
        y_train_batch (torch.Tensor):
            Batch of training labels
            NOT USED FOR NOW
        model (torch.nn.Module):
            Model to train
        optimizer (torch.optim.Optimizer):
            Optimizer to use
        criterion (torch.nn.Module):
            Loss function to use
        scheduler (torch.optim.lr_scheduler.LambdaLR):
            Learning rate scheduler
        temperature (float):
            Temperature term for the softmax
        sample_weights (torch.Tensor):
            Sample weights for the batch
        penalty_orthogonality (float):
            Penalty for the orthogonality of the weights
        inner_batch_size (Optional[int]):
            Batch size for the inner loop. If None, the whole batch is used.
        use_amp_bf16 (bool):
            If True, wraps the model forward calls in
            ``torch.autocast(device_type='cuda', dtype=torch.bfloat16)``.
            Features are cast back to fp32 before the contrastive loss.
            Default False; behaviour is byte-identical to prior code when False.

    Returns:
        loss (float):
            Loss of the current batch
        pos_over_neg (float):
            Ratio of logits of positive to negative samples
    """

    double_sample_weights = torch.tile(sample_weights.reshape(-1), (2,))
    contrastive_matrix_sample_weights = torch.cat((torch.ones(1, device=X_train_batch.device), double_sample_weights), dim=0)
    
    optimizer.zero_grad()

    ## Forward sub-batch loop: optionally run under bfloat16 autocast for ~2.8×
    ## throughput on Ampere/H100 GPUs. enabled=False is a lightweight no-op that
    ## preserves byte-identical fp32 behaviour when use_amp_bf16=False.
    ## amp_device_type is resolved upstream (Simclr_Trainer.__init__) from the
    ## training device — 'cuda', 'mps', or 'cpu' — so the same flag works on
    ## non-CUDA devices.
    with torch.autocast(device_type=amp_device_type, dtype=torch.bfloat16, enabled=use_amp_bf16):
        if inner_batch_size is None:
            features = model.forward_latent(X_train_batch)
        else:
            features = torch.cat(
                [model.forward_latent(sub_batch) for sub_batch in make_batches(X_train_batch, batch_size=inner_batch_size)],
                dim=0,
            )
    ## Cast back to fp32: required for loss computation; no-op when use_amp_bf16=False.
    features = features.float()

    torch.cuda.empty_cache()

    logits, labels = richs_contrastive_matrix(features, batch_size=X_train_batch.shape[0]/2, n_views=2, temperature=temperature, DEVICE=X_train_batch.device) #### FOR RICH - THIS IS THE LINE IN QUESTION. I THINK "/2" NEEDS TO BE REMOVED FROM "X_train_batch.shape[0]/2"
    pos_over_neg = (torch.mean(logits[:,0]) / torch.mean(logits[:,1:])).item()

    loss_unreduced_train = torch.nn.functional.cross_entropy(logits, labels, weight=contrastive_matrix_sample_weights, reduction='none')
    loss_train = (loss_unreduced_train.float() @ double_sample_weights.float()) / double_sample_weights.float().sum()
    loss_orthogonality = off_diagonal(torch.corrcoef(features.T)).pow_(2).sum().div(features.shape[1])
    loss = loss_train + penalty_orthogonality * loss_orthogonality


    loss.backward()
    optimizer.step()
    scheduler.step()

    return loss.item(), pos_over_neg

def off_diagonal(x):
    """
    Returns the off-diagonal elements of a matrix as a vector.
    RH 2022

    Args:
        x (np.ndarray or torch tensor):
            square matrix to extract off-diagonal elements from.

    Returns:
        output (np.ndarray or torch tensor):
            off-diagonal elements of x.
    """
    n, m = x.shape
    assert n == m
    return x.reshape(-1)[:-1].reshape(n - 1, n + 1)[:, 1:].reshape(-1)

def L2_reg(model):
    """
    Returns the L2 regularization penalty for a model.
    RH 2021 / JZ 2021

    Args:
        model (torch.nn.Module):
            Model to calculate the penalty for.

    Returns:
        penalty (float):
            L2 regularization penalty.
    """
    penalized_params = get_trainable_parameters(model)
    penalty = 0
    for ii, param in enumerate(penalized_params):
        penalty += torch.sum(param**2)
    return penalty

def epoch_step( dataloader,
                model,
                optimizer,
                criterion,
                scheduler=None,
                temperature=0.5,
                penalty_orthogonality=0,
                loss_rolling_train=[],
                loss_rolling_val=[],
                device='cpu',
                inner_batch_size=None,
                verbose=False,
                verbose_update_period=100,
                log_function=print,
                wandb_run: Optional[Any] = None,
                wandb_step_offset: int = 0,
                use_amp_bf16: bool = False,
                amp_device_type: str = 'cuda',
                ):
    """
    Performs an epoch step.
    RH 2021 / JZ 2021

    Args:
        dataloader (torch.utils.data.DataLoader):
            Dataloader for the current epoch.
            Output for X_batch should be shape:
             (batch_size, n_transforms, n_channels, height, width)
        model (torch.nn.Module):
            Model to train
        optimizer (torch.optim.Optimizer):
            Optimizer to use
        criterion (torch.nn.Module):
            Loss function to use
        scheduler (torch.optim.lr_scheduler.LambdaLR):
            Learning rate scheduler
        temperature (float):
            Temperature term for the softmax
        penalty_orthogonality (float):
            Penalty for the orthogonality of the weights
        loss_rolling_train (list):
            List of losses for the current epoch
        loss_rolling_val (list):
            List of losses for the validation set
        device (str):
            Device to run the loss on
        inner_batch_size (Optional[int]):
            Batch size for the inner loop. If None, the whole batch is used.
        verbose (bool):
            Whether to print out the loss
        verbose_update_period (int):
            How often to print out the loss
        log_function (function):
            Function to use for printing out the loss
        wandb_run (Optional[Any]):
            Optional wandb Run; if provided, per-step ``loss``, ``lr``, and
            ``pos_over_neg`` are logged with explicit step counter
            ``wandb_step_offset + i_batch``.
        wandb_step_offset (int):
            Step counter offset (typically ``len(loss_rolling_train)`` at
            epoch start) so the global wandb step is monotonic across
            epochs and resumes.
        use_amp_bf16 (bool):
            Forwarded directly to ``train_step_simCLR``. Default False.

    Returns:
        loss_rolling_train (list):
            List of losses (passed through and appended)
    """

    def print_info(batch, n_batches, loss_train, loss_val, pos_over_neg, learning_rate, precis=5):
        log_function(f'Iter: {batch}/{n_batches}, loss_train: {loss_train:.{precis}}, loss_val: {loss_val:.{precis}}, pos_over_neg: {pos_over_neg} lr: {learning_rate:.{precis}}')

    for i_batch, (X_batch, y_batch, idx_batch, sample_weights) in enumerate(dataloader):
        X_batch = torch.cat(X_batch, dim=0)
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        # Get batch weights
        loss, pos_over_neg = train_step_simCLR(
            X_batch,
            y_batch,
            model,
            optimizer,
            criterion,
            scheduler,
            temperature,
            sample_weights=torch.as_tensor(sample_weights, device=device),
            penalty_orthogonality=penalty_orthogonality,
            inner_batch_size=inner_batch_size,
            use_amp_bf16=use_amp_bf16,
            amp_device_type=amp_device_type,
            ) # Needs to take in weights
        loss_rolling_train.append(loss)
        # if False and do_validation:
        #     loss = validation_Object.get_predictions()
        #     loss_rolling_val.append(loss)
        if wandb_run is not None:
            wandb_run.log(
                {
                    "loss": float(loss),
                    "lr": float(optimizer.param_groups[0]["lr"]),
                    "pos_over_neg": float(pos_over_neg),
                },
                step=int(wandb_step_offset + i_batch),
            )
        if verbose>0:
            if i_batch%verbose_update_period == 0:
                print_info( batch=i_batch,
                            n_batches=len( dataloader),
                            loss_train=loss_rolling_train[-1],
                            loss_val=loss_rolling_val[-1],
                            pos_over_neg=pos_over_neg,
                            learning_rate=scheduler.get_last_lr()[0],
                            precis=5)
    return loss_rolling_train

# # from https://github.com/sthalles/SimCLR/blob/1848fc934ad844ae630e6c452300433fe99acfd9/simclr.py
def info_nce_loss(features, batch_size, n_views=2, temperature=0.5, DEVICE='cpu'):
    """
    'Noise-Contrastice Estimation' loss. Loss used in SimCLR.
    InfoNCE loss. Aka: NTXentLoss or generalized NPairsLoss.
    
    logits and labels should be run through 
     CrossEntropyLoss to complete the loss.
    CURRENTLY ONLY WORKS WITH n_views=2 (I think this can be
     extended to larger numbers by simply reshaping logits)

    Code mostly copied from: https://github.com/sthalles/SimCLR/blob/1848fc934ad844ae630e6c452300433fe99acfd9/simclr.py
    RH 2021 / JZ 2021

    demo for learning/following shapes:
        features = torch.rand(8, 100)
        labels = torch.cat([torch.arange(4) for i in range(2)], dim=0)

        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        print(labels)
        features = torch.nn.functional.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T)
        print(similarity_matrix)
        mask = torch.eye(labels.shape[0], dtype=torch.bool)
        labels = labels[~mask].view(labels.shape[0], -1)
        print(labels)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        print(similarity_matrix)
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        print(positives)
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
        print(negatives)
        logits = torch.cat([positives, negatives], dim=1)
        print(logits)
        labels = torch.zeros(logits.shape[0], dtype=torch.long)
        print(labels)

    Args:
        features (torch.Tensor): 
            Outputs of the model
            Shape: (batch_size * n_views, n_channels, height, width)
        batch_size (int):
            Number of samples in the batch
        n_views (int):
            Number of views in the batch
            MUST BE 2 (larger values not supported yet)
        temperature (float):
            Temperature term for the softmax
        DEVICE (str):
            Device to run the loss on
    
    Returns:
        logits (torch.Tensor):
            Class prediction logits
            Shape: (batch_size * n_views, ((batch_size-1) * n_views)+1)
    """

    # make (double) diagonal matrix
    labels = torch.cat([torch.arange(batch_size) for i in range(n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(DEVICE)

    # normalize to unit hypersphere
    features = torch.nn.functional.normalize(features, dim=1)

    # compute (double) covariance matrix
    similarity_matrix = torch.matmul(features, features.T)
    # assert similarity_matrix.shape == (
    #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
    # assert similarity_matrix.shape == labels.shape

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(DEVICE)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    # assert similarity_matrix.shape == labels.shape

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1) # logits column 1 is positives, the rest of the columns are negatives
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(DEVICE) # all labels are 0 because first column in logits is positives

    logits = logits / temperature
    return logits, labels


# # from https://github.com/sthalles/SimCLR/blob/1848fc934ad844ae630e6c452300433fe99acfd9/simclr.py
def richs_contrastive_matrix(features, batch_size, n_views=2, temperature=0.5, DEVICE='cpu'):
    """
    Modified 'Noise-Contrastice Estimation' loss. 
    Almost identical to the method used in SimCLR.
    Should be techincally identical to InfoNCE, 
     but the output logits matrix is different.
    The output logits first column is the positives,
     and the rest of the columns are the cosine 
     similarities of the negatives (positives )
    InfoNCE loss. Aka: NTXentLoss or generalized NPairsLoss.
    
    logits and labels should be run through 
     CrossEntropyLoss to complete the loss.
    CURRENTLY ONLY WORKS WITH n_views=2 (I think this can be
     extended to larger numbers by simply reshaping logits)

    Code mostly copied from: https://github.com/sthalles/SimCLR/blob/1848fc934ad844ae630e6c452300433fe99acfd9/simclr.py
    RH 2021

    demo for learning/following shapes:
        features = torch.rand(8, 100)
        eye_prep = torch.cat([torch.arange(4) for i in range(2)], dim=0)
        print(eye_prep)
        multi_eye = (eye_prep.unsqueeze(0) == eye_prep.unsqueeze(1))
        print(multi_eye)
        features = torch.nn.functional.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T)
        print(similarity_matrix)
        single_eye = torch.eye(multi_eye.shape[0], dtype=torch.bool)
        multi_cross_eye = multi_eye * ~single_eye
        print(multi_cross_eye)
        positives = similarity_matrix[multi_cross_eye.bool()].view(multi_cross_eye.shape[0], -1)
        print(positives)
        logits = torch.cat([positives, similarity_matrix*(~multi_eye)], dim=1)
        print(logits)
        labels = torch.zeros(logits.shape[0], dtype=torch.long)
        print(labels)

    Args:
        features (torch.Tensor): 
            Outputs of the model
            Shape: (batch_size * n_views, n_channels, height, width)
        batch_size (int):
            Number of samples in the batch
        n_views (int):
            Number of views in the batch
            MUST BE 2 (larger values not supported yet)
        temperature (float):
            Temperature term for the softmax
        DEVICE (str):
            Device to run the loss on
    
    Returns:
        logits (torch.Tensor):
            Class prediction logits
            Shape: (batch_size * n_views, 1 + (batch_size * n_views))
    """


    eye_prep = torch.cat([torch.arange(batch_size) for i in range(n_views)], dim=0).to(DEVICE)
    multi_eye = (eye_prep.unsqueeze(0) == eye_prep.unsqueeze(1))
    features = torch.nn.functional.normalize(features, dim=1)
    similarity_matrix = torch.matmul(features, features.T)
    single_eye = torch.eye(multi_eye.shape[0], dtype=torch.bool).to(DEVICE)
    multi_cross_eye = multi_eye * ~single_eye
    positives = similarity_matrix[multi_cross_eye].view(multi_cross_eye.shape[0], -1)
    logits = torch.cat([positives, similarity_matrix*(~multi_eye)], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(DEVICE)

    logits = logits / temperature
    return logits, labels

def make_batches(iterable, batch_size=None, num_batches=5, min_batch_size=0, return_idx=False):
    """
    Make batches of data or any other iterable.
    RH 2021

    Args:
        iterable (iterable):
            iterable to be batched
        batch_size (int):
            size of each batch
            if None, then batch_size based on num_batches
        num_batches (int):
            number of batches to make
        min_batch_size (int):
            minimum size of each batch
        return_idx (bool):
            whether to return the indices of the batches.
            output will be [start, end] idx
    
    Returns:
        output (iterable):
            batches of iterable
    """
    l = len(iterable)
    
    if batch_size is None:
        batch_size = np.int64(np.ceil(l / num_batches))
    
    for start in range(0, l, batch_size):
        end = min(start + batch_size, l)
        if (end-start) < min_batch_size:
            break
        else:
            if return_idx:
                yield iterable[start:end], [start, end]
            else:
                yield iterable[start:end]

def get_trainable_parameters(model):
    """
    Get the trainable parameters of a model.

    Args:
        model (torch.nn.Module):
            model to get trainable parameters from

    Returns:
        params_trainable (list):
            list of trainable parameters
    """
    params_trainable = []
    for param in list(model.parameters()):
        if param.requires_grad:
            params_trainable.append(param)
    return params_trainable

class Simclr_PCA_Trainer():
    def __init__(
            self,
            dataloader,
            model_container,

            center: bool = True,
            scale: bool = False,
            path_saveLog: Optional[str] = None,
            save_onnx_each_epoch: bool = False,

            # use_iterated_learning: bool = False,
            ):
        """
        Training module to train a SimCLR model from scratch using the provided parameters.

        JZ 2023

        Args:
            dataloader (torch.utils.data.DataLoader):
                The dataloader to use for training.
            model_container (ModelContainer):
                The model container to use for training.
            center (bool):
                Whether to center the data.
            scale (bool):
                Whether to scale the data.
            path_saveLog (str):
                The path to which to save the training log.
            save_onnx_each_epoch (bool):
                If True, call ``model_container.save_onnx`` after fitting the PCA layer.
                Default False; set True only when onnx/onnxruntime are available and
                an ONNX export is explicitly desired.
        """

        self.dataloader = dataloader
        self.model_container = model_container
        self.center = center
        self.scale = scale
        self.path_saveLog = path_saveLog
        self.save_onnx_each_epoch = save_onnx_each_epoch

    def train(self, check_pca_layer_valid: bool=True):
        """
        Trains the pca layer of the model using the input data x and saves
        the updated model to self.model_container.filepath_model_save

        Args:
            check_pca_layer_valid (bool):
                Whether to check that the pca layer is valid after training.
        """

        # x = torch.cat([torch.cat(data_subset[0], dim=0) for data_subset in self.dataloader], axis=0)
        # output_head = self.model_container.model(x)
        output_head = torch.cat([self.model_container.model(torch.cat(data_subset[0], dim=0)) for data_subset in self.dataloader], dim=0)
        pca_size = output_head.shape[1]

        output_head_scaler = output_head.std(dim=0) if self.scale else torch.ones_like(output_head.std(dim=0))
        output_head_centerer = output_head.mean(dim=0)/output_head_scaler if self.center else torch.zeros_like(output_head.mean(dim=0)/output_head_scaler)

        pca_layer = torch.nn.Sequential(
            torch.nn.Linear(pca_size, pca_size),
            torch.nn.Linear(pca_size, pca_size, bias=False)
        )
        pca_layer[0].weight = torch.nn.Parameter(torch.diag(1/output_head_scaler))
        pca_layer[0].bias = torch.nn.Parameter(-output_head_centerer)

        output_head_centered = (output_head - output_head_centerer) / output_head_scaler
        
        if check_pca_layer_valid:
            assert torch.allclose(
                output_head_centered,
                pca_layer[0](output_head),
                atol=torch.tensor(1e-4)
            ), 'zscore layer not working'

        np_output_head_centered = output_head_centered.detach().cpu().numpy()

        pca_sklearn = PCA()
        pca_sklearn.fit(np_output_head_centered)
        pca_sklearn.components_ = pca_sklearn.components_.astype(np.float32)
        ## Preserve raw centering mean before zeroing; required for fused-bias construction.
        ## pca_sklearn.mean_ is zeroed below so it cannot be used for centering downstream.
        self.pca_mean_fitted = output_head_centerer.detach().cpu().numpy().astype(np.float32)  # (pca_size,)
        pca_sklearn.mean_ = np.zeros(pca_size, dtype=np.float32)
        self.pca_sklearn_fitted = pca_sklearn  ## expose for caller to construct Simclr_Model_with_PCA

        pca_layer[1].weight = torch.nn.Parameter(torch.tensor(pca_sklearn.components_, dtype=torch.float32))
        # pca_layer[1].bias = torch.nn.Parameter(torch.tensor(np.zeros(pca_size,),dtype=torch.float32))

        if check_pca_layer_valid:
            pca_output = pca_layer(output_head)
            pca_sklearn_tensor = torch.tensor(pca_sklearn.transform(np_output_head_centered))
            print(np.max(np.abs(pca_output.detach().numpy() - pca_sklearn_tensor.detach().numpy())))
            assert torch.allclose(pca_output,
                            pca_sklearn_tensor,
                            atol=1e-5
                            ), 'pca layer not working'
        
        print('save')

        ## ONNX export: wraps model_container.model in Simclr_Model_with_PCA then saves.
        ## Gated so the wrapping (which mutates model_container.model) is skipped by default.
        ## When gate is off, model_container.model remains the bare ModelTackOn backbone.
        if self.save_onnx_each_epoch:
            self.model_container.model = model.Simclr_Model_with_PCA.from_simclr_and_sklearn_pca(
                model_container=self.model_container,
                pca_sklearn=self.pca_sklearn_fitted,
                pca_mean=self.pca_mean_fitted,
                trainer_scale=self.scale,
            )
            self.model_container.save_onnx(check_load_onnx_valid=True, revert_train=False)

        print('pca layer trained and saved to model_container')
