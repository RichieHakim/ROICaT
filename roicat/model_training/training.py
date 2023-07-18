import numpy as np
import matplotlib.pyplot as plt
import gc

import torch
import torch.cuda
from torch.autograd import Variable

from sklearn.metrics import accuracy_score

import time

import util


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
    
    Returns:
        loss (float):
            Loss of the current batch
        pos_over_neg (float):
            Ratio of logits of positive to negative samples
    """

    double_sample_weights = torch.tile(sample_weights.reshape(-1), (2,))
    contrastive_matrix_sample_weights = torch.cat((torch.ones(1, device=X_train_batch.device), double_sample_weights), dim=0)
    
    optimizer.zero_grad()

    if inner_batch_size is None:
        features = model.forward(X_train_batch)
    else:
        features = torch.cat([model.forward(sub_batch) for sub_batch in make_batches(X_train_batch, batch_size=inner_batch_size)], dim=0)
    
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
    penalized_params = util.get_trainable_parameters(model)
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
                do_validation=False,
                validation_Object=None,
                verbose=False,
                verbose_update_period=100,
                log_function=print,
                
                X_val=None,
                y_val=None
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
        mode (str):
            'semi-supervised' or 'supervised'
        loss_rolling_train (list):
            List of losses for the current epoch
        device (str):
            Device to run the loss on
        do_validation (bool):
            Whether to do validation
            RH: NOT IMPLEMENTED YET. Keep False for now.
        validation_Object (torch.utils.data.DataLoader):
            Dataloader for the validation set
            RH: NOT IMPLEMENTED YET.
        loss_rolling_val (list):
            List of losses for the validation set
            RH: NOT IMPLEMENTED YET. 
             Keep [None or np.nan] for now.
        verbose (bool):
            Whether to print out the loss
        verbose_update_period (int):
            How often to print out the loss

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
            ) # Needs to take in weights
        loss_rolling_train.append(loss)
        # if False and do_validation:
        #     loss = validation_Object.get_predictions()
        #     loss_rolling_val.append(loss)
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