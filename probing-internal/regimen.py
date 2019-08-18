"""Classes for training and running inference on probes."""
import os
import sys

from torch import optim
import torch
from tqdm import tqdm

class ProbeRegimen:
  """Basic regimen for training and running inference on probes.
  
  Tutorial help from:
  https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

  Attributes:
    optimizer: the optimizer used to train the probe
    scheduler: the scheduler used to set the optimizer base learning rate
  """

  def __init__(self, args):
    self.args = args
    self.max_epochs = args['probe_training']['epochs']
    self.params_path = os.path.join(args['reporting']['root'], args['probe']['params_path'])
    self.max_gradient_steps = args['probe_training']['max_gradient_steps'] if 'max_gradient_steps' in args['probe_training'] else sys.maxsize
    self.dev_eval_gradient_steps = args['probe_training']['eval_dev_every'] if 'eval_dev_every' in args['probe_training'] else -1

  def set_optimizer(self, probe):
    """Sets the optimizer and scheduler for the training regimen.
  
    Args:
      probe: the probe PyTorch model the optimizer should act on.
    """
    if 'weight_decay' in self.args['probe_training']:
      weight_decay = self.args['probe_training']['weight_decay']
    else:
      weight_decay = 0
    if 'scheduler_patience' in self.args['probe_training']:
      scheduler_patience = self.args['probe_training']['scheduler_patience']
    else:
      scheduler_patience = 0

    self.optimizer = optim.Adam(probe.parameters(), lr=0.001, weight_decay=weight_decay)
    self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5,patience=scheduler_patience)

  def train_until_convergence(self, probe, model, loss, train_dataset, dev_dataset):
    """ Trains a probe until a convergence criterion is met.

    Trains until loss on the development set does not improve by more than epsilon
    for 5 straight epochs.

    Writes parameters of the probe to disk, at the location specified by config.

    Args:
      probe: An instance of probe.Probe, transforming model outputs to predictions
      model: An instance of model.Model, transforming inputs to word reprs
      loss: An instance of loss.Loss, computing loss between predictions and labels
      train_dataset: a torch.DataLoader object for iterating through training data
      dev_dataset: a torch.DataLoader object for iterating through dev data
    """
    self.set_optimizer(probe)
    min_dev_loss = sys.maxsize
    min_dev_loss_epoch = -1
    gradient_steps = 0
    eval_dev_every = self.dev_eval_gradient_steps if self.dev_eval_gradient_steps != -1 else (len(train_dataset))
    eval_index = 0
    min_dev_loss_eval_index = -1
    for epoch_index in tqdm(range(self.max_epochs), desc='[training]'):
      epoch_train_loss = 0
      epoch_train_epoch_count = 0
      epoch_dev_epoch_count = 0
      epoch_train_loss_count = 0
      for batch in tqdm(train_dataset, desc='[training batch]'):
        probe.train()
        self.optimizer.zero_grad()
        observation_batch, label_batch, length_batch, _ = batch
        word_representations = model(observation_batch)
        predictions = probe(word_representations)
        batch_loss, count = loss(predictions, label_batch, length_batch)
        batch_loss.backward()
        epoch_train_loss += batch_loss.detach().cpu().numpy()*count.detach().cpu().numpy()
        epoch_train_epoch_count += 1
        epoch_train_loss_count += count.detach().cpu().numpy()
        self.optimizer.step()
        gradient_steps += 1
        if gradient_steps % eval_dev_every == 0:
          eval_index += 1
          if gradient_steps >= self.max_gradient_steps:
            tqdm.write('Hit max gradient steps; stopping')
            return
          epoch_dev_loss = 0
          epoch_dev_loss_count = 0
          for batch in tqdm(dev_dataset, desc='[dev batch]'):
            self.optimizer.zero_grad()
            probe.eval()
            observation_batch, label_batch, length_batch, _ = batch
            word_representations = model(observation_batch)
            predictions = probe(word_representations)
            batch_loss, count = loss(predictions, label_batch, length_batch)
            epoch_dev_loss += batch_loss.detach().cpu().numpy()*count.detach().cpu().numpy()
            epoch_dev_loss_count += count.detach().cpu().numpy()
            epoch_dev_epoch_count += 1
          self.scheduler.step(epoch_dev_loss)
          tqdm.write('[epoch {}] Train loss: {}, Dev loss: {}'.format(epoch_index,
              epoch_train_loss/epoch_train_loss_count, epoch_dev_loss/epoch_dev_loss_count))
          if epoch_dev_loss / epoch_dev_loss_count < min_dev_loss - 0.001:
            torch.save(probe.state_dict(), self.params_path)
            min_dev_loss = epoch_dev_loss / epoch_dev_loss_count
            min_dev_loss_epoch = epoch_index
            min_dev_loss_eval_index = eval_index
            tqdm.write('Saving probe parameters')
          elif min_dev_loss_eval_index < eval_index - 4:
            tqdm.write('Early stopping')
            return

  def predict(self, probe, model, dataset):
    """ Runs probe to compute predictions on a dataset.

    Args:
      probe: An instance of probe.Probe, transforming model outputs to predictions
      model: An instance of model.Model, transforming inputs to word reprs
      dataset: A pytorch.DataLoader object 

    Returns:
      A list of predictions for each batch in the batches yielded by the dataset
    """
    probe.eval()
    predictions_by_batch = []
    for batch in tqdm(dataset, desc='[predicting]'):
      observation_batch, label_batch, length_batch, _ = batch
      word_representations = model(observation_batch)
      predictions = probe(word_representations)
      predictions_by_batch.append(predictions.detach().cpu().numpy())
    return predictions_by_batch
