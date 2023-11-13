import numpy as np

import torch
import pytorch_lightning as pl
import timm
from hydra.utils import instantiate

## loss

class CosineLoss(torch.nn.Module):
  def __init__(self, p=1, degrees=False, scale=1):
    super().__init__()
    self.p = p
    self.degrees = degrees
    self.scale = scale
  def forward(self, x, y):
    if self.degrees:
      x = torch.deg2rad(x)
      y = torch.deg2rad(y)
    return torch.mean((1-torch.cos(x-y))**self.p) * self.scale
    
## model
class AngleParserLegacy(torch.nn.Module):
  def __init__(self, ndim, angle_range=180, angle_min=0):
    super().__init__()
    assert ndim in [1,2]
    self.ndim = ndim
    self.angle_range = angle_range
    self.angle_min = angle_min
  def forward(self, batch):
    if self.ndim == 1:
      return self.angle_range * (0.5 - torch.sigmoid(batch[:,0])) + self.angle_min
    elif self.ndim == 2:
      preds_x = 1. - 2*torch.sigmoid(batch[:,0])
      preds_y = 1. - 2*torch.sigmoid(batch[:,1])
      preds_direction = self.angle_range/360.*torch.rad2deg(torch.arctan2(preds_x, preds_y)) + self.angle_min
      return preds_direction
    raise ValueError

class AngleParser2d(torch.nn.Module):
  def __init__(self, angle_range=180, angle_min=0):
    super().__init__()
    self.angle_range = angle_range
    self.angle_min = angle_min
  def forward(self, batch):
    r = torch.linalg.norm(batch, dim=1)
    preds_sin = batch[:,0]/r
    preds_cos = batch[:,1]/r
    preds_direction = self.angle_range/360.*torch.rad2deg(torch.arctan2(preds_sin, preds_cos)) + self.angle_min
    return preds_direction

class AngleRegularizer(torch.nn.Module):
  def __init__(self, strength=1.0, scale=1.0, p=2):
    super().__init__()
    self.strength = strength
    self.scale = scale
    self.p = p
  def forward(self, batch):
    r = torch.linalg.norm(batch, dim=1)
    return self.strength * torch.norm(r - self.scale, p=self.p)

class AngleRegularizerLog(torch.nn.Module):
  def __init__(self, strength=1.0, scale=1.0, p=2):
    super().__init__()
    self.strength = strength
    self.scale = scale
    self.p = p
  def forward(self, batch):
    r = torch.linalg.norm(batch, dim=1)
    return self.strength * torch.norm(torch.log(r/self.scale), p=self.p)

class StripsModel(pl.LightningModule):
  def __init__(self, 
              model_name = 'resnet18',
               lr=0.001,
               optimizer_hparams=dict(),
               lr_hparams=dict(classname='MultiStepLR', kwargs=dict(milestones=[100, 150], gamma=0.1)),
               loss_hparams=dict(rotation_weight=10., white_frac_weight=50.),
               angle_hparams=dict(angle_range=180.),
               regularizer_hparams=None,
               sigmoid_smoother=10.
               ):
    super().__init__()
    # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
    self.save_hyperparameters()
    # Create model - implemented in non-abstract classes
    self.model =  timm.create_model(model_name, in_chans=1, num_classes=4) #2 + self.hparams.angle_hparams['ndim'])
    self.angle_parser = AngleParser2d(**self.hparams.angle_hparams)
    self.regularizer = self._get_regularizer(self.hparams.regularizer_hparams)
    self.losses = {
        'direction': CosineLoss(1., True),
        'period': torch.nn.functional.mse_loss,
        'white_fraction': torch.nn.functional.mse_loss
    }
    self.losses_weights = {
      'direction': self.hparams.loss_hparams['rotation_weight'],
      'period': 1,
      'white_fraction': self.hparams.loss_hparams['white_frac_weight'],
      'regularization': self.hparams.loss_hparams.get('regularization_weight', 0.)
    }
  
  def _get_regularizer(self, regularizer_params):
    if regularizer_params is None:
      return None
    else:
      return instantiate(regularizer_params)
    

  def forward(self, x, return_raw=False):
    """get predictions from image batch"""
    preds = self.model(x) # preds: logit angle_sin, logit angle_cos, period, logit white fraction or logit angle, period, logit white fraction
    preds_direction = self.angle_parser(preds)
    preds_period = preds[:,-2]
    preds_white_frac = torch.sigmoid(preds[:,-1]*self.hparams.sigmoid_smoother) #white fraction is between 0 and 1, so we take sigmoid fo this

    outputs = [preds_direction, preds_period, preds_white_frac]
    if return_raw:
      outputs.append(preds)
      
    return tuple(outputs)

  def configure_optimizers(self):
    # AdamW is Adam with a correct implementation of weight decay (see here
    # for details: https://arxiv.org/pdf/1711.05101.pdf)    
    optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, **self.hparams.optimizer_hparams)
    # scheduler = getattr(torch.optim.lr_scheduler, self.hparams.lr_hparams['classname'])(optimizer, **self.hparams.lr_hparams['kwargs'])
    scheduler = instantiate({**self.hparams.lr_hparams, '_partial_': True})(optimizer)
    return [optimizer], [scheduler]

  def process_batch_supervised(self, batch):
    """get predictions, losses and mean errors (MAE)"""

    # get predictions
    preds = {}
    preds['direction'], preds['period'], preds['white_fraction'], preds_raw = self.forward(batch['image'], return_raw=True) # preds: angle, period, white fraction, raw preds

    # calculate losses
    losses = {
        'direction': self.losses['direction'](2*batch['direction'], 2*preds['direction']),
        'period': self.losses['period'](batch['period'], preds['period']),
        'white_fraction': self.losses['white_fraction'](batch['white_fraction'], preds['white_fraction']),
    }
    if self.regularizer is not None:
      losses['regularization'] = self.regularizer(preds_raw[:,:2])
    
    losses['final'] = \
      losses['direction']*self.losses_weights['direction'] + \
      losses['period']*self.losses_weights['period'] + \
      losses['white_fraction']*self.losses_weights['white_fraction'] + \
      losses.get('regularization', 0.)*self.losses_weights.get('regularization', 0.)

    # calculate mean errors
    period_difference = np.mean(abs(
      batch['period'].detach().cpu().numpy() - \
      preds['period'].detach().cpu().numpy()
    ))

    a1 = batch['direction'].detach().cpu().numpy()
    a2 = preds['direction'].detach().cpu().numpy()
    angle_difference = np.mean(0.5*np.degrees(np.arccos(np.cos(2*np.radians(a2-a1)))))

    white_frac_difference = np.mean(abs(preds['white_fraction'].detach().cpu().numpy()-batch['white_fraction'].detach().cpu().numpy()))

    mae = {
      'period': period_difference,
      'direction': angle_difference,
      'white_fraction': white_frac_difference
    }

    return preds, losses, mae

  def log_all(self, losses, mae, prefix=''):
    self.log(f"{prefix}angle_loss", losses['direction'].item())
    self.log(f"{prefix}period_loss", losses['period'].item())
    self.log(f"{prefix}white_fraction_loss", losses['white_fraction'].item())
    self.log(f"{prefix}period_difference", mae['period'])
    self.log(f"{prefix}angle_difference", mae['direction'])
    self.log(f"{prefix}white_fraction_difference", mae['white_fraction'])
    self.log(f"{prefix}loss", losses['final'])
    if 'regularization' is losses:
      self.log(f"{prefix}regularization_loss", losses['regularization'].item())
  
  def training_step(self, batch, batch_idx):
    # "batch" is the output of the training data loader.
    preds, losses, mae = self.process_batch_supervised(batch)
    self.log_all(losses, mae, prefix='train_')

    return losses['final'] 
  
  def validation_step(self, batch, batch_idx):
    preds, losses, mae = self.process_batch_supervised(batch)
    self.log_all(losses, mae, prefix='val_')
  
  def test_step(self, batch, batch_idx):
    preds, losses, mae = self.process_batch_supervised(batch)
    self.log_all(losses, mae, prefix='test_')

# class StripsModel(StripsModelGeneral):
#   def __init__(self, model_name, *args, **kwargs):
#     super().__init__( *args, **kwargs)
#     self.model = timm.create_model(model_name, in_chans=1, num_classes=4)
#   def forward(self, x):
#     """get predictions from image batch"""
#     preds = self.model(x) # preds: logit angle_sin, logit angle_cos, period, logit white fraction
#     preds_sin = 1. - 2*torch.sigmoid(preds[:,0])
#     preds_cos = 1. - 2*torch.sigmoid(preds[:,1])
#     preds_direction = 0.5*torch.rad2deg(torch.arctan2(preds_sin, preds_cos))
#     preds_period = preds[:,2]
#     preds_white_frac = torch.sigmoid(preds[:,3]) #white fraction is between 0 and 1, so we take sigmoid fo this
#     return preds_direction, preds_period, preds_white_frac

# class StripsModelAngle1(StripsModelGeneral):
#   def __init__(self, model_name, *args, **kwargs):
#     super().__init__( *args, **kwargs)
#     self.model = timm.create_model(model_name, in_chans=1, num_classes=3)
#   def forward(self, x):
#     """get predictions from image batch"""
#     preds = self.model(x) # preds: logit angle_sin, logit angle
#     preds_direction = torch.pi * torch.sigmoid(preds[:,0])
#     preds_period = preds[:,1]
#     preds_white_frac = torch.sigmoid(preds[:,2]) #white fraction is between 0 and 1, so we take sigmoid fo this
#     return preds_direction, preds_period, preds_white_frac       
        