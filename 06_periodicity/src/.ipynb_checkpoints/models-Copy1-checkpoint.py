import numpy as np

import torch
import pytorch_lightning as pl
import timm

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

class StripsModel(pl.LightningModule):
  def __init__(self, 
               model_name = 'resnet18',
               optimizer_hparams=dict(lr=0.001),
               lr_hparams=dict(classname='MultiStepLR', kwargs=dict(milestones=[100, 150], gamma=0.1)), 
               loss_hparams=dict(rotation_weight=10., white_frac_weight=50.),
               ):
    super().__init__()
    # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
    self.save_hyperparameters()
    # Create model
    self.model =  timm.create_model(model_name, in_chans=1, num_classes=4) #sin, cos, period, white frac
    # classifier = list(self.model.children())[-1]
    # try:
    #     torch.nn.init.zeros_(classifier.weight)
    #     torch.nn.init.constant_(classifier.bias, 25)
    # except:
    #     pass
    self.losses = {
        'direction': CosineLoss(1., True),
        'period': torch.nn.functional.mse_loss,
        'white_fraction': torch.nn.functional.mse_loss
    }
    self.losses_weights = {
      'direction': self.hparams.loss_hparams['rotation_weight'],
      'period': 1,
      'white_fraction': self.hparams.loss_hparams['white_frac_weight'],
    }
    # Create loss module
    # self.loss_module = torch.nn.MSELoss(reduction='none')
  def configure_optimizers(self):
    # AdamW is Adam with a correct implementation of weight decay (see here
    # for details: https://arxiv.org/pdf/1711.05101.pdf)
    optimizer = torch.optim.AdamW(self.parameters(), **self.hparams.optimizer_hparams)
    scheduler = getattr(torch.optim.lr_scheduler, self.hparams.lr_hparams['classname'])(optimizer, **self.hparams.lr_hparams['kwargs'])
    return [optimizer], [scheduler]

  def forward(self, x):
    """get predictions from image batch"""
    preds = self.model(x) # preds: logit angle_sin, logit angle_cos, period, logit white fraction
    preds_sin = 1. - 2*torch.sigmoid(preds[:,0])
    preds_cos = 1. - 2*torch.sigmoid(preds[:,1])
    preds_direction = 0.5*torch.rad2deg(torch.arctan2(preds_sin, preds_cos))
    preds_period = preds[:,2]
    preds_white_frac = torch.sigmoid(preds[:,3]) #white fraction is between 0 and 1, so we take sigmoid fo this
    return preds_direction, preds_period, preds_white_frac

  def process_batch_supervised(self, batch):
    """get predictions, losses and mean errors (MAE)"""

    # get predictions
    preds = {}
    preds['direction'], preds['period'], preds['white_fraction'] = self.forward(batch['image']) # preds: angle, period, white fraction

    # calculate losses
    losses = {
        'direction': self.losses['direction'](2*batch['direction'], 2*preds['direction']),
        'period': self.losses['period'](batch['period'], preds['period']),
        'white_fraction': self.losses['white_fraction'](batch['white_fraction'], preds['white_fraction']),
    }
    losses['final'] = \
      losses['direction']*self.losses_weights['direction'] + \
      losses['period']*self.losses_weights['period'] + \
      losses['white_fraction']*self.losses_weights['white_fraction']

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