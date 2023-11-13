import numpy as np

import cv2
from albumentations.augmentations.geometric import functional as albu_geom
import albumentations as A

## transforms

class ReadImage:
  def __init__(self, mode=None, target_label='image_path', image_label='image'):
    self.read_mode = mode
    self.target_label = target_label
    self.image_label = image_label
  def __call__(self, **kwargs):
    kwargs[self.image_label] = cv2.imread(kwargs[self.target_label], self.read_mode)
    return kwargs

class RandomResizeTransform:
  def __init__(self, scale_range, target_label='scale'):
    self.scale_range = scale_range
    self.target_label = target_label
  def __call__(self, image, **kwargs):
    scale = np.exp(np.random.uniform(*np.log(self.scale_range)))
    kwargs['image'] = albu_geom.scale(image, scale)
    kwargs[self.target_label] = kwargs[self.target_label]*scale
    return kwargs
  def __repr__(self) -> str:
     return f'RandomResizeTransform({self.scale_range}, {self.target_label})'

class ResizeToTransform:
  def __init__(self, size, target_label='scale'):
    self.size = size
    self.target_label = target_label
  def __call__(self, image, **kwargs):
    scale = self.size/max(image.shape)
    kwargs['image'] = albu_geom.longest_max_size(image, max_size=self.size, interpolation=cv2.INTER_LINEAR)
    kwargs[self.target_label] = kwargs[self.target_label]*scale
    return kwargs
  def __repr__(self) -> str:
     return f'ResizeToTransform({self.size}, {self.target_label})'

class RotateTransform:
  def __init__(self, rotation_range, target_label='rotation', transform_kwargs={}):
    self.rotation_range = rotation_range
    self.target_label = target_label
    self.transform_kwargs = transform_kwargs
  def __call__(self, image, **kwargs):
    rotation = np.random.uniform(*self.rotation_range)
    kwargs['image'] = albu_geom.rotate(image, rotation, **self.transform_kwargs)
    kwargs[self.target_label] = (kwargs[self.target_label] + rotation + 90) % 180  - 90#TODO spr +/-
    return kwargs
  def __repr__(self) -> str:
     return f'RotateTransform({self.rotation_range}, {self.target_label}, {self.transform_kwargs})'

class HorizontalFlipTransform:
  def __init__(self, p=0.5, target_label='rotation', transform_kwargs={}):
    self.p = p
    self.target_label = target_label
  def __call__(self, image, **kwargs):
    apply_transform = np.random.rand() <= self.p
    if apply_transform:
      kwargs['image'] = np.ascontiguousarray(image[:, ::-1, ...])
      kwargs[self.target_label] = -kwargs[self.target_label]
    else:
      kwargs['image'] = image
    return kwargs
  def __repr__(self) -> str:
     return f'HorizontalFlipTransform({self.p}, {self.target_label})'
