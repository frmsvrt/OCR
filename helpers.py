import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import collections
import cv2
import numpy as np
import random

def sharping(im):
  im = np.array(im)
  gk = cv2.getGaussianKernel(21, 5)
  low_pass = cv2.filter2D(im, -1, gk)
  res = im - low_pass
  ret = im + res
  return res

def blur(im):
  im = np.array(im)
  return cv2.GaussianBlur(im, (5,5), 0)

def affineT(im):
  im = np.array(im)
  srcTri = np.array([[0,0],[im.shape[1]-1,0],[0,im.shape[0]-1]]).astype(np.float32)
  dstTri = np.array([[0,im.shape[1]*0.05],[im.shape[1]*0.99,im.shape[0]*0.05],
    [im.shape[1]*0.05,im.shape[0]*0.9]]).astype(np.float32)
  warp_mat = cv2.getAffineTransform(srcTri, dstTri)
  warp_dst = cv2.warpAffine(im, warp_mat, (im.shape[1], im.shape[0]),
      borderValue=(255,255,255))
  center = (warp_dst.shape[1]//2, warp_dst.shape[0]//2)
  angle = random.choice(np.linspace(-5, 5, 30))
  scale = 0.8
  rot_mat = cv2.getRotationMatrix2D(center, angle, scale)
  ret = cv2.warpAffine(warp_dst, rot_mat, (warp_dst.shape[1], warp_dst.shape[0]),
      borderValue=(255,255,255))
  return ret


class AttrDict(dict):
  __getattr__ = dict.__getitem__
  __setattr__ = dict.__setitem__


class Converter(object):
    def __init__(self, alphabet, ignore_case=True):
        self.alphabet = alphabet + '*'  # for `-1` index

        self.dict = {}
        for i, char in enumerate(alphabet):
            self.dict[char] = i + 1

    # TODO: classmethod decorator
    def encode(self, text):
        if isinstance(text, str):
            text = [self.dict[char] for char in text]
            length = [len(text)]
        elif isinstance(text, collections.Iterable):
            try:
              length = [len(str(s)) for s in text]
            except:
              print(text)
            text = ''.join([str(s) for s in text])
            text, _ = self.encode(text)
        return (torch.IntTensor(text), torch.IntTensor(length))

    def decode(self, t, length, raw=False):
        if length.numel() == 1:
            length = length[0]
            assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(), length)
            if raw:
                return ''.join([self.alphabet[i - 1] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i] - 1])
                return ''.join(char_list)
        else:
            # batch mode
            assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(
                        t[index:index + l], torch.IntTensor([l]), raw=raw))
                index += l
            return texts

    def decode_probs(self, probs):
        seq_len, batch_size = probs.shape[:2]
        lengths = torch.IntTensor(batch_size).fill_(seq_len)
        _, probs = probs.max(2)
        probs = probs.transpose(1, 0).contiguous().reshape(-1)
        preds = self.decode(probs, lengths)
        return preds


class ToTensorTarget(object):
    def __call__(self, sample):
        sat_img, label = sample['img'], sample['label']
        # sat_img = -1 + 2.0 * sat_img/255.0
        # print(sat_img.shape)
        return {'img': transforms.functional.to_tensor(sat_img.copy()),
                'label' : sample['label']}


class NormalizeTarget(transforms.Normalize):
    def __call__(self, sample):
        return {'img': transforms.functional.normalize(sample['img'], self.mean, self.std),
                'label': sample['label']}


class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        img = cv2.resize(sample['img'], self.size)
        return {'img' : img, 'label' : sample['label']}


class Sharpnes(object):
  def __call__(self, sample):
    p = np.random.rand()
    if p > 0.6:
      return {'img' : sharping(sample['img']), 'label' : sample['label']}
    else:
      return sample


class Blur(object):
  def __call__(self, sample):
    p = np.random.rand()
    if p > 0.6:
      return {'img' : blur(sample['img']), 'label' : sample['label']}
    else:
      return sample


class Affine(object):
  def __call__(self, sample):
    p = np.random.rand()
    if p > 0.66:
      return {'img' : affineT(sample['img']), 'label' : sample['label']}
    else:
      return sample

class Mono(object):
  def __call__(self, sample):
    im = sample['img']
    im = np.mean(np.array(im), axis=-1)
    im = np.expand_dims(im, axis=-1)
    print(im.shape)
    return {'img' : im, 'label': sample['label']}
