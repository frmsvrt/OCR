import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import collections
import cv2

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
            length = [len(s) for s in text]
            text = ''.join(text)
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
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        sat_img, label = sample['img'], sample['label']
        return {'img': transforms.functional.to_tensor(sat_img.copy()),
                'label' : sample['label']}


class NormalizeTarget(transforms.Normalize):
    """Normalize a tensor and also return the target"""

    def __call__(self, sample):
        return {'img': transforms.functional.normalize(sample['img'], self.mean, self.std),
                'label': sample['label']}

class Resize(object):
    """Resize."""
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        img = cv2.resize(sample['img'], self.size)
        return {'img' : img, 'label' : sample['label']}
