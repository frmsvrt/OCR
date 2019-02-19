import itertools
import os
from helpers import AttrDict
import numpy as np

FONTS = './misc/fonts/'
BGS = './misc/pictures/'

fonts = [os.path.join(FONTS, font) for font in\
         os.listdir(FONTS)]

bgs = [os.path.join(BGS, bg) for bg in\
         os.listdir(BGS)]

def LetterRange(start, end):
    return list(map(chr, range(ord(start), ord(end) + 1)))
VOCAB = LetterRange('a', 'z') + LetterRange('A', 'Z') + LetterRange('0', '9')

file_ids = [''.join(i) for i in itertools.product(VOCAB, repeat=4)]
file_index = {f: i for (i, f) in enumerate(file_ids)}

with open('./misc/dicts/vocabulary.txt', 'r') as f:
    d = f.readlines()

def generator_cfg():
    cfg = AttrDict()
    cfg.fonts = fonts
    # cfg.dict = lines
    cfg.bgs = bgs
    cfg.fs = [n for n in range(15,50,3)]
    cfg.sw = np.linspace(1,2.5,10)
    cfg.d = d
    cfg.fnames = file_ids
    _alph = ''
    for l in cfg.d:
        for c in l:
            if c not in _alph:
                _alph += c
    cfg.alph = ''.join(sorted(_alph))
    cfg.colors = 'black,yellow,red,green,magenta,blue'
    return cfg

def trainer_cfg():
    cfg = AttrDict()
    cfg.DATANAME = './data2/data.csv'
    cfg.bs = 256
    cfg.epochs = 60
    cfg.lr = 1e-3
    cfg.wl2 = 1e-7
    cfg.pivot = 15
    cfg.valid_loss = float('Inf')
    cfg.train_loss = float('Inf')
    cfg.val_acc = float('Inf')
    cfg.train_acc = float('Inf')
    return cfg
