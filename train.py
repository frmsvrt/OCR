import os
import sys
import numpy as np

from sklearn.model_selection import train_test_split
import pandas as pd

from warpctc_pytorch import CTCLoss

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.functional as F
from torch.autograd import Variable as V
from torchvision import transforms

from helpers import Converter, Resize, ToTensorTarget, NormalizeTarget
from models import crnn, densenet
from configs import generator_cfg, trainer_cfg
from datareader import DataStream

t_cfg = trainer_cfg()
g_cfg = generator_cfg()

def _acc(preds, labels, lengths, total_size, converter):
    acc = 0
    preds = converter.decode_probs(preds)
    labels = converter.decode(labels, lengths)
    for pred, label in zip(preds, labels):
        if pred == label:
            acc += 1
    ret = acc / total_size
    return ret


def main():
    global t_cfg
    global g_cfg
    print(g_cfg.alph)
    transform = transforms.Compose([Resize((128, 32)),
                                    ToTensorTarget()])
                                    # NormalizeTarget([0.3956, 0.5763, 0.5616],
                                    #                 [0.1535, 0.1278, 0.1299])])

    # data preparation
    # create one fold split with 1/5 ration for validation
    data = pd.read_csv(t_cfg.DATANAME, sep=';', header=None)
    train_data, valid_data = train_test_split(data, test_size=.2, random_state=312)

    # train_data.to_csv('./train_data.csv')
    # valid_data.to_csv('./valid_data.csv')

    # define data flow for train and valid
    tds = DataStream(train_data, transform=transform)
    tdl = DataLoader(tds, batch_size=t_cfg.bs, shuffle=True, num_workers=23)
    vds = DataStream(valid_data, transform=transform)
    vdl = DataLoader(vds, batch_size=t_cfg.bs, shuffle=True, num_workers=23)
    converter = Converter(g_cfg.alph, ignore_case=False)

    # model/criterion define and optimizator selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = crnn.CRNN(3, len(g_cfg.alph) + 1, 256).to(device)
    # model = densenet.DenseNet(num_classes=len(g_cfg.alph)+1).to(device)
    # criterion = nn.CTCLoss()
    criterion = CTCLoss()
    optimizer = optim.Adam(model.parameters(),
                           lr=t_cfg.lr,
                           weight_decay=t_cfg.wl2,
                           )
    # lr_sched = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5,50],gamma=0.1)

    for epoch in range(t_cfg.epochs):
        lr_sched.step()
        loss, acc = do_epoch(tdl,
                             model,
                             optimizer,
                             criterion,
                             lr_sched,
                             device,
                             converter,
                             mode='train',
                             )
        print()
        if epoch > 40:
          if loss < t_cfg.valid_loss:
            t_cfg.valid_loss = loss
            acc = do_epoch(vdl,
                model,
                optimizer,
                criterion,
                lr_sched,
                device,
                converter,
                mode='valid',
                )
            torch.save(model, './model.pt')
            print('Validation acc: %.3f' % acc)
        print('Finished epoch: %d' % epoch, 'Loss: %.5f' % loss,
              'Acc: %.3f' % acc)

def do_epoch(dl,
             model,
             optimizer,
             criterion,
             lr_sched,
             device,
             converter,
             mode='train',
             ):
    global t_cfg
    global g_cfg
    L = []
    A = []
    if mode == 'train':
        model.train()
        # handle smaller batch
        try:
            for idx, sample in enumerate(dl):
                X = V(sample['img'].to(device))
                Y, Y_lengths = converter.encode(sample['label'])
                # Y = Y.to(device)

                optimizer.zero_grad()

                y_hat = model(X)

                preds_size = torch.IntTensor(t_cfg.bs).fill_(y_hat.shape[0])
                loss = criterion(y_hat, Y, preds_size, Y_lengths) / Y.size()[0]
                loss.backward()
                optimizer.step()

                l = loss.detach().cpu().numpy()

                L.append(l)

                # acc = _acc(y_hat, Y, Y_lengths, t_cfg.bs, converter)
                acc = 0
                A.append(acc)

                print('\r', 'Train step: %d' % (idx+1), '|', len(dl),
                      'Loss %.7f' % np.mean(L), end=' ')
        except:
            print('--')
        return np.mean(L), np.mean(A)

    elif mode == 'valid':
        with torch.no_grad():
            for idx, sample in enumerate(dl):
                X = V(sample['img'].to(device))
                Y, Y_lengths = converter.encode(sample['label'])
                # Y = Y.to(device)
                y_hat = model(X)

                preds_size = torch.IntTensor(t_cfg.bs).fill_(y_hat.shape[0])
                acc = _acc(y_hat, Y, Y_lengths, t_cfg.bs, converter)
                A.append(acc)

                print('\r', 'Val step: %d' % (idx + 1), '|', len(dl),
                      'Acc: %.3f' % acc, end=' ')
        return np.mean(A)

if __name__ == '__main__':
    main()
