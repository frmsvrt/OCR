import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.functional as F
from torch.autograd import Variable as V
from torchvision import transforms

from helpers import Converter, Resize, ToTensorTarget, NormalizeTarget
from models import crnn
from configs import generator_cfg, trainer_cfg
from datareader import DataStream

t_cfg = trainer_cfg()
g_cfg = generator_cfg()

def main():
    global t_cfg
    global g_cfg
    transform = transforms.Compose([Resize((204, 32)),
                                    ToTensorTarget(),
                                    NormalizeTarget([0.3956, 0.5763, 0.5616],
                                                    [0.1535, 0.1278, 0.1299])])

    ds = DataStream(t_cfg.DATANAME, transform=transform)
    dl = DataLoader(ds, batch_size=t_cfg.bs, shuffle=True)
    converter = Converter(g_cfg.alph)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = crnn.CRNN(3, len(g_cfg.alph) + 1, 256).to(device)
    criterion = nn.CTCLoss()
    optimizer = optim.Adam(model.parameters(),
                           lr=t_cfg.lr,
                           weight_decay=t_cfg.wl2,
                           )
    lr_sched = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    for epoch in range(t_cfg.epochs):
        loss = do_epoch(dl,
                        model,
                        optimizer,
                        criterion,
                        lr_sched,
                        device,
                        converter,
                        mode='train',
                        )

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
    if mode == 'train':
        model.train()
        for idx, sample in enumerate(dl):
            X = V(sample['img'].to(device))
            Y, Y_lengths = converter.encode(sample['label'])
            # Y = Y.to(device)

            optimizer.zero_grad()

            y_hat = model(X)

            preds_size = torch.IntTensor(t_cfg.bs).fill_(y_hat.shape[0])
            loss = criterion(y_hat, Y, preds_size, Y_lengths) / t_cfg.bs
            loss.backward()
            optimizer.step()

            l = loss.detach().cpu().numpy()

            L.append(l)
            print('\r', np.mean(L))

    return np.mean(L)

if __name__ == '__main__':
    main()
