import torch
import torch.nn as nn

class ConvRelu(nn.Module):
    def __init__(self, nin, nout, k, s, p, bn=False):
        super().__init__()
        self.conv = nn.Conv2d(nin, nout,
                              kernel_size=k,
                              stride=s,
                              padding=p)
        self.activation = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(nout)
        self.bn_ = bn

    def forward(self, x):
        x = self.conv(x)
        if self.bn_:
            x = self.bn(x)
        x = self.activation(x)
        return x

class LSTM(nn.Module):
    def __init__(self, nin, nh, nout):
        super(LSTM, self).__init__()
        self.rnn = nn.LSTM(nin, nh, bidirectional=True)
        self.embedding = nn.Linear(nh * 2, nout)
        if torch.cuda.is_available():
            self.ngpu = torch.cuda.device_count()

    def forward(self, input):
        recurrent, _ = data_parallel(
            self.rnn, input, self.ngpu)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)
        output = output.view(T, b, -1)

        return output

class CRNN(nn.Module):
    def __init__(self, c, nc, nh):
        super(CRNN, self).__init__()
        layers = [ConvRelu(c, 64, 3, 1, 1),
                 nn.MaxPool2d(2,2),
                 ConvRelu(64, 128, 3, 1, 1),
                 nn.MaxPool2d(2,2),
                 ConvRelu(128, 256, 3, 1, 1, True),
                 ConvRelu(256, 256, 3, 1, 1),
                 nn.MaxPool2d((2,2),(2,1),(0,1)),
                 ConvRelu(256, 512, 3, 1, 1, True),
                 ConvRelu(512, 512, 3, 1, 1),
                 nn.MaxPool2d((2,2), (2,1), (0,1)),
                 ConvRelu(512, 512, 2, 1, 0, True),
                 ]

        self.cnn = nn.Sequential(*layers)
        self.rnn = nn.Sequential(LSTM(512, nh, nh),
                                LSTM(nh, nh, nc))

    def forward(self, x):
        x = self.cnn(x)
        print(x.size())
        b, c, h, w = x.size()
        x = x.squeeze(2)
        x = x.permute(2, 0, 1)
        x = self.rnn(x)
        print(x.size())
        return x
