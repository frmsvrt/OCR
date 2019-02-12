import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_dim, batch_size, ntoken, nhid=512, nlayers=2):
        super().__init__()
        self.nhid = nhid
        self.nlayers = nlayers

        # The LSTM takes
        self.lstm = nn.LSTM(input_dim, self.nhid, nlayers)
        self.fc = nn.Linear(self.nhid, ntoken)
        self.hidden = self.init_hidden(batch_size)

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        return (weight.new_zeros(self.nlayers, batch_size, self.nhid),
                weight.new_zeros(self.nlayers, batch_size, self.nhid))

    def forward(self, images):
        seq_len = images.shape[1]
        images = images.permute(1, 0, 2).contiguous()
        lstm_out, self.hidden = self.lstm(images, self.hidden)
        lstm_features = self.fc(lstm_out)
        return lstm_features
