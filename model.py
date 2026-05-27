import torch
import torch.nn as nn
import torchvision.models as models


class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()

        padding = kernel_size // 2

        self.conv = nn.Conv2d(
            input_dim + hidden_dim,
            4 * hidden_dim,
            kernel_size,
            padding=padding
        )

        self.hidden_dim = hidden_dim

    def forward(self, x, h, c):

        combined = torch.cat([x, h], dim=1)

        gates = self.conv(combined)

        i, f, o, g = torch.chunk(gates, 4, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next


class ConvLSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        self.cell = ConvLSTMCell(input_dim, hidden_dim)

    def forward(self, x):

        B, T, C, H, W = x.shape

        h = torch.zeros(B, self.cell.hidden_dim, H, W).to(x.device)
        c = torch.zeros(B, self.cell.hidden_dim, H, W).to(x.device)

        outputs = []

        for t in range(T):

            h, c = self.cell(x[:, t], h, c)
            outputs.append(h)

        outputs = torch.stack(outputs, dim=1)

        return outputs


class LaneDetectionModel(nn.Module):

    def __init__(self):
        super().__init__()

        resnet = models.resnet18(weights="DEFAULT")

        self.encoder = nn.Sequential(*list(resnet.children())[:-2])

        self.convlstm = ConvLSTM(512, 256)

        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(256, 128, 2, stride=2),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 16, 2, stride=2),
            nn.ReLU(),

            nn.ConvTranspose2d(16, 1, 2, stride=2),
            nn.Sigmoid()
        )

    def forward(self, x):

        B, T, C, H, W = x.shape

        features = []

        for t in range(T):

            f = self.encoder(x[:, t])
            features.append(f)

        features = torch.stack(features, dim=1)

        temporal = self.convlstm(features)

        output = self.decoder(temporal[:, -1])

        return output