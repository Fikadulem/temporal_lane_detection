import torch
import torch.nn as nn
import torchvision.models as models


class ConvLSTMCell(nn.Module):
    """Single convolutional LSTM cell.

    Computes the four gates (input, forget, output, cell) from the
    concatenation of the current spatial input and the previous hidden
    state using one shared 2-D convolution, then updates the cell and
    hidden state tensors.

    Args:
        input_dim (int): Number of channels in the input feature map.
        hidden_dim (int): Number of channels in the hidden state.
        kernel_size (int): Spatial kernel size for the gate convolution.
    """

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
        """Perform one ConvLSTM step.

        Args:
            x (torch.Tensor): Input feature map, shape (B, input_dim, H, W).
            h (torch.Tensor): Previous hidden state, shape (B, hidden_dim, H, W).
            c (torch.Tensor): Previous cell state, shape (B, hidden_dim, H, W).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Updated hidden state and cell
            state, both of shape (B, hidden_dim, H, W).
        """
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
    """Multi-step ConvLSTM that processes an entire frame sequence.

    Wraps a single :class:`ConvLSTMCell` and iterates over the time
    dimension, collecting the hidden state at every timestep.

    Args:
        input_dim (int): Number of channels in each input feature map.
        hidden_dim (int): Number of channels in the hidden state.
    """

    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        self.cell = ConvLSTMCell(input_dim, hidden_dim)

    def forward(self, x):
        """Process a sequence of feature maps through the ConvLSTM cell.

        Args:
            x (torch.Tensor): Input sequence, shape (B, T, C, H, W).

        Returns:
            torch.Tensor: Hidden states for all timesteps,
            shape (B, T, hidden_dim, H, W).
        """
        B, T, C, H, W = x.shape

        h = torch.zeros(B, self.cell.hidden_dim, H, W).to(x.device)
        c = torch.zeros(B, self.cell.hidden_dim, H, W).to(x.device)

        outputs = []

        for t in range(T):

            h, c = self.cell(x[:, t], h, c)
            outputs.append(h)

        outputs = torch.stack(outputs, dim=1)

        return outputs


def build_decoder(input_dim):
    """Build the shared transposed-convolution decoder head.

    Progressively upsamples the feature map by a factor of 2 five times,
    from the encoder spatial resolution back to the original input size,
    producing a single-channel lane probability map in [0, 1].

    Args:
        input_dim (int): Number of input channels to the first
            transposed convolution.

    Returns:
        nn.Sequential: Decoder module with five upsampling blocks and a
        final sigmoid activation.
    """
    return nn.Sequential(

        nn.ConvTranspose2d(input_dim, 128, 2, stride=2),
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


class ConvLSTMLaneDetectionModel(nn.Module):
    """Temporal lane segmentation model using a ConvLSTM.

    Encodes each frame in a sequence with a shared ResNet-18 backbone,
    feeds the resulting feature sequence through a ConvLSTM to capture
    temporal dependencies, then decodes the final hidden state into a
    pixel-wise lane probability map.

    Input shape:  (B, T, 3, H, W)
    Output shape: (B, 1, H, W)
    """

    def __init__(self):
        super().__init__()

        resnet = models.resnet18(weights="DEFAULT")

        self.encoder = nn.Sequential(*list(resnet.children())[:-2])

        self.convlstm = ConvLSTM(512, 256)

        self.decoder = build_decoder(256)

    def forward(self, x):
        """Run the ConvLSTM lane detection model on a frame sequence.

        Args:
            x (torch.Tensor): Input sequence, shape (B, T, 3, H, W).

        Returns:
            torch.Tensor: Lane probability map, shape (B, 1, H, W),
            values in [0, 1].
        """
        B, T, C, H, W = x.shape

        features = []

        for t in range(T):

            f = self.encoder(x[:, t])
            features.append(f)

        features = torch.stack(features, dim=1)

        temporal = self.convlstm(features)

        output = self.decoder(temporal[:, -1])

        return output


class CNNLaneDetectionModel(nn.Module):
    """CNN baseline lane segmentation model without recurrence.

    Encodes each frame in a sequence with a shared ResNet-18 backbone,
    aggregates temporal features by mean pooling across the time dimension,
    projects the aggregated map with two convolutional layers, then decodes
    it into a pixel-wise lane probability map.

    Input shape:  (B, T, 3, H, W)
    Output shape: (B, 1, H, W)
    """

    def __init__(self):
        super().__init__()

        resnet = models.resnet18(weights="DEFAULT")

        self.encoder = nn.Sequential(*list(resnet.children())[:-2])

        self.temporal_projection = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.decoder = build_decoder(256)

    def forward(self, x):
        """Run the CNN baseline lane detection model on a frame sequence.

        Args:
            x (torch.Tensor): Input sequence, shape (B, T, 3, H, W).

        Returns:
            torch.Tensor: Lane probability map, shape (B, 1, H, W),
            values in [0, 1].
        """
        B, T, C, H, W = x.shape

        features = []

        for t in range(T):

            f = self.encoder(x[:, t])
            features.append(f)

        features = torch.stack(features, dim=1)
        aggregated = features.mean(dim=1)
        projected = self.temporal_projection(aggregated)

        output = self.decoder(projected)

        return output


class LaneDetectionModel(ConvLSTMLaneDetectionModel):
    """Alias for ConvLSTMLaneDetectionModel retained for backward compatibility."""

    pass


def get_model(model_type="convlstm"):
    """Instantiate and return the requested lane detection model.

    Args:
        model_type (str): Architecture to build. ``'convlstm'`` returns
            :class:`ConvLSTMLaneDetectionModel`; ``'cnn'`` returns
            :class:`CNNLaneDetectionModel`.

    Returns:
        nn.Module: Instantiated model (weights not loaded).

    Raises:
        ValueError: If ``model_type`` is not ``'convlstm'`` or ``'cnn'``.
    """
    if model_type == "convlstm":
        return ConvLSTMLaneDetectionModel()

    if model_type == "cnn":
        return CNNLaneDetectionModel()

    raise ValueError(f"Unsupported model_type: {model_type}. Use 'convlstm' or 'cnn'.")