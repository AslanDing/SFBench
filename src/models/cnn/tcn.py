import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        # self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
        #                                    stride=stride, padding=padding, dilation=dilation))
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,groups=n_inputs,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class ChannelWiseTCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(ChannelWiseTCNBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        padding = (kernel_size - 1) * dilation  # Causal padding

        # Channel-wise convolution with shared parameters
        # Use a single 1D conv kernel and apply it to all channels
        self.shared_conv = nn.Conv1d(
            in_channels=1,  # Treat each channel independently
            out_channels=1,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
            bias=False
        )

        # 1x1 convolution to adjust output channels
        self.pointwise = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1
        )

        # BatchNorm and ReLU
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

        # Residual connection
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        # Input shape: (batch, in_channels, seq_len)
        batch, channels, seq_len = x.shape

        # Apply shared convolution to each channel
        out = torch.zeros_like(x)
        for i in range(channels):
            # Extract single channel and apply shared conv
            channel_data = x[:, i:i+1, :]  # Shape: (batch, 1, seq_len)
            out[:, i:i+1, :] = self.shared_conv(channel_data)

        # Trim padding to maintain causality
        out = out[:, :, :-self.shared_conv.padding[0]]

        # Pointwise convolution to mix channels
        out = self.pointwise(out)

        # Apply BatchNorm and ReLU
        out = self.bn(out)
        out = self.relu(out)

        # Residual connection
        residual = self.residual(x)
        out = out + residual

        return self.relu(out)

class ChannelWiseTCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, kernel_size, dropout=0.2):
        super(ChannelWiseTCN, self).__init__()
        layers = []
        for i in range(num_layers):
            dilation = 2 ** i
            layers.append(
                ChannelWiseTCNBlock(
                    in_channels if i == 0 else hidden_channels,
                    hidden_channels,
                    kernel_size,
                    dilation
                )
            )
            layers.append(nn.Dropout(dropout))
        self.network = nn.Sequential(*layers)
        self.out_conv = nn.Conv1d(hidden_channels, 1, 1)  # Example output layer

    def forward(self, x):
        out = self.network(x)
        out = self.out_conv(out)
        return out


class SharedParamConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super(SharedParamConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = (kernel_size - 1) * dilation  # Causal padding

        # Define shared weight for a single input channel
        self.shared_weight = nn.Parameter(
            torch.randn(out_channels, 1, kernel_size)
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        # Input shape: (batch, in_channels, seq_len)
        # Expand shared weight to match in_channels
        weight = self.shared_weight.expand(-1, self.in_channels, -1)  # Shape: (out_channels, in_channels, kernel_size)

        # Apply standard convolution with shared weights
        out = F.conv1d(
            x,
            weight,
            bias=self.bias,
            stride=1,
            padding=self.padding,
            dilation=self.dilation
        )

        # Trim padding to maintain causality
        out = out[:, :, :-self.padding]
        return out

class SharedParamTCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(SharedParamTCNBlock, self).__init__()
        self.conv = SharedParamConv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        residual = self.residual(x)
        out = out + residual
        return self.relu(out)

class SharedParamTCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, kernel_size, dropout=0.2):
        super(SharedParamTCN, self).__init__()
        layers = []
        for i in range(num_layers):
            dilation = 2 ** i
            layers.append(
                SharedParamTCNBlock(
                    in_channels if i == 0 else hidden_channels,
                    hidden_channels,
                    kernel_size,
                    dilation
                )
            )
            layers.append(nn.Dropout(dropout))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        out = self.network(x)
        return out


class TCN(nn.Module):

    def __init__(self, input_length,span_length,output_length,enc_in, dec_in, c_out,
                 # input_size, output_size, num_channels,
                 kernel_size=2, dropout=0.3, emb_dropout=0.1, tied_weights=False):
        super(TCN, self).__init__()
        # self.encoder = nn.Linear(span_length+output_length, input_length)
        self.encoder = nn.Linear( input_length,span_length+output_length)
        # self.tcn = TemporalConvNet(input_length, [enc_in,enc_in,enc_in], kernel_size, dropout=dropout)
        # self.tcn = ChannelWiseTCN(input_length, enc_in, 3, kernel_size, dropout=dropout)
        self.tcn = SharedParamTCN(enc_in, enc_in, 3, kernel_size, dropout=dropout)

        self.decoder = nn.Linear(input_length, output_length)
        if tied_weights:
            self.decoder.weight = self.encoder.weight
            print("Weight tied")
        self.drop = nn.Dropout(emb_dropout)
        self.emb_dropout = emb_dropout
        self.init_weights()

    def init_weights(self):
        self.encoder.weight.data.normal_(0, 0.01)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.normal_(0, 0.01)

    def forward(self, x_enc):
        """Input ought to have dimension (N, C_in, L_in), where L_in is the seq_len; here the input is (N, L, C)"""

        B,N,T = x_enc.shape
        # encoder_input = x_enc.transpose(2,1)
        encoder_input = x_enc

        # emb = self.drop(self.encoder(encoder_input))
        emb = self.drop(encoder_input)
        y = self.tcn(emb).transpose(1, 2)
        y = self.decoder(y.transpose(1, 2))

        return y.contiguous()

