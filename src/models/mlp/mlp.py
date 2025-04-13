import torch
import torch.nn as nn
from neuralforecast.models import MLP as mlp


class MLP_his(mlp):
    def __init__(self,input_length,span_length,output_length,enc_in, dec_in, c_out,
                 num_layers=2,
                 hidden_size=1024):
        super(MLP_his, self).__init__(h=output_length+span_length,input_size=input_length)

        # Architecture
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        input_size_first_layer = (
                enc_in * input_length
        )

        # MultiLayer Perceptron
        layers = [
            nn.Linear(in_features=input_size_first_layer, out_features=hidden_size)
        ]
        for i in range(num_layers - 1):
            layers += [nn.Linear(in_features=hidden_size, out_features=hidden_size)]
        self.mlp = nn.ModuleList(layers)

        # Adapter with Loss dependent dimensions
        self.out = nn.Linear(
            in_features=hidden_size, out_features=output_length * c_out
        )

        self.output_length = output_length
        self.c_out = c_out


    def forward(self, x_enc):
        # x_enc = B N T
        B,N,T = x_enc.shape
        batch_size = B
        x_enc = x_enc.view(B,T*N)
        insample_y = x_enc

        # Flatten MLP inputs [B, L+H, C] -> [B, (L+H)*C]

        y_pred = insample_y.clone()
        for layer in self.mlp:
            y_pred = torch.relu(layer(y_pred))
        y_pred = self.out(y_pred)

        y_pred = y_pred.view(batch_size,N, self.h)
        # y_pred = y_pred.transpose(2,1)
        return y_pred


class MLP(mlp):
    def __init__(self,input_length,span_length,output_length,enc_in, dec_in, c_out,
                 num_layers=2,
                 hidden_size=1024):
        super(MLP, self).__init__(h=output_length+span_length,input_size=input_length)

        # Architecture
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        input_size_first_layer = (
                input_length
        )

        # MultiLayer Perceptron
        layers = [
            nn.Linear(in_features=input_size_first_layer, out_features=hidden_size)
        ]
        for i in range(num_layers - 1):
            layers += [nn.Linear(in_features=hidden_size, out_features=hidden_size)]
        self.mlp = nn.ModuleList(layers)

        # Adapter with Loss dependent dimensions
        self.out = nn.Linear(
            in_features=hidden_size, out_features=output_length
        )

        self.output_length = output_length
        self.c_out = c_out


    def forward(self, x_enc):
        # x_enc = B N T
        B,N,T = x_enc.shape
        batch_size = B
        x_enc = x_enc.view(B*N,T)
        insample_y = x_enc

        # Flatten MLP inputs [B, L+H, C] -> [B, (L+H)*C]

        y_pred = insample_y.clone()
        for layer in self.mlp:
            y_pred = torch.relu(layer(y_pred))
        y_pred = self.out(y_pred)

        y_pred = y_pred.view(batch_size, N, self.h)
        # y_pred = y_pred.transpose(2,1)
        return y_pred



