import torch
import torch.nn as nn
from neuralforecast.models import TCN as tcn
from neuralforecast.common._modules import  MLP, TemporalConvolutionEncoder
from typing import List, Optional

class TCN(tcn):
    def __init__(self,input_length,span_length,output_length,enc_in, dec_in, c_out,
                 kernel_size: int = 2,
                 dilations: List[int] = [1, 2, 4, 8, 16],
                 encoder_hidden_size: int = 128,
                 encoder_activation: str = "ReLU",
                 context_size: int = 10,
                 decoder_hidden_size: int = 128,
                 decoder_layers: int = 2):
        super(TCN, self).__init__(h=output_length+span_length,input_size=input_length)

        self.kernel_size = kernel_size
        self.dilations = dilations
        self.encoder_hidden_size = encoder_hidden_size
        self.encoder_activation = encoder_activation

        # Context adapter
        self.context_size = context_size

        # MLP decoder
        self.decoder_hidden_size = decoder_hidden_size
        self.decoder_layers = decoder_layers

        # TCN input size (1 for target variable y)
        input_encoder = (
                enc_in, # 1 + self.hist_exog_size + self.stat_exog_size + self.futr_exog_size
        )

        # ---------------------------------- Instantiate Model -----------------------------------#
        # Instantiate historic encoder
        self.hist_encoder = TemporalConvolutionEncoder(
            in_channels=input_encoder,
            out_channels=self.encoder_hidden_size,
            kernel_size=self.kernel_size,  # Almost like lags
            dilations=self.dilations,
            activation=self.encoder_activation,
        )

        # Context adapter
        self.context_adapter = nn.Linear(in_features=self.input_size, out_features=h)

        # Decoder MLP
        self.mlp_decoder = MLP(
            in_features=self.encoder_hidden_size + self.futr_exog_size,
            out_features=c_out, #self.loss.outputsize_multiplier,
            hidden_size=self.decoder_hidden_size,
            num_layers=self.decoder_layers,
            activation="ReLU",
            dropout=0.0,
        )

    def forward(self, x_enc):

        # x_enc = B N T
        B,N,T = x_enc.shape
        encoder_input = x_enc.transpose(2,1)

        batch_size, input_size = encoder_input.shape[:2]

        # TCN forward
        hidden_state = self.hist_encoder(encoder_input)  # [B, L, C]

        # Context adapter
        hidden_state = hidden_state.permute(0, 2, 1)  # [B, L, C] -> [B, C, L]
        context = self.context_adapter(hidden_state)  # [B, C, L] -> [B, C, h]

        context = context.swapaxes(1, 2)  # [B, C + F, h] -> [B, h, C + F]

        # Final forecast
        output = self.mlp_decoder(context)  # [B, h, C + F] -> [B, h, n_output]

        output = output.transpose(2,1)
        return output


