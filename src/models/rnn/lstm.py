import torch
import torch.nn as nn
from neuralforecast.models import LSTM as lstm
from neuralforecast.common._modules import MLP

from typing import Optional
import warnings



class LSTMhis(lstm):
    def __init__(self,input_length,span_length,output_length,enc_in, dec_in, c_out,
                 inference_input_size: Optional[int] = None,
                 encoder_n_layers: int = 2,
                 encoder_hidden_size: int = 128,
                 encoder_bias: bool = True,
                 encoder_dropout: float = 0.0,
                 context_size: Optional[int] = None,
                 decoder_hidden_size: int = 128,
                 decoder_layers: int = 2):
        super(LSTMhis, self).__init__(h=output_length+span_length,input_size=input_length)

        # LSTM
        self.encoder_n_layers = encoder_n_layers
        self.encoder_hidden_size = encoder_hidden_size
        self.encoder_bias = encoder_bias
        self.encoder_dropout = encoder_dropout

        # Context adapter
        if context_size is not None:
            warnings.warn(
                "context_size is deprecated and will be removed in future versions."
            )

        # MLP decoder
        self.decoder_hidden_size = decoder_hidden_size
        self.decoder_layers = decoder_layers

        # LSTM input size (1 for target variable y)
        input_encoder = (
                enc_in
        )

        # Instantiate model
        self.rnn_state = None
        self.maintain_state = False
        self.hist_encoder = nn.LSTM(
            input_size=input_encoder,
            hidden_size=self.encoder_hidden_size,
            num_layers=self.encoder_n_layers,
            bias=self.encoder_bias,
            dropout=self.encoder_dropout,
            batch_first=True,
            proj_size=self.loss.outputsize_multiplier if self.RECURRENT else 0,
        )

        # Decoder MLP
        if not self.RECURRENT:
            self.mlp_decoder = MLP(
                in_features=self.encoder_hidden_size + self.futr_exog_size,
                out_features=self.loss.outputsize_multiplier,
                hidden_size=self.decoder_hidden_size,
                num_layers=self.decoder_layers,
                activation="ReLU",
                dropout=0.0,
            )


        self.output_length = output_length
        self.c_out = c_out

    def forward(self, x_enc):

        B,N,T = x_enc.shape

        x_enc = x_enc.tanspose(2,1)
        encoder_input = x_enc
        batch_size, seq_len = encoder_input.shape[:2]

        if self.RECURRENT:
            if self.maintain_state:
                rnn_state = self.rnn_state
            else:
                rnn_state = None

            output, rnn_state = self.hist_encoder(
                encoder_input, rnn_state
            )  # [B, seq_len, n_output]
            if self.maintain_state:
                self.rnn_state = rnn_state
        else:
            hidden_state, _ = self.hist_encoder(
                encoder_input, None
            )  # [B, seq_len, rnn_hidden_state]
            hidden_state = hidden_state[
                           :, -self.h:
                           ]  # [B, seq_len, rnn_hidden_state] -> [B, h, rnn_hidden_state]

            output = self.mlp_decoder(
                hidden_state
            )  # [B, h, rnn_hidden_state + F] -> [B, seq_len, n_output]

        output = output[:, -self.h:,:].transpose(2,1)
        return output

class LSTM(lstm):
    def __init__(self,input_length,span_length,output_length,enc_in, dec_in, c_out,
                 inference_input_size: Optional[int] = None,
                 encoder_n_layers: int = 2,
                 encoder_hidden_size: int = 4,
                 encoder_bias: bool = True,
                 encoder_dropout: float = 0.0,
                 context_size: Optional[int] = None,
                 decoder_hidden_size: int = 4,
                 decoder_layers: int = 2):
        super(LSTM, self).__init__(h=output_length+span_length,input_size=input_length)

        # LSTM
        self.encoder_n_layers = encoder_n_layers
        self.encoder_hidden_size = encoder_hidden_size
        self.encoder_bias = encoder_bias
        self.encoder_dropout = encoder_dropout

        # Context adapter
        if context_size is not None:
            warnings.warn(
                "context_size is deprecated and will be removed in future versions."
            )

        # MLP decoder
        self.decoder_hidden_size = decoder_hidden_size
        self.decoder_layers = decoder_layers

        # LSTM input size (1 for target variable y)
        input_encoder = (
                enc_in
        )

        # Instantiate model
        self.rnn_state = None
        self.maintain_state = False
        self.hist_encoder = nn.LSTM(
            input_size=1,
            hidden_size=self.encoder_hidden_size,
            num_layers=self.encoder_n_layers,
            bias=self.encoder_bias,
            dropout=self.encoder_dropout,
            batch_first=True,
            proj_size=self.loss.outputsize_multiplier if self.RECURRENT else 0,
        )
        multi_scale = output_length//input_length + 1
        # Decoder MLP
        if not self.RECURRENT:
            self.mlp_decoder = MLP(
                in_features=self.encoder_hidden_size + self.futr_exog_size,
                out_features=multi_scale, #self.loss.outputsize_multiplier,
                hidden_size=self.decoder_hidden_size,
                num_layers=self.decoder_layers,
                activation="ReLU",
                dropout=0.0,
            )


        self.output_length = output_length
        self.c_out = c_out

    def forward(self, x_enc):

        B,N,T = x_enc.shape
        x_enc = x_enc.view(B*N,1,T)
        x_enc = x_enc.transpose(2,1)
        encoder_input = x_enc
        batch_size, seq_len = encoder_input.shape[:2]

        if self.RECURRENT:
            if self.maintain_state:
                rnn_state = self.rnn_state
            else:
                rnn_state = None

            output, rnn_state = self.hist_encoder(
                encoder_input, rnn_state
            )  # [B, seq_len, n_output]
            if self.maintain_state:
                self.rnn_state = rnn_state
        else:
            hidden_state, _ = self.hist_encoder(
                encoder_input, None
            )  # [B, seq_len, rnn_hidden_state]
            hidden_state = hidden_state[
                           :, -self.h:
                           ]  # [B, seq_len, rnn_hidden_state] -> [B, h, rnn_hidden_state]

            output = self.mlp_decoder(
                hidden_state
            )  # [B, h, rnn_hidden_state + F] -> [B, seq_len, n_output]
        output = output.view(output.shape[0],-1).view(B,N,-1)
        output = output[:,:, -self.h:]
        # output = output.view(B,N,-1)
        return output
