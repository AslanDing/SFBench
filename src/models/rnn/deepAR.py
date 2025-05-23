from neuralforecast.models import DeepAR as deepar
from neuralforecast.models.deepar import Decoder

class DeepAR(deepar):
    def __init__(self,input_length,span_length,output_length,enc_in, dec_in, c_out):
        super().__init__(h=output_length+span_length,input_size=input_length,enc_in=enc_in,
                         dec_in=dec_in)
        self.output_length = output_length
        self.c_out = c_out
        self.enc_in = enc_in
        self.dec_in = dec_in

        out_feature = output_length//input_length + 1

        self.decoder = Decoder(
            in_features=self.encoder_hidden_size,
            out_features=out_feature, #self.loss.outputsize_multiplier
            hidden_size=0,
            hidden_layers=0
        )


    def forward(self, x_enc):

        B,N,T = x_enc.shape

        x_enc = x_enc.view(B*N,T,1)
        # Parse windows_batch
        encoder_input = x_enc

        # RNN forward
        if self.maintain_state:
            rnn_state = self.rnn_state
        else:
            rnn_state = None

        hidden_state, rnn_state = self.hist_encoder(
            encoder_input, rnn_state
        )  # [B, input_size-1, rnn_hidden_state]

        if self.maintain_state:
            self.rnn_state = rnn_state

        # Decoder forward
        output = self.decoder(hidden_state)  # [B, input_size-1, output_size]
        output = output[:,:,:].view(B,N,-1)
        output = output[:,-self.c_out:,-self.output_length:]
        # Return only horizon part
        return output