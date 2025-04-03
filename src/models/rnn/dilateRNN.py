from neuralforecast.models import DilatedRNN as DRNN


class DilatedRNN(DRNN):
    def __init__(self,input_length,span_length,output_length,enc_in, dec_in, c_out):
        super().__init__(h=output_length+span_length,input_size=input_length)
        self.output_length = output_length
        self.c_out = c_out
        self.enc_in = enc_in
        self.dec_in = dec_in

    def forward(self, x_enc):

        B,N,T = x_enc.shape

        x_enc = x_enc.view(B*N,T,1)
        encoder_input = x_enc #windows_batch["insample_y"]  # [B, L, 1]

        # batch_size, seq_len = encoder_input.shape[:2]

        # DilatedRNN forward
        for layer_num in range(len(self.rnn_stack)):
            residual = encoder_input
            output, _ = self.rnn_stack[layer_num](encoder_input)
            if layer_num > 0:
                output += residual
            encoder_input = output

        # Context adapter
        output = output.permute(0, 2, 1)  # [B, L, C] -> [B, C, L]
        context = self.context_adapter(output)  # [B, C, L] -> [B, C, h]

        # Final forecast
        context = context.permute(0, 2, 1)  # [B, C + F, h] -> [B, h, C + F]
        output = self.mlp_decoder(context)  # [B, h, C + F] -> [B, h, n_output]
        output = output.view(B,N,-1)
        output = output[:,-self.c_out:,-self.output_length:]
        return output
