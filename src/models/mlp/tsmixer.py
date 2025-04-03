from neuralforecast.models import TSMixer as tsmixer

class TSMixer(tsmixer):
    def __init__(self,input_length,span_length,output_length,enc_in, dec_in, c_out):
        super().__init__(h=output_length+span_length,input_size=input_length,n_series= enc_in)
        self.output_length = output_length
        self.c_out = c_out

    def forward(self, x_enc):
        B,N,T = x_enc.shape
        x = x_enc.transpose(2,1)

        batch_size = x.shape[0]

        # TSMixer: InstanceNorm + Mixing layers + Dense output layer + ReverseInstanceNorm
        if self.revin:
            x = self.norm(x, "norm")
        x = self.mixing_layers(x)
        x = x.permute(0, 2, 1)
        x = self.out(x)
        x = x.permute(0, 2, 1)
        if self.revin:
            x = self.norm(x, "denorm")

        # x = x.reshape(
        #     batch_size, self.h, self.loss.outputsize_multiplier * self.n_series
        # )
        x = x.transpose(2,1)
        return x

