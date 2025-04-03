from neuralforecast.models import PatchTST as patchTST


class PatchTST(patchTST):
    def __init__(self,input_length,span_length,output_length,enc_in, dec_in, c_out):
        super().__init__(h=output_length+span_length,input_size=input_length,enc_in=enc_in,
                         dec_in=dec_in)
        self.output_length = output_length
        self.c_out = c_out

    def forward(self, x_enc):  # x: [batch, input_size]


        B, N, T = x_enc.shape
        # Parse windows_batch
        x_enc = x_enc.view(B*N,T,1)

        # Parse windows_batch
        x = x_enc  #windows_batch["insample_y"]

        x = x.permute(0, 2, 1)  # x: [Batch, 1, input_size]
        x = self.model(x)
        forecast = x.reshape(x.shape[0], self.h, -1)  # x: [Batch, h, c_out]

        forecast = forecast[:,-self.output_length:].view(B,N,-1)
        return forecast

