from neuralforecast.models import NLinear as linear

class NLinear(linear):
    def __init__(self,input_length,span_length,output_length,enc_in, dec_in, c_out):
        super().__init__(h=output_length+span_length,input_size=input_length)

        self.output_length = output_length
        self.c_out = c_out

    def forward(self, x_enc):
        # x_enc = B N T
        B,N,T = x_enc.shape
        batch_size = B
        x_enc = x_enc.view(B*N,T)

        # Input normalization
        last_value = x_enc[:, -1:]
        norm_insample_y = x_enc - last_value

        # Final
        forecast = self.linear(norm_insample_y) + last_value
        forecast = forecast.view(B,N,-1)
        # forecast = forecast.reshape(batch_size, self.h, self.loss.outputsize_multiplier)
        return forecast


