from neuralforecast import NeuralForecast
from neuralforecast.models import TimeLLM
from neuralforecast.utils import AirPassengersPanel

class TimeLLM(TimeLLM):
    def __init__(self,input_length,span_length,output_length,enc_in, dec_in, c_out):
        super().__init__(h=output_length+span_length,input_size=input_length,enc_in=enc_in,
                         dec_in=dec_in)
        self.output_length = output_length
        self.c_out = c_out

    def forward(self, x_enc):
        # x_enc = B N T
        x_enc = x_enc.transpose(1,2)
        # need B T N
        output = self.forecast(x_enc)
        output = output[:, -self.output_length:, -self.c_out:]
        output = output.transpose(1, 2)
        # output = B N T
        return output