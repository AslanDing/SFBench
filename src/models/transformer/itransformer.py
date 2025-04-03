from neuralforecast.models import iTransformer as itran

class iTransformer(itran):
    def __init__(self,input_length,span_length,output_length,enc_in, dec_in, c_out):
        super().__init__(h=output_length+span_length,input_size=input_length,n_series=enc_in,
                         dec_in=dec_in)
        self.output_length = output_length
        self.c_out = c_out
        self.enc_in = enc_in
        self.dec_in = dec_in


    def forward(self, x_enc):
        B, N, T = x_enc.shape
        insample_y = x_enc.transpose(2,1) #windows_batch["insample_y"]

        y_pred = self.forecast(insample_y)
        y_pred = y_pred.reshape(insample_y.shape[0], self.h, -1)
        y_pred = y_pred.transpose(2,1)
        return y_pred