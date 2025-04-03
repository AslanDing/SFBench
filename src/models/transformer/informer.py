import torch
from neuralforecast.models import Informer as informer

class Informer(informer):
    def __init__(self,input_length,span_length,output_length,enc_in, dec_in, c_out):
        super().__init__(h=output_length+span_length,input_size=input_length,n_series=enc_in,
                         dec_in=dec_in)
        self.output_length = output_length
        self.c_out = c_out
        self.enc_in = enc_in
        self.dec_in = dec_in

    def forward(self, x_enc):

        B, N, T = x_enc.shape
        x_enc = x_enc.view(B*N,T,1)
        insample_y = x_enc
        x_mark_enc = None
        x_mark_dec = None

        x_dec = torch.zeros(size=(len(insample_y), self.h, 1), device=insample_y.device)
        x_dec = torch.cat([insample_y[:, -self.label_len :, :], x_dec], dim=1)

        enc_out = self.enc_embedding(insample_y, x_mark_enc)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)  # attns visualization

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)

        forecast = dec_out[:, -self.h :]

        forecast = forecast[:,-self.output_length:].view(B,N,-1)

        return forecast


