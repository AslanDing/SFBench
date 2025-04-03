import torch
from neuralforecast.models import Autoformer as autoformer

class AutoFormer(autoformer):
    def __init__(self,input_length,span_length,output_length,enc_in, dec_in, c_out):
        super().__init__(h=output_length+span_length,input_size=input_length,enc_in=enc_in,
                         dec_in=dec_in)
        self.output_length = output_length
        self.c_out = c_out
        self.enc_in = enc_in
        self.dec_in = dec_in

    def forward(self, x_enc):
        B, N, T = x_enc.shape
        # Parse windows_batch
        x_enc = x_enc.view(B*N,T,1)
        insample_y = x_enc #.transpose(2,1) # windows_batch["insample_y"]

        x_mark_enc = None
        x_mark_dec = None

        x_dec = torch.zeros(size=(len(insample_y), self.h, 1), device=insample_y.device)
        x_dec = torch.cat([insample_y[:, -self.label_len:, :], x_dec], dim=1)

        # decomp init
        mean = torch.mean(insample_y, dim=1).unsqueeze(1).repeat(1, self.h, 1)
        zeros = torch.zeros(
            [x_dec.shape[0], self.h, x_dec.shape[2]], device=insample_y.device
        )
        seasonal_init, trend_init = self.decomp(insample_y)
        # decoder input
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = torch.cat(
            [seasonal_init[:, -self.label_len:, :], zeros], dim=1
        )
        # enc
        enc_out = self.enc_embedding(insample_y, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # dec
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(
            dec_out, enc_out, x_mask=None, cross_mask=None, trend=trend_init
        )
        # final
        dec_out = trend_part + seasonal_part

        forecast = dec_out[:, -self.h:]

        forecast = forecast[:,-self.output_length:].view(B,N,-1)

        return forecast


