import torch
import torch.nn as nn
from transformers.models.gpt2.modeling_gpt2 import GPT2Model

class MLP(nn.Module):
    '''
    Multilayer perceptron to encode/decode high dimension representation of sequential data
    '''

    def __init__(self,
                 f_in,
                 f_out,
                 hidden_dim=256,
                 hidden_layers=2,
                 dropout=0.1,
                 activation='tanh'):
        super(MLP, self).__init__()
        self.f_in = f_in
        self.f_out = f_out
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            raise NotImplementedError

        layers = [nn.Linear(self.f_in, self.hidden_dim),
                  self.activation, nn.Dropout(self.dropout)]
        for i in range(self.hidden_layers - 2):
            layers += [nn.Linear(self.hidden_dim, self.hidden_dim),
                       self.activation, nn.Dropout(dropout)]

        layers += [nn.Linear(hidden_dim, f_out)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        # x:     B x S x f_in
        # y:     B x S x f_out
        y = self.layers(x)
        return y


class AutoTimes(nn.Module):
    def __init__(self,input_length,span_length,output_length,enc_in, dec_in, c_out,
                 device = 'cpu',mlp_hidden_layers = 2,mlp_hidden_dim=256,dropout=0.1,
                 mlp_activation = 'relu', token_len = 24 ):
        super(AutoTimes, self).__init__()
        self.input_length = input_length
        self.output_length = span_length+output_length
        self.in_token_len = token_len #configs.token_len
        fold = input_length//token_len
        self.out_token_len = output_length//fold #configs.token_len
        self.device = device #f"cuda:{configs.gpu}"

        self.gpt2 = GPT2Model.from_pretrained('gpt2')
        self.hidden_dim_of_gpt2 = 768
        self.mix = False #configs.mix_embeds

        if self.mix:
            self.add_scale = nn.Parameter(torch.ones([]))

        for name, param in self.gpt2.named_parameters():
            param.requires_grad = False


        if mlp_hidden_layers == 0:
            self.encoder = nn.Linear(self.in_token_len, self.hidden_dim_of_gpt2)
            self.decoder = nn.Linear(self.hidden_dim_of_gpt2, self.out_token_len)
        else:
            self.encoder = MLP(self.in_token_len, self.hidden_dim_of_gpt2,
                               mlp_hidden_dim, mlp_hidden_layers,
                               dropout, mlp_activation)
            self.decoder = MLP(self.hidden_dim_of_gpt2, self.out_token_len,
                               mlp_hidden_dim, mlp_hidden_layers,
                               dropout, mlp_activation)

    def forecast(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        x_enc = x_enc.transpose(2,1)

        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        bs, _, n_vars = x_enc.shape
        # x_enc: [bs x nvars x seq_len]
        x_enc = x_enc.permute(0, 2, 1)
        # x_enc: [bs * nvars x seq_len]
        x_enc = x_enc.reshape(x_enc.shape[0] * x_enc.shape[1], -1)
        # fold_out: [bs * n_vars x token_num x token_len]
        fold_out = x_enc.unfold(dimension=-1, size=self.in_token_len, step=self.in_token_len)
        token_num = fold_out.shape[1]
        # times_embeds: [bs * n_vars x token_num x hidden_dim_of_gpt2]
        times_embeds = self.encoder(fold_out)
        if self.mix:
            times_embeds = times_embeds / times_embeds.norm(dim=2, keepdim=True)
            x_mark_enc = x_mark_enc / x_mark_enc.norm(dim=2, keepdim=True)
            times_embeds = times_embeds + self.add_scale * x_mark_enc
        # outputs: [bs * n_vars x token_num x hidden_dim_of_gpt2]
        outputs = self.gpt2(
            inputs_embeds=times_embeds).last_hidden_state
        # dec_out: [bs * n_vars x token_num x token_len]
        dec_out = self.decoder(outputs)
        dec_out = dec_out.reshape(bs, n_vars, -1)
        # dec_out: [bs x token_num * token_len x n_vars]
        dec_out = dec_out.permute(0, 2, 1)

        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, token_num * self.out_token_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, token_num * self.out_token_len, 1))
        dec_out = dec_out[:,-self.output_length:,:].transpose(2,1)
        return dec_out

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)