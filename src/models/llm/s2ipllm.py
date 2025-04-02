from functools import partial
from dataclasses import dataclass

import math
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from einops import rearrange
from transformers import GPT2Tokenizer


def vec_num2repr(val, base, prec, max_val):
    """
    Convert numbers to a representation in a specified base with precision.

    Parameters:
    - val (np.array): The numbers to represent.
    - base (int): The base of the representation.
    - prec (int): The precision after the 'decimal' point in the base representation.
    - max_val (float): The maximum absolute value of the number.

    Returns:
    - tuple: Sign and digits in the specified base representation.

    Examples:
        With base=10, prec=2:
            0.5   ->    50
            3.52  ->   352
            12.5  ->  1250
    """
    base = float(base)
    bs = val.shape[0]
    sign = 1 * (val >= 0) - 1 * (val < 0)
    val = np.abs(val)
    max_bit_pos = int(np.ceil(np.log(max_val) / np.log(base)).item())

    before_decimals = []
    for i in range(max_bit_pos):
        digit = (val / base ** (max_bit_pos - i - 1)).astype(int)
        before_decimals.append(digit)
        val -= digit * base ** (max_bit_pos - i - 1)

    before_decimals = np.stack(before_decimals, axis=-1)

    if prec > 0:
        after_decimals = []
        for i in range(prec):
            digit = (val / base ** (-i - 1)).astype(int)
            after_decimals.append(digit)
            val -= digit * base ** (-i - 1)

        after_decimals = np.stack(after_decimals, axis=-1)
        digits = np.concatenate([before_decimals, after_decimals], axis=-1)
    else:
        digits = before_decimals
    return sign, digits

def vec_repr2num(sign, digits, base, prec, half_bin_correction=True):
    """
    Convert a string representation in a specified base back to numbers.

    Parameters:
    - sign (np.array): The sign of the numbers.
    - digits (np.array): Digits of the numbers in the specified base.
    - base (int): The base of the representation.
    - prec (int): The precision after the 'decimal' point in the base representation.
    - half_bin_correction (bool): If True, adds 0.5 of the smallest bin size to the number.

    Returns:
    - np.array: Numbers corresponding to the given base representation.
    """
    base = float(base)
    bs, D = digits.shape
    digits_flipped = np.flip(digits, axis=-1)
    powers = -np.arange(-prec, -prec + D)
    val = np.sum(digits_flipped/base**powers, axis=-1)

    if half_bin_correction:
        val += 0.5/base**prec

    return sign * val


@dataclass
class SerializerSettings:
    """
    Settings for serialization of numbers.

    Attributes:
    - base (int): The base for number representation.
    - prec (int): The precision after the 'decimal' point in the base representation.
    - signed (bool): If True, allows negative numbers. Default is False.
    - fixed_length (bool): If True, ensures fixed length of serialized string. Default is False.
    - max_val (float): Maximum absolute value of number for serialization.
    - time_sep (str): Separator for different time steps.
    - bit_sep (str): Separator for individual digits.
    - plus_sign (str): String representation for positive sign.
    - minus_sign (str): String representation for negative sign.
    - half_bin_correction (bool): If True, applies half bin correction during deserialization. Default is True.
    - decimal_point (str): String representation for the decimal point.
    """
    base: int = 10
    prec: int = 1
    signed: bool = True
    fixed_length: bool = False
    max_val: float = 1e7
    time_sep: str = ' ,'
    bit_sep: str = ' '
    plus_sign: str = ''
    minus_sign: str = ' -'
    half_bin_correction: bool = True
    decimal_point: str = ''
    missing_str: str = ' Nan'


def serialize_arr(arr, settings: SerializerSettings):
    """
    Serialize an array of numbers (a time series) into a string based on the provided settings.

    Parameters:
    - arr (np.array): Array of numbers to serialize.
    - settings (SerializerSettings): Settings for serialization.

    Returns:
    - str: String representation of the array.
    """
    # max_val is only for fixing the number of bits in nunm2repr so it can be vmapped
    assert np.all(np.abs(arr[~np.isnan(arr)]) <= settings.max_val), f"abs(arr) must be <= max_val,\
         but abs(arr)={np.abs(arr)}, max_val={settings.max_val}"

    if not settings.signed:
        assert np.all(arr[~np.isnan(arr)] >= 0), f"unsigned arr must be >= 0"
        plus_sign = minus_sign = ''
    else:
        plus_sign = settings.plus_sign
        minus_sign = settings.minus_sign

    vnum2repr = partial(vec_num2repr, base=settings.base, prec=settings.prec, max_val=settings.max_val)
    sign_arr, digits_arr = vnum2repr(np.where(np.isnan(arr), np.zeros_like(arr), arr))

    ismissing = np.isnan(arr)

    def tokenize(arr):
        return ''.join([settings.bit_sep + str(b) for b in arr])

    bit_strs = []

    for sign, digits, missing in zip(sign_arr[0], digits_arr[0], ismissing[0]):

        if not settings.fixed_length:
            # remove leading zeros
            nonzero_indices = np.where(digits != 0)[0]
            if len(nonzero_indices) == 0:
                digits = np.array([0])
            else:
                digits = digits[nonzero_indices[0]:]
            # add a decimal point
            prec = settings.prec
            if len(settings.decimal_point):
                digits = np.concatenate([digits[:-prec], np.array([settings.decimal_point]), digits[-prec:]])
        digits = tokenize(digits)
        sign_sep = plus_sign if sign == 1 else minus_sign
        if missing:
            bit_strs.append(settings.missing_str)
        else:
            bit_strs.append(sign_sep + digits)
    bit_str = settings.time_sep.join(bit_strs)
    bit_str += settings.time_sep  # otherwise there is ambiguity in number of digits in the last time step
    return bit_str

def deserialize_str(bit_str, settings: SerializerSettings, ignore_last=False, steps=None):
    """
    Deserialize a string into an array of numbers (a time series) based on the provided settings.

    Parameters:
    - bit_str (str): String representation of an array of numbers.
    - settings (SerializerSettings): Settings for deserialization.
    - ignore_last (bool): If True, ignores the last time step in the string (which may be incomplete due to token limit etc.). Default is False.
    - steps (int, optional): Number of steps or entries to deserialize.

    Returns:
    - None if deserialization failed for the very first number, otherwise
    - np.array: Array of numbers corresponding to the string.
    """
    # ignore_last is for ignoring the last time step in the prediction, which is often a partially generated due to token limit
    orig_bitstring = bit_str
    bit_strs = bit_str.split(settings.time_sep)
    # remove empty strings
    bit_strs = [a for a in bit_strs if len(a) > 0]
    if ignore_last:
        bit_strs = bit_strs[:-1]
    if steps is not None:
        bit_strs = bit_strs[:steps]
    vrepr2num = partial(vec_repr2num,base=settings.base,prec=settings.prec,half_bin_correction=settings.half_bin_correction)
    max_bit_pos = int(np.ceil(np.log(settings.max_val)/np.log(settings.base)).item())
    sign_arr = []
    digits_arr = []
    try:
        for i, bit_str in enumerate(bit_strs):
            if bit_str.startswith(settings.minus_sign):
                sign = -1
            elif bit_str.startswith(settings.plus_sign):
                sign = 1
            else:
                assert settings.signed == False, f"signed bit_str must start with {settings.minus_sign} or {settings.plus_sign}"
            bit_str = bit_str[len(settings.plus_sign):] if sign==1 else bit_str[len(settings.minus_sign):]
            if settings.bit_sep=='':
                bits = [b for b in bit_str.lstrip()]
            else:
                bits = [b[:1] for b in bit_str.lstrip().split(settings.bit_sep)]
            if settings.fixed_length:
                assert len(bits) == max_bit_pos+settings.prec, f"fixed length bit_str must have {max_bit_pos+settings.prec} bits, but has {len(bits)}: '{bit_str}'"
            digits = []
            for b in bits:
                if b==settings.decimal_point:
                    continue
                # check if is a digit
                if b.isdigit():
                    digits.append(int(b))
                else:
                    break
            #digits = [int(b) for b in bits]
            sign_arr.append(sign)
            digits_arr.append(digits)
    except Exception as e:
        print(f"Error deserializing {settings.time_sep.join(bit_strs[i-2:i+5])}{settings.time_sep}\n\t{e}")
        print(f'Got {orig_bitstring}')
        print(f"Bitstr {bit_str}, separator {settings.bit_sep}")
        # At this point, we have already deserialized some of the bit_strs, so we return those below
    if digits_arr:
        # add leading zeros to get to equal lengths
        max_len = max([len(d) for d in digits_arr])
        for i in range(len(digits_arr)):
            digits_arr[i] = [0]*(max_len-len(digits_arr[i])) + digits_arr[i]
        return vrepr2num(np.array(sign_arr), np.array(digits_arr))
    else:
        # errored at first step
        return None


class Prompt(nn.Module):
    def __init__(self, length=2, embed_dim=768, embedding_key='mean', prompt_init='uniform', prompt_pool=False,
                 prompt_key=False, pool_size=30, top_k=4, batchwise_prompt=False, prompt_key_init='uniform', wte=None):
        super().__init__()

        self.length = length
        self.embed_dim = embed_dim
        self.prompt_pool = prompt_pool
        self.embedding_key = embedding_key
        self.prompt_init = prompt_init
        self.prompt_key = prompt_key
        self.prompt_key_init = prompt_key_init
        self.pool_size = pool_size
        print(self.pool_size)
        self.top_k = top_k
        self.batchwise_prompt = batchwise_prompt
        self.wte = wte

        if self.prompt_pool:
            prompt_pool_shape = (pool_size, length, embed_dim)
            if prompt_init == 'zero':
                self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
            elif prompt_init == 'uniform':
                self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                nn.init.uniform_(self.prompt, -1, 1)

        # if using learnable prompt keys
        if prompt_key:
            key_shape = (pool_size, embed_dim)
            if prompt_key_init == 'zero':
                self.prompt = nn.Parameter(torch.zeros(key_shape), requires_grad=False)
                print('zero initialized key')

            elif prompt_key_init == 'uniform':
                self.prompt = nn.Parameter(torch.randn(key_shape), requires_grad=False)
                nn.init.uniform_(self.prompt, -5, 5)
                print('uniform initialized key')

            elif prompt_key_init == 'gaussian':
                self.prompt = nn.Parameter(torch.randn(key_shape), requires_grad=False)
                nn.init.normal_(self.prompt, mean=0.0, std=5.0)
                print('gaussian initialized key')

            elif prompt_key_init == 'text_prototype':
                self.text_prototype_linear = nn.Linear(50257, pool_size)








        else:
            # else use mean of prompt as key
            # only compatible with prompt, not prefix
            prompt_mean = torch.mean(self.prompt, dim=1)
            self.prompt_key = prompt_mean

    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm

    def forward(self, x_embed, prompt_mask=None, cls_features=None):
        out = dict()
        if self.prompt_key:  # if self.prompt_pool:
            if self.embedding_key == 'mean':
                x_embed_mean = torch.mean(x_embed, dim=1)
            elif self.embedding_key == 'max':
                x_embed_mean = torch.max(x_embed, dim=1)[0]
            elif self.embedding_key == 'mean_max':
                x_embed_mean = torch.max(x_embed, dim=1)[0] + 2 * torch.mean(x_embed, dim=1)
            elif self.embedding_key == 'cls':
                if cls_features is None:
                    x_embed_mean = torch.max(x_embed, dim=1)[0]  # B, C
                else:
                    x_embed_mean = cls_features
            else:
                raise NotImplementedError("Not supported way of calculating embedding keys!")

            if self.prompt_key_init == 'text_prototype':
                prompt_key = self.text_prototype_linear(self.wte.transpose(0, 1)).transpose(0, 1)

            else:
                prompt_key = self.prompt

            prompt_norm = self.l2_normalize(prompt_key, dim=1)  # Pool_size, C   self.prompt_key
            x_embed_norm = self.l2_normalize(x_embed_mean, dim=1)  # B, C

            similarity = torch.matmul(x_embed_norm, prompt_norm.t())  # B, Pool_size

            if prompt_mask is None:
                _, idx = torch.topk(similarity, k=self.top_k, dim=1)  # B, top_k
                if self.batchwise_prompt:
                    prompt_id, id_counts = torch.unique(idx, return_counts=True, sorted=True)
                    # In jnp.unique, when the 'size' is specified and there are fewer than the indicated number of elements,
                    # the remaining elements will be filled with 'fill_value', the default is the minimum value along the specified dimension.
                    # Unless dimension is specified, this will be flattend if it is not already 1D.
                    if prompt_id.shape[0] < self.pool_size:
                        prompt_id = torch.cat([prompt_id, torch.full((self.pool_size - prompt_id.shape[0],),
                                                                     torch.min(idx.flatten()),
                                                                     device=prompt_id.device)])
                        id_counts = torch.cat(
                            [id_counts, torch.full((self.pool_size - id_counts.shape[0],), 0, device=id_counts.device)])
                    _, major_idx = torch.topk(id_counts, k=self.top_k)  # top_k
                    major_prompt_id = prompt_id[major_idx]  # top_k
                    # expand to batch
                    idx = major_prompt_id.expand(x_embed.shape[0], -1)  # B, top_k
            else:
                idx = prompt_mask  # B, top_k

            # batched_prompt_raw = self.prompt[idx] # B, top_k, length, C

            batched_prompt_raw = prompt_key[idx]  # B, top_k, length, C
            batched_prompt_raw = batched_prompt_raw.unsqueeze(2)  # B, top_k, 1, length, C

            batch_size, top_k, length, c = batched_prompt_raw.shape
            batched_prompt = batched_prompt_raw.reshape(batch_size, top_k * length, c)  # B, top_k * length, C

            out['prompt_idx'] = idx

            # Debugging, return sim as well
            out['prompt_norm'] = prompt_norm
            out['x_embed_norm'] = x_embed_norm
            out['similarity'] = similarity

            # Put pull_constraint loss calculation inside
            batched_key_norm = prompt_norm[idx]  # B, top_k, C
            out['selected_key'] = batched_key_norm
            x_embed_norm = x_embed_norm.unsqueeze(1)  # B, 1, C
            sim = batched_key_norm * x_embed_norm  # B, top_k, C
            reduce_sim = torch.sum(sim) / x_embed.shape[0]  # Scalar

            out['reduce_sim'] = reduce_sim
        else:
            if self.prompt_init == 'zero':
                self.prompt = nn.Parameter(torch.zeros(self.length, self.embed_dim))
            elif self.prompt_init == 'uniform':
                self.prompt = nn.Parameter(torch.randn(self.length, self.embed_dim))
                nn.init.uniform_(self.prompt)
            batched_prompt = self.prompt.unsqueeze(0).expand(x_embed.shape[0], -1, -1)

        # The input with the prompt concatenated to the front. [B, prompt+token, C]
        out['total_prompt_len'] = batched_prompt.shape[1]
        out['prompted_embedding'] = torch.cat([batched_prompt, x_embed], dim=1)
        out['prompt_key'] = prompt_key  # prompt_key

        return out



class S2IPLLM(nn.Module):

    def __init__(self, input_length,span_length,output_length,enc_in=1, dec_in=1, c_out=1,
                 patch_size = 16, stride = 8, pretrained = True, gpt_layers = 6,
                 d_model = 768,prompt_length=4, trend_length = 4, seasonal_length = 2,
                 pool_size = 1000, prompt_init = 'text_prototype'):
        super(S2IPLLM, self).__init__()

        self.is_ln = 0
        self.task_name ='long_term_forecast'
        self.output_length = output_length
        self.pred_len = output_length + span_length
        self.seq_len = input_length
        self.patch_size = patch_size
        self.stride = stride
        self.d_ff = 768
        self.patch_num = (input_length - self.patch_size) // self.stride + 1
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
        self.patch_num += 1

        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        self.trend_length = trend_length
        self.seasonal_length = seasonal_length
        self.pool_size = pool_size
        self.prompt_length = prompt_length
        self.prompt_init = prompt_init

        if pretrained == True:

            self.gpt2 = GPT2Model.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True)
            self.gpt2.h = self.gpt2.h[:gpt_layers]


        else:
            print("------------------no pretrain------------------")
            self.gpt2 = GPT2Model(GPT2Config())

        for i, (name, param) in enumerate(self.gpt2.named_parameters()):
            if 'ln' in name or 'wpe' in name:  # or 'mlp' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False  # False

        if True: #self.task_name == 'long_term_forecast':

            self.in_layer = nn.Linear(patch_size * 3, d_model)
            self.out_layer = nn.Linear(int(d_model / 3 * (self.patch_num + self.prompt_length)),
                                       output_length+ span_length)

            self.prompt_pool = Prompt(length=1, embed_dim=768, embedding_key='mean', prompt_init='uniform',
                                      prompt_pool=False,
                                      prompt_key=True, pool_size=self.pool_size,
                                      top_k=self.prompt_length, batchwise_prompt=False,
                                      prompt_key_init=self.prompt_init, wte=self.gpt2.wte.weight)

            for layer in (self.gpt2, self.in_layer, self.out_layer):
                layer.cuda()
                layer.train()

    def forward(self, x_enc):  # , x_mark_enc, x_dec, x_mark_dec, mask=None
        x_enc = x_enc.transpose(1,2)
        dec_out, res = self.forecast(x_enc)  # , x_mark_enc, x_dec, x_mark_dec
        dec_out = dec_out[:, -self.output_length:, :]
        dec_out = dec_out.transpose(1,2)
        return dec_out


    def forecast(self, x_enc): # , x_mark_enc, x_dec, x_mark_dec

        B, L, M = x_enc.shape

        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        x = rearrange(x_enc, 'b l m -> (b m) l')

        def decompose(x):
            df = pd.DataFrame(x)
            trend = df.rolling(window=self.trend_length, center=True).mean().fillna(method='bfill').fillna(
                method='ffill')
            detrended = df - trend
            seasonal = detrended.groupby(detrended.index % self.seasonal_length).transform('mean').fillna(
                method='bfill').fillna(method='ffill')
            residuals = df - trend - seasonal
            combined = np.stack([trend, seasonal, residuals], axis=1)
            return combined

        decomp_results = np.apply_along_axis(decompose, 1, x.cpu().numpy())
        x = torch.tensor(decomp_results).to(self.gpt2.device)
        x = rearrange(x, 'b l c d  -> b c (d l)', c=3)
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        x = rearrange(x, 'b c n p -> b n (c p)', c=3)
        pre_prompted_embedding = self.in_layer(x.float())

        outs = self.prompt_pool(pre_prompted_embedding)
        prompted_embedding = outs['prompted_embedding']
        # sim = outs['similarity']
        # prompt_key = outs['prompt_key']
        simlarity_loss = outs['reduce_sim']

        last_embedding = self.gpt2(inputs_embeds=prompted_embedding).last_hidden_state
        outputs = self.out_layer(last_embedding.reshape(B * M * 3, -1))

        outputs = rearrange(outputs, '(b m c) h -> b m c h', b=B, m=M, c=3)
        outputs = outputs.sum(dim=2)
        outputs = rearrange(outputs, 'b m l -> b l m')

        res = dict()
        res['simlarity_loss'] = simlarity_loss

        outputs = outputs * stdev[:, :, :M]
        outputs = outputs + means[:, :, :M]

        return outputs, res

