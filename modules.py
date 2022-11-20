import ipdb
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch import Tensor

from compressai.layers import ResidualBlockUpsample, DenoiseBlock
from compressai.layers import ResidualBlock, ResidualBlockWithStride

from compressai.ops.parametrizers import NonNegativeParametrizer

class GDN(nn.Module):
    r"""Generalized Divisive Normalization layer.

    Introduced in `"Density Modeling of Images Using a Generalized Normalization
    Transformation" <https://arxiv.org/abs/1511.06281>`_,
    by Balle Johannes, Valero Laparra, and Eero P. Simoncelli, (2016).

    .. math::

       y[i] = \frac{x[i]}{\sqrt{\beta[i] + \sum_j(\gamma[j, i] * x[j]^2)}}

    """

    def __init__(
        self,
        in_channels: int,
        inverse: bool = False,
        beta_min: float = 1e-6,
        gamma_init: float = 0.1,
    ):
        super().__init__()

        beta_min = float(beta_min)
        gamma_init = float(gamma_init)
        self.inverse = bool(inverse)

        self.beta_reparam = NonNegativeParametrizer(minimum=beta_min)
        beta = torch.ones(in_channels)
        beta = self.beta_reparam.init(beta)
        self.beta = nn.Parameter(beta)

        self.gamma_reparam = NonNegativeParametrizer()
        gamma = gamma_init * torch.eye(in_channels)
        gamma = self.gamma_reparam.init(gamma)
        self.gamma = nn.Parameter(gamma)

    def forward(self, x: Tensor) -> Tensor:
        _, C, _, _ = x.size()

        beta = self.beta_reparam(self.beta)
        gamma = self.gamma_reparam(self.gamma)
        gamma = gamma.reshape(C, C, 1, 1)
        norm = F.conv2d(x**2, gamma, beta)

        if self.inverse:
            norm = torch.sqrt(norm)
        else:
            norm = torch.rsqrt(norm)

        out = x * norm

        return out

class AFModule(nn.Module):
    def __init__(self, c_in):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_features=c_in+3,
                      out_features=c_in),

            nn.LeakyReLU(),

            nn.Linear(in_features=c_in,
                      out_features=c_in),

            nn.Sigmoid()
        )

    def forward(self, x, snr):
        B, _, H, W = x.size()
        context = torch.mean(x, dim=(2, 3))
        snr_context = snr.repeat_interleave(B // snr.size(0), dim=0)

        # snr_context = torch.ones(B, 1, requires_grad=True).to(x.device) * snr
        context_input = torch.cat((context, snr_context), dim=1)
        atten_weights = self.layers(context_input).view(B, -1, 1, 1)
        atten_mask = torch.repeat_interleave(atten_weights, H, dim=2)
        atten_mask = torch.repeat_interleave(atten_mask, W, dim=3)
        out = atten_mask * x
        return out

class EncoderCell(nn.Module):
    def __init__(self, c_in, c_feat, c_out, attn=False):
        super().__init__()
        self.c_in = c_in
        self.c_feat = c_feat
        self.c_out = c_out

        if attn:
            self._attn_arch()
        else:
            self._regular_arch()


    def _regular_arch(self):
        self.layers = nn.ModuleDict({
            'rbws1': ResidualBlockWithStride(
                in_ch=self.c_in,
                out_ch=self.c_feat,
                stride=2),

            'rb1': ResidualBlock(
                in_ch=self.c_feat,
                out_ch=self.c_feat),

            'rbws2': ResidualBlock(
                in_ch=self.c_feat,
                out_ch=self.c_feat),

            'rb2': ResidualBlock(
                in_ch=self.c_feat,
                out_ch=self.c_feat),


            'rbws3': ResidualBlockWithStride(
                in_ch=self.c_feat,
                out_ch=self.c_feat,
                stride=2),


            'rbws4': ResidualBlock(
                in_ch=self.c_feat,
                out_ch=self.c_out),
        })
    
    def _attn_arch(self):
        self.layers = nn.ModuleDict({
            'rbws1': ResidualBlockWithStride(
                in_ch=self.c_in,
                out_ch=self.c_feat,
                stride=2),

            'rb1': ResidualBlock(
                in_ch=self.c_feat,
                out_ch=self.c_feat),

            'af1': AFModule(c_in=self.c_feat),

            'rbws2': ResidualBlock(
                in_ch=self.c_feat,
                out_ch=self.c_feat),

            'rb2': ResidualBlock(
                in_ch=self.c_feat,
                out_ch=self.c_feat),

            'af2': AFModule(c_in=self.c_feat),

            'rbws3': ResidualBlockWithStride(
                in_ch=self.c_feat,
                out_ch=self.c_feat,
                stride=2),

            'af3': AFModule(c_in=self.c_feat),

            'rbws4': ResidualBlock(
                in_ch=self.c_feat,
                out_ch=self.c_out),
            
            'af4': AFModule(c_in=self.c_out),
        })


    def forward(self, x, snr=0):

        for key in self.layers:
            if key[:2] == 'af':
                x = self.layers[key](x, snr)
            else:
                x = self.layers[key](x)

        out = x
        return out


class Parity_EncoderCell(nn.Module):
    def __init__(self, c_in, c_feat, c_out, attn = False):
        super().__init__()
        self.c_in = c_in
        self.c_feat = c_feat
        self.c_out = c_out

        if attn:
            self._attn_arch()
        else:
            self._regular_arch()


    def _regular_arch(self):
        self.layers = nn.ModuleDict({
            'dis1': ResidualBlock(
                in_ch=self.c_in,
                out_ch=self.c_feat),
            
            'dis2': ResidualBlock(
                in_ch=self.c_feat,
                out_ch=self.c_feat),

            'dis3': ResidualBlock(
                in_ch=self.c_feat,
                out_ch=self.c_feat),

            'dis4': ResidualBlock(
                in_ch=self.c_feat,
                out_ch=self.c_out),

        })
    
    def _attn_arch(self):
        self.layers = nn.ModuleDict({
            'dis1': ResidualBlock(
                in_ch=self.c_in,
                out_ch=self.c_feat),
            
            'dis2': ResidualBlock(
                in_ch=self.c_feat,
                out_ch=self.c_feat),
            
            'af1': AFModule(c_in=self.c_feat),

            'dis3': ResidualBlock(
                in_ch=self.c_feat,
                out_ch=self.c_feat),

            'dis4': ResidualBlock(
                in_ch=self.c_feat,
                out_ch=self.c_out),
            
            'af2': AFModule(c_in=self.c_out),

        })
    
    

    def forward(self, x, snr = 0):

        for key in self.layers:
            if key[:2] == 'af':
                x = self.layers[key](x, snr)
            else:
                x = self.layers[key](x)
        
        out = x
        return out


class Regenerate_Cell(nn.Module):
    def __init__(self, c_in, c_feat, c_out, attn = False):
        super().__init__()
        self.c_in = c_in
        self.c_feat = c_feat
        self.c_out = c_out

        if attn:
            self._attn_arch()
        else:
            self._regular_arch()


    def _regular_arch(self):
        self.layers = nn.ModuleDict({
            'dis1': ResidualBlock(
                in_ch=self.c_in,
                out_ch=self.c_feat),
            
            'dis2': ResidualBlock(
                in_ch=self.c_feat,
                out_ch=self.c_feat),

            'dis3': ResidualBlock(
                in_ch=self.c_feat,
                out_ch=self.c_feat),

            'dis4': ResidualBlock(
                in_ch=self.c_feat,
                out_ch=self.c_out),

        })
    
    def _attn_arch(self):
        self.layers = nn.ModuleDict({
            # The decoder
            'drb1': ResidualBlock(
                in_ch=self.c_in,
                out_ch=self.c_feat),

            'daf1': AFModule(c_in=self.c_feat),

            'drbu1': ResidualBlockUpsample(
                in_ch=self.c_feat,
                out_ch=self.c_feat,
                upsample=2),

            'daf2': AFModule(c_in=self.c_feat),

            'drb2': ResidualBlock(
                in_ch=self.c_feat,
                out_ch=self.c_feat),
            
            'daf3': AFModule(c_in=self.c_feat),

            'drbu2': ResidualBlockUpsample(
                in_ch=self.c_feat,
                out_ch=3,
                upsample=2),

            'daf4': AFModule(c_in=3),
            # The re-encoder

            'erbws1': ResidualBlockWithStride(
                in_ch=3,
                out_ch=self.c_feat,
                stride=2),


            'eaf1': AFModule(c_in=self.c_feat),

            'erb1': ResidualBlock(
                in_ch=self.c_feat,
                out_ch=self.c_feat),


            'eaf2': AFModule(c_in=self.c_feat),

            'erbws2': ResidualBlockWithStride(
                in_ch=self.c_feat,
                out_ch=self.c_feat,
                stride=2),

            'eaf3': AFModule(c_in=self.c_feat),

            'erb2': ResidualBlock(
                in_ch=self.c_feat,
                out_ch=self.c_out),
            
            'eaf4': AFModule(c_in=self.c_out),
        })
    
    

    def forward(self, x, snr = 0):

        for key in self.layers:
            if key[1:3] == 'af':
                x = self.layers[key](x, snr)
            elif key == 'erbws1':
                regen = x
                x = self.layers[key](x)
            else:
                x = self.layers[key](x)
        
        out = x
        return out, regen


class DecoderCell(nn.Module):
    def __init__(self, c_in, c_feat, c_out, attn=False):
        super().__init__()
        self.c_in = c_in
        self.c_feat = c_feat
        self.c_out = c_out

        #self.denoiser = DenoiseBlock(c_in, c_in)
        if attn:
            self._attn_arch()
        else:
            self._reduced_arch()

    def _reduced_arch(self):
        self.layers = nn.ModuleDict({
            
            'rb1': ResidualBlock(
                in_ch=self.c_in,
                out_ch=self.c_feat),

            'rbu1': ResidualBlock(
                in_ch=self.c_feat,
                out_ch=self.c_feat),

            'rb2': ResidualBlock(
                in_ch=self.c_feat,
                out_ch=self.c_feat),

            'rbu2': ResidualBlockUpsample(
                in_ch=self.c_feat,
                out_ch=self.c_feat,
                upsample=2),

            'rb3': ResidualBlock(
                in_ch=self.c_feat,
                out_ch=self.c_feat),

            'rbu4': ResidualBlockUpsample(
                in_ch=self.c_feat,
                out_ch=self.c_out,
                upsample=2),
        })

    def _attn_arch(self):
        self.layers = nn.ModuleDict({

            'rb1': ResidualBlock(
                in_ch=self.c_in,
                out_ch=self.c_feat),

            'rbu1': ResidualBlock(
                in_ch=self.c_feat,
                out_ch=self.c_feat),

            'af1': AFModule(c_in=self.c_feat),

            'rb2': ResidualBlock(
                in_ch=self.c_feat,
                out_ch=self.c_feat),

            'rbu2': ResidualBlockUpsample(
                in_ch=self.c_feat,
                out_ch=self.c_feat,
                upsample=2),

            'af2': AFModule(c_in=self.c_feat),

            'rb3': ResidualBlock(
                in_ch=self.c_feat,
                out_ch=self.c_feat),
            
            'af3': AFModule(c_in=self.c_feat),

            'rbu4': ResidualBlockUpsample(
                in_ch=self.c_feat,
                out_ch=self.c_out,
                upsample=2),

            'af4': AFModule(c_in=self.c_out),

        })

    def forward(self, x, snr=0):
        
        #x = self.denoiser(x)
        for key in self.layers:
            if key[:2] == 'af':
                x = self.layers[key](x, snr)
            else:
                x = self.layers[key](x)

        out = x
        return out

class MergeCell(nn.Module):
    # DF mode, merge 2 inputs
    def __init__(self, c_in, c_feat, c_out):
        super().__init__()
        self.c_in = c_in
        self.c_feat = c_feat
        self.c_out = c_out

        self.arch()

    def arch(self):
        self.layers = nn.ModuleDict({

            'rb1': ResidualBlock(
                in_ch=self.c_in,
                out_ch=self.c_feat),

            'rb2': ResidualBlock(
                in_ch=self.c_feat,
                out_ch=self.c_feat),

            'rb3': ResidualBlock(
                in_ch=self.c_feat,
                out_ch=self.c_out),
        })

    def forward(self, x):

        for key in self.layers:
            x = self.layers[key](x)

        out = x
        return out



class DenCell(nn.Module):
    # DF mode
    def __init__(self, c_in, c_feat, c_out, attn=False):
        super().__init__()
        self.c_in = c_in
        self.c_feat = c_feat
        self.c_out = c_out

        #self.denoiser = DenoiseBlock(c_in, c_out)
        if attn:
            self.attn_arch()
        else:
            self.regular_arch()

    def regular_arch(self):
        self.layers = nn.ModuleDict({

            'rb1': ResidualBlock(
                in_ch=self.c_in,
                out_ch=self.c_feat),

            #'gdn1': GDN(in_channels=self.c_feat),
            #'bn1': nn.BatchNorm2d(self.c_feat),

            #'rb2': ResidualBlock(
            #    in_ch=self.c_feat,
            #    out_ch=self.c_feat),

            #'gdn2 ': GDN(in_channels=self.c_feat),
            #'bn2': nn.BatchNorm2d(self.c_feat),

            'rb3': ResidualBlock(
                in_ch=self.c_feat,
                out_ch=self.c_out),
        })

    def attn_arch(self):
        self.layers = nn.ModuleDict({

            'rb1': ResidualBlock(
                in_ch=self.c_in,
                out_ch=self.c_feat),

            'bn1': nn.BatchNorm2d(self.c_feat),
            #'af1': AFModule(c_in=self.c_feat),

            'rb2': ResidualBlock(
                in_ch=self.c_feat,
                out_ch=self.c_feat),

            'bn2': nn.BatchNorm2d(self.c_feat),
            #'af2': AFModule(c_in=self.c_feat),

            'rb3': ResidualBlock(
                in_ch=self.c_feat,
                out_ch=self.c_out),
        })

    def forward(self, x, snr=0):

        #skip = x
        # denoise first
        #x = self.denoiser(x)
        for key in self.layers:
            if key[:2] == 'af':
                x = self.layers[key](x, snr)
            else:
                x = self.layers[key](x)

        out = x #+ skip
        return out


class EstCell(nn.Module):
    # AF mode, determine MRC coeff
    def __init__(self, c_in, c_feat, c_out, attn = False):
        super().__init__()
        self.c_in = c_in
        self.c_feat = c_feat
        self.c_out = c_out

        if attn:
            self.attn_arch()
        else:
            self.regular_arch()

    def regular_arch(self):
        self.layers = nn.ModuleDict({

            'rbws1': ResidualBlockWithStride(
                in_ch=self.c_in,
                out_ch=self.c_feat,
                stride=2),

            'rbws2': ResidualBlockWithStride(
                in_ch=self.c_feat,
                out_ch=self.c_out,
                stride=2),

            'linear': nn.Linear(self.c_out*4, 1)
        })
    
    def attn_arch(self):
        self.layers = nn.ModuleDict({

            'rbws1': ResidualBlockWithStride(
                in_ch=self.c_in,
                out_ch=self.c_feat,
                stride=2),
            
            'af1': AFModule(c_in=self.c_feat),

            'rbws2': ResidualBlockWithStride(
                in_ch=self.c_feat,
                out_ch=self.c_out,
                stride=2),
            
            'af1': AFModule(c_in=self.c_out),

            'linear': nn.Linear(self.c_out*4, 1)
        })

    def forward(self, x, snr = 0):

        for key in self.layers:
            if key == 'linear':
                x = x.view(x.shape[0], -1)

            if key[:2] == 'af':
                x = self.layers[key](x, snr)
            else:
                x = self.layers[key](x)

        out = torch.sigmoid(x)
        return out


class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.percentage = percentage
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.best_epoch = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True

    def step(self, metrics, epoch):
        if self.patience == 0:
            return False, self.best, self.best_epoch, self.num_bad_epochs

        if self.best is None:
            self.best = metrics
            self.best_epoch = epoch
            return False, self.best, self.best_epoch, 0

        if torch.isnan(metrics):
            return True, self.best, self.best_epoch, self.num_bad_epochs

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
            self.best_epoch = epoch
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True, self.best, self.best_epoch, self.num_bad_epochs

        return False, self.best, self.best_epoch, self.num_bad_epochs

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                            best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                            best * min_delta / 100)

    def get_state_dict(self):
        state_dict = {
            'best': self.best,
            'best_epoch': self.best_epoch,
            'num_bad_epochs': self.num_bad_epochs,
        }
        return state_dict

    def load_state_dict(self, state_dict):
        self.best = state_dict['best']
        self.best_epoch = state_dict['best_epoch']
        self.num_bad_epochs = state_dict['num_bad_epochs']

    def reset(self):
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self.best_epoch = None
        self._init_is_better(self.mode, self.min_delta, self.percentage)