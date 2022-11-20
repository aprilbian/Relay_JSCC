import numpy as np
from modules import *
from utils import *

# def end2end relay system 

class RelayAFR(nn.Module):
    def __init__(self,  args, enc, dec):
        super().__init__()

        self.c_feat = args.cfeat
        self.c_out = args.cout

        self.sr_link = args.sr_link
        self.sd_link = args.sd_link
        self.rd_link = args.rd_link
        self.sr_rng = args.sr_rng
        self.sd_rng = args.sd_rng
        self.rd_rng = args.rd_rng
        self.args = args

        self.enc = enc                      # Source encoder
        self.dec = dec                      # Source decoder

    def complex_sig(self, shape, device):
        sig_real = torch.randn(*shape)
        sig_imag = torch.randn(*shape)
        return (torch.complex(sig_real, sig_imag)/np.sqrt(2)).to(device)

    def pwr_normalize(self, sig):
        _, num_ele = sig.shape[0], torch.numel(sig[0])
        pwr_sig = torch.sum(torch.abs(sig)**2, dim=-1)/num_ele
        sig = sig/torch.sqrt(pwr_sig.unsqueeze(-1))

        return sig


    def r2d_link(self, sig_sr, snr_rd):
        '''Relay to Destination'''
        batch_sz, sig_len = sig_sr.shape
        device = sig_sr.device

        sig_sr = self.pwr_normalize(sig_sr)

        noise_rd = self.complex_sig([batch_sz, sig_len], device)
        noise_pwr = 10**(-snr_rd/10)

        if self.args.channel_mode == 'fading':
            h_rd = self.complex_sig([batch_sz, 1], device)
            y_rd = h_rd*sig_sr + torch.sqrt(noise_pwr)*noise_rd
        else:
            y_rd = sig_sr + torch.sqrt(noise_pwr)*noise_rd
            h_rd = 1

        return y_rd, h_rd



    def forward(self, img, is_train):
        # img: (B,3,H,W); flag: training or not
        device = img.device

        # channel snr settings
        if self.args.adapt and is_train:
            snr_sd = self.sd_link + self.sd_rng*(2*torch.rand(1)-1).to(device)
            snr_sr = self.sr_link + self.sr_rng*(2*torch.rand(1)-1).to(device)
            snr_rd = self.rd_link + self.rd_rng*(2*torch.rand(1)-1).to(device)
        else:
            snr_sd = self.sd_link + self.sd_rng*torch.tensor([0]).to(device)
            snr_sr = self.sr_link + self.sr_rng*torch.tensor([0]).to(device)
            snr_rd = self.rd_link + self.rd_rng*torch.tensor([0]).to(device)

        snr_comb = torch.cat((snr_sd, snr_sr, snr_rd)).unsqueeze(0)                     # [1,3]

        # Source node
        x = self.enc(img, snr_comb)
        B,C,H,W = x.shape

        sig_s = x.view(B,-1,2)
        sig_s = torch.view_as_complex(sig_s)
        sig_s = self.pwr_normalize(sig_s)

        # assume quasi-static channel
        coef_shape = [B, 1]
        noise_shape = [B, int(C*H*W/2)]

        # S->D
        noise_sd = self.complex_sig(noise_shape, device)
        noise_sd_pwr = 10**(-snr_sd/10)
        if self.args.channel_mode == 'fading':
            h_sd = self.complex_sig(coef_shape, device)
            y_sd = h_sd*sig_s + torch.sqrt(noise_sd_pwr)*noise_sd
        else:
            y_sd = sig_s + torch.sqrt(noise_sd_pwr)*noise_sd


        # S->R
        noise_sr = self.complex_sig(noise_shape, device)
        noise_sr_pwr = 10**(-snr_sr/10)
        if self.args.channel_mode == 'fading':
            h_sr = self.complex_sig(coef_shape, device)
            y_sr = h_sr*sig_s + torch.sqrt(noise_sr_pwr)*noise_sr 
            E_pwr_sr = torch.sqrt(torch.abs(h_sr)**2 + noise_sr_pwr)  # norm coef
        else:
            y_sr = sig_s + torch.sqrt(noise_sr_pwr)*noise_sr 
            E_pwr_sr = torch.sqrt(1 + noise_sr_pwr)  # norm coef
        

        # R->D
        noise_rd_pwr = 10**(-snr_rd/10)
        y_rd, h_rd  = self.r2d_link(y_sr, snr_rd)         # (B, C*H*W/2)

        ##### Receiver
        alpha = 1/E_pwr_sr
        # MRC
        if self.args.channel_mode == 'fading':
            if self.args.is_coop:
                y_comb = torch.conj(h_sr*h_rd)*y_rd*alpha/(noise_rd_pwr+alpha**2*torch.abs(h_rd)**2*noise_sr_pwr) + torch.conj(h_sd)*y_sd/noise_sd_pwr
                info_coef = torch.conj(h_sd)*h_sd/noise_sd_pwr + torch.conj(h_sr*h_rd)*h_sr*h_rd*alpha**2/(noise_rd_pwr+alpha**2*torch.abs(h_rd)**2*noise_sr_pwr)       # y_comb = info_coef*x + noise_comb
                y_comb = y_comb/info_coef.real
            else:
                y_comb = torch.conj(h_sd)*y_sd/(torch.abs(h_sd)**2+noise_sd_pwr)
        else:
            if self.args.is_coop:
                y_comb = alpha*y_rd/(alpha**2*noise_sr_pwr + 10**(-snr_rd/10)) + y_sd/noise_sd_pwr
                info_coef = 1/noise_sd_pwr + alpha**2/(alpha**2*noise_sr_pwr + 10**(-snr_rd/10))        # y_comb = info_coef*x + noise_comb
                y_comb = y_comb/info_coef
            else:
                y_comb = y_sd
        
        y_comb = torch.view_as_real(y_comb).view(B,C,H,W)

        output = self.dec(y_comb, snr_comb)

        return output




class RelayPFR(nn.Module):
    def __init__(self,  args, enc, dec, parity_enc):
        super().__init__()

        self.c_feat = args.cfeat
        self.c_out = args.cout

        self.sd_link = args.sd_link
        self.sr_link = args.sr_link
        self.rd_link = args.rd_link
        self.sr_rng = args.sr_rng
        self.sd_rng = args.sd_rng
        self.rd_rng = args.rd_rng
        self.args = args

        self.enc = enc                      # Source encoder
        self.parity_enc = parity_enc        # generate parity symbols
        self.dec = dec                      # Source decoder

    def complex_sig(self, shape, device):
        sig_real = torch.randn(*shape)
        sig_imag = torch.randn(*shape)
        return (torch.complex(sig_real, sig_imag)/np.sqrt(2)).to(device)

    def pwr_normalize(self, sig):
        _, num_ele = sig.shape[0], torch.numel(sig[0])
        pwr_sig = torch.sum(torch.abs(sig)**2, dim=-1)/num_ele
        sig = sig/torch.sqrt(pwr_sig.unsqueeze(-1))

        return sig



    def forward(self, img, is_train):
        # img: (B,3,H,W)
        device = img.device

        # channel snr settings
        if self.args.adapt and is_train:
            snr_sd = self.sd_link + self.sd_rng*(2*torch.rand(1)-1).to(device)
            snr_sr = self.sr_link + self.sr_rng*(2*torch.rand(1)-1).to(device)
            snr_rd = self.rd_link + self.rd_rng*(2*torch.rand(1)-1).to(device)
        else:
            snr_sd = self.sd_link + self.sd_rng*torch.tensor([0]).to(device)
            snr_sr = self.sr_link + self.sr_rng*torch.tensor([0]).to(device)
            snr_rd = self.rd_link + self.rd_rng*torch.tensor([0]).to(device)

        snr_comb = torch.cat((snr_sd, snr_sr, snr_rd)).unsqueeze(0)                     # [1,3]

        # Source node
        x = self.enc(img, snr_comb)
        B,C,H,W = x.shape

        sig_s1 = x.view(B,-1,2)
        sig_s1 = torch.view_as_complex(sig_s1)
        sig_s1 = self.pwr_normalize(sig_s1)

        # assume quasi-static channel
        coef_shape = [B, 1]
        noise_shape = [B, int(C*H*W/2)]

        # S->R
        noise_sr = self.complex_sig(noise_shape, device)
        noise_sr_pwr = 10**(-snr_sr/10)
        if self.args.channel_mode == 'fading':
            h_sr = self.complex_sig(coef_shape, device)
            y_sr = h_sr*sig_s1 + torch.sqrt(noise_sr_pwr)*noise_sr
            y_sr = torch.conj(h_sr)*y_sr/(torch.abs(h_sr)**2+noise_sr_pwr)
        else:
            y_sr = sig_s1 + torch.sqrt(noise_sr_pwr)*noise_sr
        
        # R->D
        y_sr = torch.view_as_real(y_sr).view(B,C,H,W)
        x2 = self.parity_enc(y_sr, snr_comb)
        
        sig_sr = x2.view(B,-1,2)
        sig_sr = torch.view_as_complex(sig_sr)
        sig_sr = self.pwr_normalize(sig_sr)

        noise_rd = self.complex_sig(noise_shape, device)
        noise_rd_pwr = 10**(-snr_rd/10)

        if self.args.channel_mode == 'fading':
            h_rd = self.complex_sig(coef_shape, device)
            y_rd = h_rd*sig_sr + torch.sqrt(noise_rd_pwr)*noise_rd
            y_rd = torch.conj(h_rd)*y_rd/(torch.abs(h_rd)**2+noise_rd_pwr)
        else:
            h_rd = 1
            y_rd = sig_sr + torch.sqrt(noise_rd_pwr)*noise_rd
        

        # S->D
        noise_sd = self.complex_sig(noise_shape, device)
        noise_sd_pwr = 10**(-snr_sd/10)
        if self.args.channel_mode == 'fading':
            h_sd = self.complex_sig(coef_shape, device)
            y_sd = h_sd*sig_s1 + torch.sqrt(noise_sd_pwr)*noise_sd
            y_sd = torch.conj(h_sd)*y_sd/(torch.abs(h_sd)**2+noise_sd_pwr)
        else:
            y_sd = sig_s1 + torch.sqrt(noise_sd_pwr)*noise_sd


        ### Receiver
        y_sd, y_rd = torch.view_as_real(y_sd).view(B,C,H,W), torch.view_as_real(y_rd).view(B,C,H,W)
        y_comb = torch.cat((y_sd, y_rd), dim=1)

        output = self.dec(y_comb, snr_comb)

        return output


class RelayPFR1(nn.Module):
    def __init__(self,  args, enc, dec, parity_enc):
        super().__init__()

        self.c_feat = args.cfeat
        self.c_out = args.cout

        self.sd_link = args.sd_link
        self.sr_link = args.sr_link
        self.rd_link = args.rd_link
        self.sr_rng = args.sr_rng
        self.sd_rng = args.sd_rng
        self.rd_rng = args.rd_rng
        self.args = args

        self.enc = enc                      # Source encoder
        self.parity_enc = parity_enc        # generate parity symbols
        self.dec = dec                      # Source decoder

    def complex_sig(self, shape, device):
        sig_real = torch.randn(*shape)
        sig_imag = torch.randn(*shape)
        return (torch.complex(sig_real, sig_imag)/np.sqrt(2)).to(device)

    def pwr_normalize(self, sig):
        _, num_ele = sig.shape[0], torch.numel(sig[0])
        pwr_sig = torch.sum(torch.abs(sig)**2, dim=-1)/num_ele
        sig = sig/torch.sqrt(pwr_sig.unsqueeze(-1))

        return sig



    def forward(self, img, is_train):
        # img: (B,3,H,W)
        device = img.device

        # channel snr settings
        if self.args.adapt and is_train:
            snr_sd = self.sd_link + self.sd_rng*(2*torch.rand(1)-1).to(device)
            snr_sr = self.sr_link + self.sr_rng*(2*torch.rand(1)-1).to(device)
            snr_rd = self.rd_link + self.rd_rng*(2*torch.rand(1)-1).to(device)
        else:
            snr_sd = self.sd_link + self.sd_rng*torch.tensor([0]).to(device)
            snr_sr = self.sr_link + self.sr_rng*torch.tensor([0]).to(device)
            snr_rd = self.rd_link + self.rd_rng*torch.tensor([0]).to(device)

        snr_comb = torch.cat((snr_sd, snr_sr, snr_rd)).unsqueeze(0)                     # [1,3]

        # Source node
        x = self.enc(img, snr_comb)
        B,C,H,W = x.shape

        sig_s1 = x.view(B,-1,2)
        sig_s1 = torch.view_as_complex(sig_s1)
        sig_s1 = self.pwr_normalize(sig_s1)

        # assume quasi-static channel
        coef_shape = [B, 1]
        noise_shape = [B, int(C*H*W/2)]

        # S->R
        noise_sr = self.complex_sig(noise_shape, device)
        noise_sr_pwr = 10**(-snr_sr/10)
        if self.args.channel_mode == 'fading':
            h_sr = self.complex_sig(coef_shape, device)
            y_sr = h_sr*sig_s1 + torch.sqrt(noise_sr_pwr)*noise_sr
            y_sr = torch.conj(h_sr)*y_sr/(torch.abs(h_sr)**2+noise_sr_pwr)
        else:
            y_sr = sig_s1 + torch.sqrt(noise_sr_pwr)*noise_sr
        
        # R->D
        x2 = torch.view_as_real(y_sr).view(B,C,H,W)
        x2 = self.parity_enc(x2, snr_comb)
        
        sig_sr = x2.view(B,-1,2)
        sig_sr = torch.view_as_complex(sig_sr)
        sig_sr = y_sr + sig_sr
        sig_sr = self.pwr_normalize(sig_sr)

        noise_rd = self.complex_sig(noise_shape, device)
        noise_rd_pwr = 10**(-snr_rd/10)

        if self.args.channel_mode == 'fading':
            h_rd = self.complex_sig(coef_shape, device)
            y_rd = h_rd*sig_sr + torch.sqrt(noise_rd_pwr)*noise_rd
            y_rd = torch.conj(h_rd)*y_rd/(torch.abs(h_rd)**2+noise_rd_pwr)
        else:
            h_rd = 1
            y_rd = sig_sr + torch.sqrt(noise_rd_pwr)*noise_rd
        

        # S->D
        noise_sd = self.complex_sig(noise_shape, device)
        noise_sd_pwr = 10**(-snr_sd/10)
        if self.args.channel_mode == 'fading':
            h_sd = self.complex_sig(coef_shape, device)
            y_sd = h_sd*sig_s1 + torch.sqrt(noise_sd_pwr)*noise_sd
            y_sd = torch.conj(h_sd)*y_sd/(torch.abs(h_sd)**2+noise_sd_pwr)
        else:
            y_sd = sig_s1 + torch.sqrt(noise_sd_pwr)*noise_sd


        ### Receiver
        y_sd, y_rd = torch.view_as_real(y_sd).view(B,C,H,W), torch.view_as_real(y_rd).view(B,C,H,W)
        y_comb = torch.cat((y_sd, y_rd), dim=1)

        output = self.dec(y_comb, snr_comb)

        return output




class RelayDF(nn.Module):
    def __init__(self,  args, enc, dec, regen):
        super().__init__()

        self.c_feat = args.cfeat
        self.c_out = args.cout

        self.sd_link = args.sd_link
        self.sr_link = args.sr_link
        self.rd_link = args.rd_link
        self.sr_rng = args.sr_rng
        self.sd_rng = args.sd_rng
        self.rd_rng = args.rd_rng
        self.args = args

        self.enc = enc                      # Source encoder
        self.regen = regen                  # Decoder + Regenerator
        self.dec = dec                      # Source decoder


    def forward(self, img, is_train):
        # img: (B,3,H,W)
        device = img.device

        # channel snr settings
        if self.args.adapt and is_train:
            snr_sd = self.sd_link + self.sd_rng*(2*torch.rand(1)-1).to(device)
            snr_sr = self.sr_link + self.sr_rng*(2*torch.rand(1)-1).to(device)
            snr_rd = self.rd_link + self.rd_rng*(2*torch.rand(1)-1).to(device)
        else:
            snr_sd = self.sd_link + self.sd_rng*torch.tensor([0]).to(device)
            snr_sr = self.sr_link + self.sr_rng*torch.tensor([0]).to(device)
            snr_rd = self.rd_link + self.rd_rng*torch.tensor([0]).to(device)

        snr_comb = torch.cat((snr_sd, snr_sr, snr_rd)).unsqueeze(0)                     # [1,3]

        # Source node
        x = self.enc(img, snr_comb)
        B,C,H,W = x.shape

        sig_s1 = x.view(B,-1,2)
        sig_s1 = torch.view_as_complex(sig_s1)
        sig_s1 = pwr_normalize(sig_s1)

        # assume quasi-static channel
        coef_shape = [B, 1]
        noise_shape = [B, int(C*H*W/2)]

        # S->R
        noise_sr = complex_sig(noise_shape, device)
        noise_sr_pwr = 10**(-snr_sr/10)
        if self.args.channel_mode == 'fading':
            h_sr = complex_sig(coef_shape, device)
            y_sr = h_sr*sig_s1 + torch.sqrt(noise_sr_pwr)*noise_sr
            y_sr = torch.conj(h_sr)*y_sr/(torch.abs(h_sr)**2+noise_sr_pwr)
        else:
            y_sr = sig_s1 + torch.sqrt(noise_sr_pwr)*noise_sr
        
        # R->D
        y_sr = torch.view_as_real(y_sr).view(B,C,H,W)
        x2, regen = self.regen(y_sr, snr_comb)
        
        sig_sr = x2.view(B,-1,2)
        sig_sr = torch.view_as_complex(sig_sr)
        sig_sr = pwr_normalize(sig_sr)

        noise_rd = complex_sig(noise_shape, device)
        noise_rd_pwr = 10**(-snr_rd/10)

        if self.args.channel_mode == 'fading':
            h_rd = complex_sig(coef_shape, device)
            y_rd = h_rd*sig_sr + torch.sqrt(noise_rd_pwr)*noise_rd
            y_rd = torch.conj(h_rd)*y_rd/(torch.abs(h_rd)**2+noise_rd_pwr)
        else:
            h_rd = 1
            y_rd = sig_sr + torch.sqrt(noise_rd_pwr)*noise_rd
        

        # S->D
        noise_sd = complex_sig(noise_shape, device)
        noise_sd_pwr = 10**(-snr_sd/10)
        if self.args.channel_mode == 'fading':
            h_sd = complex_sig(coef_shape, device)
            y_sd = h_sd*sig_s1 + torch.sqrt(noise_sd_pwr)*noise_sd
            y_sd = torch.conj(h_sd)*y_sd/(torch.abs(h_sd)**2+noise_sd_pwr)
        else:
            y_sd = sig_s1 + torch.sqrt(noise_sd_pwr)*noise_sd


        ### Receiver
        y_sd, y_rd = torch.view_as_real(y_sd).view(B,C,H,W), torch.view_as_real(y_rd).view(B,C,H,W)
        y_comb = torch.cat((y_sd, y_rd), dim=1)

        output = self.dec(y_comb, snr_comb)

        return output, regen