import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
from transformers import AutoModel
from contextlib import nullcontext

class SpectralConvergengeLoss(torch.nn.Module):
    """Spectral convergence loss module."""

    def __init__(self):
        """Initilize spectral convergence loss module."""
        super(SpectralConvergengeLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.
        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
        Returns:
            Tensor: Spectral convergence loss value.
        """
        return torch.norm(y_mag - x_mag, p=1) / torch.norm(y_mag, p=1)

class STFTLoss(torch.nn.Module):
    """STFT loss module."""

    def __init__(self, fft_size=1024, shift_size=120, win_length=600, window=torch.hann_window):
        """Initialize STFT loss module."""
        super(STFTLoss, self).__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.to_mel = torchaudio.transforms.MelSpectrogram(sample_rate=24000, n_fft=fft_size, win_length=win_length, hop_length=shift_size, window_fn=window)

        self.spectral_convergenge_loss = SpectralConvergengeLoss()

    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Spectral convergence loss value.
            Tensor: Log STFT magnitude loss value.
        """
        x_mag = self.to_mel(x)
        mean, std = -4, 4
        x_mag = (torch.log(1e-5 + x_mag) - mean) / std
        
        y_mag = self.to_mel(y)
        mean, std = -4, 4
        y_mag = (torch.log(1e-5 + y_mag) - mean) / std
        
        sc_loss = self.spectral_convergenge_loss(x_mag, y_mag)    
        return sc_loss


class MultiResolutionSTFTLoss(torch.nn.Module):
    """Multi resolution STFT loss module."""

    def __init__(self,
                 fft_sizes=[1024, 2048, 512],
                 hop_sizes=[120, 240, 50],
                 win_lengths=[600, 1200, 240],
                 window=torch.hann_window):
        """Initialize Multi resolution STFT loss module.
        Args:
            fft_sizes (list): List of FFT sizes.
            hop_sizes (list): List of hop sizes.
            win_lengths (list): List of window lengths.
            window (str): Window function type.
        """
        super(MultiResolutionSTFTLoss, self).__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.stft_losses = torch.nn.ModuleList()
        for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses += [STFTLoss(fs, ss, wl, window)]

    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Multi resolution spectral convergence loss value.
            Tensor: Multi resolution log STFT magnitude loss value.
        """
        sc_loss = 0.0
        for f in self.stft_losses:
            sc_l = f(x, y)
            sc_loss += sc_l
        sc_loss /= len(self.stft_losses)

        return sc_loss
    
    
@torch.no_grad()
def _lastdim_chunk_bounds(L: int, max_chunks: int = 4):
    """
    Split a length L into <= max_chunks nearly-equal segments.
    Always returns at least one chunk.
    """
    max_chunks = max(1, int(max_chunks))
    if L <= 0:
        return [(0, 0)]
    n_chunks = min(max_chunks, L)
    base = L // n_chunks
    rem  = L %  n_chunks
    bounds = []
    start = 0
    for i in range(n_chunks):
        seg = base + (1 if i < rem else 0)
        bounds.append((start, start + seg))
        start += seg
    return bounds

# memory-safe feature matching:
# - stream across last dimension in chunks
# - accumulate abs-diff in fp32 and normalize by total elements
def feature_loss(fmap_r, fmap_g, max_chunks: int = 8):
    total_loss = 0.0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            # shape can be [B,C,T] (MSD/MPD) or [B,C,H,W]; we chunk along the last dim
            assert rl.shape == gl.shape, "Feature map shapes must match"
            numel  = rl.numel()
            last_L = rl.shape[-1]

            # stream over last dimension
            part_sum = rl.new_tensor(0.0, dtype=torch.float32)
            for s, e in _lastdim_chunk_bounds(last_L, max_chunks=max_chunks):
                r = rl[..., s:e]
                g = gl[..., s:e]
                # accumulate in fp32 for numerical stability in bf16 runs
                part_sum += (g - r).abs().float().sum()

            # mean over all elements (same scale as original code)
            total_loss = total_loss + (part_sum / float(numel))

    return total_loss * 2.0


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1-dr)**2)
        g_loss = torch.mean(dg**2)
        loss += (r_loss + g_loss)
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1-dg)**2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses

""" https://dl.acm.org/doi/abs/10.1145/3573834.3574506 """
def discriminator_TPRLS_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        tau = 0.04
        m_DG = torch.median((dr-dg))
        L_rel = torch.mean((((dr - dg) - m_DG)**2)[dr < dg + m_DG])
        loss += tau - F.relu(tau - L_rel)
    return loss

def generator_TPRLS_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    for dg, dr in zip(disc_real_outputs, disc_generated_outputs):
        tau = 0.04
        m_DG = torch.median((dr-dg))
        L_rel = torch.mean((((dr - dg) - m_DG)**2)[dr < dg + m_DG])
        loss += tau - F.relu(tau - L_rel)
    return loss

class GeneratorLoss(torch.nn.Module):

    def __init__(self, mpd, msd, fm_max_chunks: int = 8,
                 amp_mode: str = "bf16", amp_enabled: bool = True):
        super(GeneratorLoss, self).__init__()
        # UNWRAPPED modules are passed in from train_first.py (6b)
        self.mpd = mpd
        self.msd = msd
        self.fm_max_chunks = int(fm_max_chunks)
        self.amp_mode = str(amp_mode).lower()
        self.amp_enabled = bool(amp_enabled)
        # Best effort: many HifiGAN-style Ds expose a 'discriminators' list
        self._mpd_list = getattr(self.mpd, "discriminators", None)
        self._msd_list = getattr(self.msd, "discriminators", None)

    def _amp_ctx(self):
        if (not self.amp_enabled) or self.amp_mode == "fp32":
            return nullcontext()
        return torch.cuda.amp.autocast(
            dtype=(torch.float16 if self.amp_mode == "fp16" else torch.bfloat16)
        )


    def _split_forward_one(self, d, y, y_hat):
        """Run a single discriminator: real (no-grad) then fake (with grads)."""
        with torch.inference_mode():
            with self._amp_ctx():
                y_r, fmap_r = d(y)

        # G-step: we don't update D, so don't build weight grads for D.
        # This keeps gradients flowing to y_hat, but avoids grad state for D's weights.
        req = []
        for p in d.parameters():
            req.append(p.requires_grad)
            if p.requires_grad:
                p.requires_grad_(False)
        try:
            with self._amp_ctx():
                y_g, fmap_g = d(y_hat)
        finally:
            for p, r in zip(d.parameters(), req):
                if p.requires_grad != r:
                    p.requires_grad_(r)

        return y_r, y_g, fmap_r, fmap_g
        
    def forward(self, y, y_hat):

        total = 0.0
        # MPD (streamed)
        if self._mpd_list is not None:
            for d in self._mpd_list:
                y_r, y_g, fr, fg = self._split_forward_one(d, y, y_hat)
                total = total + generator_loss([y_g])[0] \
                              + generator_TPRLS_loss([y_r], [y_g]) \
                              + feature_loss([fr], [fg], max_chunks=self.fm_max_chunks)
        else:
            y_r_list, y_g_list, fr_list, fg_list = self.mpd(y, y_hat)
            total = total + generator_loss(y_g_list)[0] \
                          + generator_TPRLS_loss(y_r_list, y_g_list) \
                          + feature_loss(fr_list, fg_list, max_chunks=self.fm_max_chunks)
        # MSD (streamed)
        if self._msd_list is not None:
            # Optional: gate the smallest-hop branch to reduce peak VRAM
            import os
            msd_list = self._msd_list[:-1] if os.getenv("MSD_SKIP_LAST","0")=="1" and len(self._msd_list)>1 else self._msd_list
            for d in msd_list:
                y_r, y_g, fr, fg = self._split_forward_one(d, y, y_hat)
                total = total + generator_loss([y_g])[0] \
                              + generator_TPRLS_loss([y_r], [y_g]) \
                              + feature_loss([fr], [fg], max_chunks=self.fm_max_chunks)
        else:
            y_r_list, y_g_list, fr_list, fg_list = self.msd(y, y_hat)
            total = total + generator_loss(y_g_list)[0] \
                          + generator_TPRLS_loss(y_r_list, y_g_list) \
                          + feature_loss(fr_list, fg_list, max_chunks=self.fm_max_chunks)
        
        return total.mean()
    
class DiscriminatorLoss(torch.nn.Module):

    def __init__(self, mpd, msd):
        super(DiscriminatorLoss, self).__init__()
        self.mpd = mpd
        self.msd = msd
        
    def forward(self, y, y_hat):
        # MPD
        y_df_hat_r, y_df_hat_g, _, _ = self.mpd(y, y_hat)
        loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(y_df_hat_r, y_df_hat_g)
        # MSD
        y_ds_hat_r, y_ds_hat_g, _, _ = self.msd(y, y_hat)
        loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(y_ds_hat_r, y_ds_hat_g)
        
        loss_rel = discriminator_TPRLS_loss(y_df_hat_r, y_df_hat_g) + discriminator_TPRLS_loss(y_ds_hat_r, y_ds_hat_g)


        d_loss = loss_disc_s + loss_disc_f + loss_rel
        
        return d_loss.mean()
   
    
class WavLMLoss(torch.nn.Module):

    def __init__(self, model, wd, model_sr, slm_sr=16000):
        super(WavLMLoss, self).__init__()
        self.wavlm = AutoModel.from_pretrained(model)
        self.wd = wd
        self.resample = torchaudio.transforms.Resample(model_sr, slm_sr)
     
    def forward(self, wav, y_rec):
        with torch.no_grad():
            wav_16 = self.resample(wav)
            wav_embeddings = self.wavlm(input_values=wav_16, output_hidden_states=True).hidden_states
        y_rec_16 = self.resample(y_rec)
        y_rec_embeddings = self.wavlm(input_values=y_rec_16.squeeze(), output_hidden_states=True).hidden_states

        floss = 0
        for er, eg in zip(wav_embeddings, y_rec_embeddings):
            floss += torch.mean(torch.abs(er - eg))
        
        return floss.mean()
    
    def generator(self, y_rec):
        y_rec_16 = self.resample(y_rec)
        y_rec_embeddings = self.wavlm(input_values=y_rec_16, output_hidden_states=True).hidden_states
        y_rec_embeddings = torch.stack(y_rec_embeddings, dim=1).transpose(-1, -2).flatten(start_dim=1, end_dim=2)
        y_df_hat_g = self.wd(y_rec_embeddings)
        loss_gen = torch.mean((1-y_df_hat_g)**2)
        
        return loss_gen
    
    def discriminator(self, wav, y_rec):
        with torch.no_grad():
            wav_16 = self.resample(wav)
            wav_embeddings = self.wavlm(input_values=wav_16, output_hidden_states=True).hidden_states
            y_rec_16 = self.resample(y_rec)
            y_rec_embeddings = self.wavlm(input_values=y_rec_16, output_hidden_states=True).hidden_states

            y_embeddings = torch.stack(wav_embeddings, dim=1).transpose(-1, -2).flatten(start_dim=1, end_dim=2)
            y_rec_embeddings = torch.stack(y_rec_embeddings, dim=1).transpose(-1, -2).flatten(start_dim=1, end_dim=2)

        y_d_rs = self.wd(y_embeddings)
        y_d_gs = self.wd(y_rec_embeddings)
        
        y_df_hat_r, y_df_hat_g = y_d_rs, y_d_gs
        
        r_loss = torch.mean((1-y_df_hat_r)**2)
        g_loss = torch.mean((y_df_hat_g)**2)
        
        loss_disc_f = r_loss + g_loss
                        
        return loss_disc_f.mean()

    def discriminator_forward(self, wav):
        with torch.no_grad():
            wav_16 = self.resample(wav)
            wav_embeddings = self.wavlm(input_values=wav_16, output_hidden_states=True).hidden_states
            y_embeddings = torch.stack(wav_embeddings, dim=1).transpose(-1, -2).flatten(start_dim=1, end_dim=2)

        y_d_rs = self.wd(y_embeddings)
        
        return y_d_rs