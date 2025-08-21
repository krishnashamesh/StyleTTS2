import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:64")
os.environ.setdefault("BITSANDBYTES_NOWELCOME", "1")
os.environ.setdefault("D_UPDATE_EVERY", "2")

import os.path as osp
import re
import sys
import yaml
import shutil
import numpy as np
import torch
import click
import warnings
import traceback
warnings.simplefilter('ignore')

# load packages
import random
import yaml
from munch import Munch
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
import librosa

from models import *
from meldataset import build_dataloader
from utils import *
from losses import *
from optimizers import build_optimizer
import time
import math

from accelerate import Accelerator
from accelerate.utils import LoggerType
from accelerate import DistributedDataParallelKwargs

from torch.utils.tensorboard import SummaryWriter

try:
    from torch.backends.cuda import sdp_kernel
    sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False)
except Exception:
    pass

import atexit, faulthandler, signal, sys, os, time, threading, subprocess, logging

import gc
import json
from collections import Counter

from accelerate.logging import get_logger
logger = get_logger(__name__, log_level="INFO")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@click.command()
@click.option('-p', '--config_path', default='Configs/config.yml', type=str)
def main(config_path):
    config = yaml.safe_load(open(config_path))

    log_dir = config['log_dir']
    
    _redirect_io(log_dir)
    _print_last_oom()

    _start_logger_auto_flush(logger.logger)
    _install_signal_handlers(logger.logger)

    if not osp.exists(log_dir): os.makedirs(log_dir, exist_ok=True)
    shutil.copy(config_path, osp.join(log_dir, osp.basename(config_path)))
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    
    #Turn on mixed precision + TF32
    accelerator = Accelerator(project_dir=log_dir, split_batches=False, kwargs_handlers=[ddp_kwargs],device_placement=True, gradient_accumulation_steps=8, mixed_precision="bf16")    
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    # ---- Tunables via env (sane speed-leaning defaults) ----
    D_UPDATE_EVERY   = int(os.getenv("D_UPDATE_EVERY", "2"))  # update D every N G updates
    SLM_UPDATE_EVERY = int(os.getenv("SLM_UPDATE_EVERY", "2"))  # compute WavLM loss every N G updates
    # Style cap ramp
    STYLE_CAP_FINAL  = int(os.getenv("STYLE_CAP_FINAL",  "160"))
    STYLE_CAP_WARM   = int(os.getenv("STYLE_CAP_WARM",   "96"))
    TMA_RAMP_EPOCHS  = int(os.getenv("TMA_RAMP_EPOCHS",  "2"))
    # Curriculum warmup (fraction of total epochs) and start factor
    CURRIC_WARM_FRAC = float(os.getenv("CURRIC_WARM_FRAC", "0.10"))
    CURRIC_START     = float(os.getenv("CURRIC_START",     "0.90"))
    
    
    if accelerator.is_main_process:
        writer = SummaryWriter(log_dir + "/tensorboard")

    # write logs
    file_handler = logging.FileHandler(osp.join(log_dir, 'train.log'))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(levelname)s:%(asctime)s: %(message)s'))
    logger.logger.addHandler(file_handler)
    
    batch_size = config.get('batch_size', 10)
    device = accelerator.device
    
    epochs = config.get('epochs_1st', 200)
    save_freq = config.get('save_freq', 2)
    log_interval = config.get('log_interval', 10)
    saving_epoch = config.get('save_freq', 2)
    
    data_params = config.get('data_params', None)
    sr = config['preprocess_params'].get('sr', 24000)
    train_path = data_params['train_data']
    val_path = data_params['val_data']
    root_path = data_params['root_path']
    min_length = data_params['min_length']
    OOD_data = data_params['OOD_data']
    
    max_len = config.get('max_len', 200)
    
    # load data
    train_list, val_list = get_data_path_list(train_path, val_path)

    #Performance improvement
    nw = min(16, os.cpu_count())

    train_dataloader = build_dataloader(train_list,
                                        root_path,
                                        OOD_data=OOD_data,
                                        min_length=min_length,
                                        batch_size=batch_size,
                                        num_workers=nw,
                                        dataset_config={},
                                        prefetch_factor=6,
                                        persistent_workers=True,
                                        device=device)

    val_dataloader = build_dataloader(val_list,
                                      root_path,
                                      OOD_data=OOD_data,
                                      min_length=min_length,
                                      batch_size=batch_size,
                                      validation=True,
                                      num_workers=nw,
                                      device=device,
                                      prefetch_factor=6,
                                      persistent_workers=True,
                                      dataset_config={})
    
    with accelerator.main_process_first():
        # load pretrained ASR model
        ASR_config = config.get('ASR_config', False)
        ASR_path = config.get('ASR_path', False)
        text_aligner = load_ASR_models(ASR_path, ASR_config)

        # load pretrained F0 model
        F0_path = config.get('F0_path', False)
        pitch_extractor = load_F0_models(F0_path)

        # load BERT model
        from Utils.PLBERT.util import load_plbert
        BERT_path = config.get('PLBERT_dir', False)
        plbert = load_plbert(BERT_path)

    scheduler_params = {
        "max_lr": float(config['optimizer_params'].get('lr', 1e-4)),
        "pct_start": float(config['optimizer_params'].get('pct_start', 0.0)),
        "epochs": epochs,
        "steps_per_epoch": len(train_dataloader),
    }
    
    model_params = recursive_munch(config['model_params'])
    multispeaker = model_params.multispeaker
    model = build_model(model_params, text_aligner, pitch_extractor, plbert)

    best_loss = float('inf')  # best test loss
    last_val_avg = float('nan')
    loss_train_record = list([])
    loss_test_record = list([])

    loss_params = Munch(config['loss_params'])
    TMA_epoch = loss_params.TMA_epoch
    
    for k in model:
        model[k] = accelerator.prepare(model[k])
    
    train_dataloader, val_dataloader = accelerator.prepare(
        train_dataloader, val_dataloader
    )
    
    _ = [model[key].to(device) for key in model]

    class EMA:
        def __init__(self, module, decay=0.999):
            self.decay = decay
            self.m = module
            self.shadow = {k: p.detach().clone().float() for k,p in self.m.state_dict().items()}
            self.backup = None
        @torch.no_grad()
        def update(self):
            for k, p in self.m.state_dict().items():
                self.shadow[k].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)
        @torch.no_grad()
        def apply_to(self):
            self.backup = {k: p.detach().clone() for k,p in self.m.state_dict().items()}
            self.m.load_state_dict(self.shadow, strict=False)
        @torch.no_grad()
        def restore(self):
            self.m.load_state_dict(self.backup, strict=False)
            self.backup = None

    def _unwrap(m):
        return m.module if hasattr(m, "module") else m

    ema = {
        "decoder":       EMA(_unwrap(model["decoder"]),       decay=0.999),
        "text_encoder":  EMA(_unwrap(model["text_encoder"]),  decay=0.999),
        "style_encoder": EMA(_unwrap(model["style_encoder"]), decay=0.999),
    }

    # initialize optimizers after preparing models for compatibility with FSDP
    optimizer = build_optimizer({key: model[key].parameters() for key in model},
                                  scheduler_params_dict= {key: scheduler_params.copy() for key in model},
                               lr=float(config['optimizer_params'].get('lr', 1e-4)),
                               impl=str(config.get('optimizer_params', {}).get('impl', os.environ.get('OPTIM_IMPL', 'torch'))))
    
    for k, v in optimizer.optimizers.items():
        optimizer.optimizers[k] = accelerator.prepare(optimizer.optimizers[k])
        optimizer.schedulers[k] = accelerator.prepare(optimizer.schedulers[k])
    
    with accelerator.main_process_first():
        if config.get('pretrained_model', '') != '':
            model, optimizer, start_epoch, iters = load_checkpoint(model,  optimizer, config['pretrained_model'],
                                        load_only_params=config.get('load_only_params', True))
        else:
            start_epoch = 0
            iters = 0
    
    # in case not distributed
    try:
        n_down = model.text_aligner.module.n_down
    except:
        n_down = model.text_aligner.n_down
    
    # wrapped losses for compatibility with mixed precision
    stft_loss = MultiResolutionSTFTLoss().to(device)
    gl = GeneratorLoss(model.mpd, model.msd).to(device)
    dl = DiscriminatorLoss(model.mpd, model.msd).to(device)
    wl = WavLMLoss(model_params.slm.model, 
                   model.wd, 
                   sr, 
                   model_params.slm.sr).to(device)

    for epoch in range(start_epoch, epochs):
        # metrics: reset peak memory and init per-epoch bucket counters
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        mel_bucket_counts = Counter()
        style_bucket_counts = Counter()

        running_loss = 0
        start_time = time.time()

        acc_steps = accelerator.gradient_accumulation_steps

        _ = [model[key].train() for key in model]

        for i, batch in enumerate(train_dataloader):
            #log_print(f"Starting Batch {i}: Size = {len(batch[0])}", logger)
            waves = batch[0]
            batch = [b.to(device, non_blocking=True) for b in batch[1:]]
            texts, input_lengths, _, _, mels, mel_input_length, _ = batch
            
            with torch.no_grad():
                mask = length_to_mask(mel_input_length // (2 ** n_down)).to(device)
                text_mask = length_to_mask(input_lengths).to(texts.device)

            if epoch < TMA_epoch:
                with torch.no_grad():
                    ppgs, s2s_pred, s2s_attn = model.text_aligner(mels, mask, texts)
            else:
                ppgs, s2s_pred, s2s_attn = model.text_aligner(mels, mask, texts)

            s2s_attn = s2s_attn.transpose(-1, -2)
            s2s_attn = s2s_attn[..., 1:]
            s2s_attn = s2s_attn.transpose(-1, -2)

            with torch.no_grad():
                attn_mask = (~mask).unsqueeze(-1).expand(mask.shape[0], mask.shape[1], text_mask.shape[-1]).float().transpose(-1, -2)
                attn_mask = attn_mask.float() * (~text_mask).unsqueeze(-1).expand(text_mask.shape[0], text_mask.shape[1], mask.shape[-1]).float()
                attn_mask = (attn_mask < 1)

            s2s_attn.masked_fill_(attn_mask, 0.0)
                        
            with torch.no_grad():
                mask_ST = mask_from_lens(s2s_attn, input_lengths, mel_input_length // (2 ** n_down))
                s2s_attn_mono = maximum_path(s2s_attn, mask_ST)

            # encode
            t_en = model.text_encoder(texts, input_lengths, text_mask)

            # 50% of chance of using monotonic version
            if bool(random.getrandbits(1)):
                asr = (t_en @ s2s_attn)
            else:
                asr = (t_en @ s2s_attn_mono)
    
            # get clips

            mel_input_length_all = (
                mel_input_length if accelerator.num_processes == 1
                else accelerator.gather(mel_input_length)
            )

            effective_max_len = 160 if (epoch < TMA_epoch + 2) else max_len
            base_mel_len = min([int(mel_input_length_all.min().item() / 2 - 1), effective_max_len // 2])
            base_mel_len_st = int(mel_input_length.min().item() / 2 - 1)

            # Faster curriculum by default; env-overridable
            cf = curriculum_factor(
                epoch,
                warmup_epochs=math.ceil(epochs * CURRIC_WARM_FRAC),
                start=CURRIC_START,
                end=1.0
            )

            # Style window cap + gentle ramp after TMA starts
            if epoch < TMA_epoch:
                style_cap = STYLE_CAP_WARM
            else:
                ramp_alpha = min(1.0, (epoch - TMA_epoch + 1) / TMA_RAMP_EPOCHS)
                style_cap = int(STYLE_CAP_WARM + (STYLE_CAP_FINAL - STYLE_CAP_WARM) * ramp_alpha)

            # Decide target half-frame lengths (no forced minimums to avoid overshoot on short clips)
            target_half    = int(base_mel_len    * cf)
            target_half_st = int(base_mel_len_st * cf)
            if target_half < 2 or target_half_st < 2:
                # Batch too short for the requested windows — skip cheaply
                logger.info(f"[skip] tiny windows: target_half={target_half}, target_half_st={target_half_st}, base={base_mel_len}/{base_mel_len_st}, cf={cf:.3f}")
                continue
            # Floor-bucket WITHOUT rounding up
            mel_len    = _bucket_half_len_floor(target_half)
            mel_len_st = min(_bucket_half_len_floor(target_half_st), style_cap)
            
            # metrics: record bucketed lengths in FULL frames (mel_len values are half-frames)
            try:
                mel_full   = int(mel_len * 2)
                style_full = int(mel_len_st * 2)
                # If you already bucket upstream, these are already bucketed; we still snap to canonical bins for clean stats
                mel_b   = _nearest_bucket_full_frames(mel_full)
                style_b = _nearest_bucket_full_frames(style_full)
                mel_bucket_counts[mel_b]   += 1
                style_bucket_counts[style_b] += 1
            except Exception:
                pass

            en = []
            gt = []
            wav = []
            st = []
            wav_cpu = []
            
            for bib in range(len(mel_input_length)):

                mel_length = int(mel_input_length[bib].item() / 2)  # available half-frames
                # With floor-bucket and min-half calculation above, this is guaranteed > 0
                random_start = np.random.randint(0, mel_length - mel_len)
                en.append(asr[bib, :, random_start:random_start + mel_len])
                gt.append(mels[bib, :, (random_start * 2):((random_start + mel_len) * 2)])

                y = waves[bib][(random_start * 2) * 300:((random_start + mel_len) * 2) * 300]

                wav_cpu.append(torch.from_numpy(y))
                
                # style reference (better to be different from the GT)
                random_start  = np.random.randint(0, mel_length - mel_len_st)
                st.append(mels[bib, :, (random_start * 2):((random_start + mel_len_st) * 2)])


            en = torch.stack(en)
            gt = torch.stack(gt).detach()
            st = torch.stack(st).detach()

            wav = torch.stack(wav_cpu).float().pin_memory().to(device, non_blocking=True)

            # clip too short to be used by the style encoder
            if gt.shape[-1] < 80:
                continue
                
            with torch.no_grad():    
                real_norm = log_norm(gt.unsqueeze(1)).squeeze(1).detach()
                F0_real, _, _ = model.pitch_extractor(gt.unsqueeze(1))
                
            s = model.style_encoder(st.unsqueeze(1) if multispeaker else gt.unsqueeze(1))
            
            y_rec = model.decoder(en, F0_real, real_norm, s)
            
            # ---------- Discriminator loss (gated cadence) ----------
            d_loss_val = 0.0
            with accelerator.accumulate(model["mpd"]):  # any model key works
                if epoch >= TMA_epoch and accelerator.sync_gradients:
                    # Only update D every Nth accumulation boundary
                    do_d_update = ((iters // acc_steps) % D_UPDATE_EVERY) == 0
                    if do_d_update:
                        d_loss = dl(wav.detach().unsqueeze(1).float(), y_rec.detach()).mean()
                        accelerator.backward(d_loss / acc_steps)
                        # step both discriminators
                        optimizer.step('msd')
                        optimizer.step('mpd')
                        d_loss_val = float(d_loss.detach().item())
                    # Note: grads are cleared globally below with optimizer.zero_grad(...)
                # else: skip D forward/back entirely for real compute savings


            with accelerator.accumulate(model["decoder"]):
                # no zero_grad here; we’re accumulating
                loss_mel = stft_loss(y_rec.squeeze(), wav.detach())

                if epoch >= TMA_epoch:  # start TMA training
                    loss_s2s = 0
                    for _s2s_pred, _text_input, _text_length in zip(s2s_pred, texts, input_lengths):
                        loss_s2s += F.cross_entropy(_s2s_pred[:_text_length], _text_input[:_text_length])
                    loss_s2s /= texts.size(0)

                    loss_mono    = F.l1_loss(s2s_attn, s2s_attn_mono) * 10
                    loss_gen_all = gl(wav.detach().unsqueeze(1).float(), y_rec).mean()
                    # Intermittent WavLM feature loss to save compute
                    if SLM_UPDATE_EVERY <= 1:
                        loss_slm = wl(wav.detach(), y_rec).mean()
                    else:
                        use_slm = ((iters // acc_steps) % SLM_UPDATE_EVERY) == 0
                        if use_slm:
                            loss_slm = wl(wav.detach(), y_rec).mean()
                        else:
                            loss_slm = torch.as_tensor(0.0, device=device)

                    g_loss = (
                        loss_params.lambda_mel  * loss_mel +
                        loss_params.lambda_mono * loss_mono +
                        loss_params.lambda_s2s  * loss_s2s +
                        loss_params.lambda_gen  * loss_gen_all +
                        loss_params.lambda_slm  * loss_slm
                    )
                else:
                    loss_s2s = 0
                    loss_mono = 0
                    loss_gen_all = 0
                    loss_slm = 0
                    g_loss = loss_mel

                running_loss += accelerator.gather(loss_mel).mean().item()

                accelerator.backward(g_loss / acc_steps)

                if accelerator.sync_gradients:  # only step/zero at the accumulation boundary
                    def _clip_and_check(name, max_norm=1.0):
                        torch.nn.utils.clip_grad_norm_(model[name].parameters(),
                                                       max_norm, error_if_nonfinite=False, foreach=False)
                        for p in model[name].parameters():
                            g = getattr(p, "grad", None)
                            if g is not None and not torch.isfinite(g).all():
                                # zero-out bad grads so they can't reach the optimizer
                                p.grad = None
                                return False
                        return True

                    # safe step wrapper (skip-on-error, keep training alive)
                    def _safe_step(name):
                        try:
                            optimizer.step(name)
                            return True
                        except Exception as e:
                            if accelerator.is_main_process:
                                log_print(f"[step-skip] {name}: {type(e).__name__}: {e}", logger)
                            # clear grads for this module so we don't accumulate garbage
                            for p in model[name].parameters():
                                if getattr(p, "grad", None) is not None:
                                    p.grad = None
                            try:
                                torch.cuda.empty_cache()
                            except Exception:
                                pass
                            return False

                    # Text/Style/Decoder
                    if _clip_and_check('text_encoder'):   _safe_step('text_encoder')
                    if _clip_and_check('style_encoder'):  _safe_step('style_encoder')
                    if _clip_and_check('decoder'):        _safe_step('decoder')
                    # Post-TMA modules
                    if epoch >= TMA_epoch:
                        if _clip_and_check('text_aligner'):    _safe_step('text_aligner')
                        if _clip_and_check('pitch_extractor'): _safe_step('pitch_extractor')

                    optimizer.zero_grad(set_to_none=True)

                    # EMA only when we actually stepped
                    ema["text_encoder"].update()
                    ema["style_encoder"].update()
                    ema["decoder"].update()
          
            d_loss = d_loss_val
            iters = iters + 1
            
            if (i+1)%log_interval == 0 and accelerator.is_main_process:
                log_print ('Epoch [%d/%d], Step [%d/%d], Mel Loss: %.5f, Gen Loss: %.5f, Disc Loss: %.5f, Mono Loss: %.5f, S2S Loss: %.5f, SLM Loss: %.5f'
                        %(epoch+1, epochs, i+1, len(train_list)//batch_size, running_loss / log_interval, loss_gen_all, d_loss, loss_mono, loss_s2s, loss_slm), logger)
                
                writer.add_scalar('train/mel_loss', running_loss / log_interval, iters)
                writer.add_scalar('train/gen_loss', loss_gen_all, iters)
                writer.add_scalar('train/d_loss', d_loss, iters)
                writer.add_scalar('train/mono_loss', loss_mono, iters)
                writer.add_scalar('train/s2s_loss', loss_s2s, iters)
                writer.add_scalar('train/slm_loss', loss_slm, iters)

                running_loss = 0
                
                #log_print('Time elasped:', time.time()-start_time, logger)
                log_print(f"Time elapsed: {time.time() - start_time:.2f} seconds", logger)
                # log_print(f"Ending Batch {i}: Size = {len(batch[0])}", logger)
                                
        loss_test = 0

        _ = [model[key].eval() for key in model]

        # log_print("Crossed Epoch Run", logger)

        effective_max_len = 160 if (epoch < TMA_epoch + 2) else max_len

        for _k in ema: ema[_k].apply_to()
        with torch.no_grad():
            iters_test = 0
            for batch_idx, batch in enumerate(val_dataloader):
                optimizer.zero_grad()

                waves = batch[0]
                batch = [b.to(device, non_blocking=True) for b in batch[1:]]
                texts, input_lengths, _, _, mels, mel_input_length, _ = batch

                with torch.no_grad():
                    mask = length_to_mask(mel_input_length // (2 ** n_down)).to(device)
                    ppgs, s2s_pred, s2s_attn = model.text_aligner(mels, mask, texts)

                    s2s_attn = s2s_attn.transpose(-1, -2)
                    s2s_attn = s2s_attn[..., 1:]
                    s2s_attn = s2s_attn.transpose(-1, -2)

                    text_mask = length_to_mask(input_lengths).to(texts.device)
                    attn_mask = (~mask).unsqueeze(-1).expand(mask.shape[0], mask.shape[1], text_mask.shape[-1]).float().transpose(-1, -2)
                    attn_mask = attn_mask.float() * (~text_mask).unsqueeze(-1).expand(text_mask.shape[0], text_mask.shape[1], mask.shape[-1]).float()
                    attn_mask = (attn_mask < 1)
                    s2s_attn.masked_fill_(attn_mask, 0.0)

                # encode
                t_en = model.text_encoder(texts, input_lengths, text_mask)
                
                asr = (t_en @ s2s_attn)

                # get clips

                mel_input_length_all = (
                    mel_input_length if accelerator.num_processes == 1
                    else accelerator.gather(mel_input_length)
                )
                
                # Mirror training: safe max derived from batch min length and warmup cap
                base_mel_len = min([int(mel_input_length_all.min().item() / 2 - 1), effective_max_len // 2])


                cf = curriculum_factor(epoch, warmup_epochs=math.ceil(epochs*0.3333), start=0.6, end=1.0) 

                # Use the same floor-bucketing to keep shapes consistent and allocator happy
                mel_len    = max(40, _bucket_half_len_floor(int(base_mel_len * cf)))
                
                en = []
                gt = []
                wav = []

                #Batch the wave transfers (avoid per-item .to(device))
                wav_cpu = []

                for bib in range(len(mel_input_length)):
                    mel_length = int(mel_input_length[bib].item() / 2)
                    # With floor-bucket and min-half calculation, randint high is guaranteed > 0
                    random_start = np.random.randint(0, mel_length - mel_len)
                    en.append(asr[bib, :, random_start:random_start + mel_len])
                    gt.append(mels[bib, :, (random_start * 2):((random_start + mel_len) * 2)])
                    y = waves[bib][(random_start * 2) * 300:((random_start + mel_len) * 2) * 300]

                    wav_cpu.append(torch.from_numpy(y))

                wav = torch.stack(wav_cpu).float().pin_memory().to(device, non_blocking=True)

                en = torch.stack(en)
                gt = torch.stack(gt).detach()

                # --- Validation guards: too-short or too-silent windows ---
                # Too-short (frames) for downstream modules
                if gt.shape[-1] < 80:
                    continue
                # Too-silent: STFT spectral convergence can NaN when target norm ~ 0.
                # 1-second window at 24kHz -> this catches flat/silent crops.
                # Cheap RMS check per item; skip batch if any is silent.
                rms = wav.float().pow(2).mean(dim=1).sqrt()
                if (rms < 5e-5).any():
                    # Optional: uncomment to log once per batch
                    if accelerator.is_main_process:
                        log_print("[val] silent batch; skipping", logger)
                    continue
                F0_real, _, F0 = model.pitch_extractor(gt.unsqueeze(1))
                s = model.style_encoder(gt.unsqueeze(1))
                real_norm = log_norm(gt.unsqueeze(1)).squeeze(1)

                # Optional numeric guard (cheap, avoids rare NaNs from upstream components)
                F0_real  = torch.nan_to_num(F0_real, nan=0.0, posinf=0.0, neginf=0.0)
                real_norm = torch.nan_to_num(real_norm, nan=0.0, posinf=0.0, neginf=0.0)

                y_rec = model.decoder(en, F0_real, real_norm, s)

                if not torch.isfinite(y_rec).all():
                    if accelerator.is_main_process:
                        log_print("[val] non-finite model output; skipping batch", logger)
                    continue

                loss_mel = stft_loss(y_rec.squeeze(), wav.detach())
                # Skip non-finite losses to avoid poisoning the epoch average
                loss_mel_g = accelerator.gather(loss_mel).mean()
                if not torch.isfinite(loss_mel_g):
                    if accelerator.is_main_process:
                        log_print("[val] non-finite loss encountered; skipping batch", logger)
                    continue
                loss_test += loss_mel_g.item()
                iters_test += 1

        #restore training weights
        for _k in ema: ema[_k].restore()
        log_print(f"Time elapsed in the overall Epoch: {time.time() - start_time:.2f} seconds", logger)
        

        if accelerator.is_main_process:
            log_print(f"Epochs: {epoch + 1}", logger)
            val_avg = (loss_test / iters_test) if iters_test > 0 else float('nan')
            last_val_avg = val_avg
            log_print(f"Validation loss: {val_avg:.3f}\n\n\n\n", logger)
            log_print('\n\n\n', logger)

            if iters_test > 0:
                writer.add_scalar('eval/mel_loss', val_avg, epoch + 1)

            attn_image = get_image(s2s_attn[0].cpu().numpy().squeeze())
            writer.add_figure('eval/attn', attn_image, epoch)
            
            with torch.no_grad():
                for bib in range(len(asr)):
                    mel_length = int(mel_input_length[bib].item())
                    gt = mels[bib, :, :mel_length].unsqueeze(0)
                    en = asr[bib, :, :mel_length // 2].unsqueeze(0)
                                        
                    F0_real, _, _ = model.pitch_extractor(gt.unsqueeze(1))
                    F0_real = F0_real.unsqueeze(0)
                    s = model.style_encoder(gt.unsqueeze(1))
                    real_norm = log_norm(gt.unsqueeze(1)).squeeze(1)
                    
                    y_rec = model.decoder(en, F0_real, real_norm, s)
                    
                    writer.add_audio('eval/y' + str(bib), y_rec.cpu().numpy().squeeze(), epoch, sample_rate=sr)
                    if epoch == 0:
                        writer.add_audio('gt/y' + str(bib), waves[bib].squeeze(), epoch, sample_rate=sr)
                    
                    if bib >= 6:
                        break

            if epoch % saving_epoch == 0:
                if iters_test > 0 and (val_avg < best_loss):
                    best_loss = val_avg
                log_print('Saving..', logger)
                state = {
                    'net':  {key: model[key].state_dict() for key in model}, 
                    'optimizer': optimizer.state_dict(),
                    'iters': iters,
                    'val_loss': val_avg,
                    'epoch': epoch,
                }
                save_path = osp.join(log_dir, 'epoch_1st_%05d.pth' % epoch)
                torch.save(state, save_path)
                gc.collect()
                torch.cuda.empty_cache()
            

                try:
                    max_alloc_mb = (torch.cuda.max_memory_allocated() / (1024**2)) if torch.cuda.is_available() else 0.0
                    max_resv_mb  = (torch.cuda.max_memory_reserved()  / (1024**2)) if torch.cuda.is_available() else 0.0
                    logger.info(f"[mem] epoch={epoch} max_alloc={max_alloc_mb:.1f} MiB max_reserved={max_resv_mb:.1f} MiB")
                    top_mel = sorted(mel_bucket_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:5]
                    top_sty = sorted(style_bucket_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:5]
                    logger.info(f"[buckets] mel_top={top_mel} style_top={top_sty}")
                    dist_path = osp.join(log_dir, f"epoch_{epoch:05d}_buckets.json")
                    with open(dist_path, "w") as f:
                        json.dump({"mel": dict(mel_bucket_counts), "style": dict(style_bucket_counts)}, f, indent=2)
                except Exception as _e:
                    logger.warning(f"bucket/memory logging failed: {_e}")
                                
    if accelerator.is_main_process:
        log_print('Saving..', logger)
        state = {
            'net':  {key: model[key].state_dict() for key in model}, 
            'optimizer': optimizer.state_dict(),
            'iters': iters,
            'val_loss': last_val_avg,
            'epoch': epoch,
        }
        save_path = osp.join(log_dir, config.get('first_stage_path', 'first_stage.pth'))
        torch.save(state, save_path)

        gc.collect()
        torch.cuda.empty_cache()

        

# ------------------------------------------------------------------
# 1.  Redirect everything to log_dir/train_stdout.log
# ------------------------------------------------------------------
def _redirect_io(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, 'train_stdout.log')
    log_file = open(log_path, 'a', buffering=1)  # line-buffered
    sys.stdout = log_file
    sys.stderr = log_file
    faulthandler.enable(file=log_file)           # C-level faults
    print(f'\n\n=== New run @ {time.ctime()} ===', flush=True)


# ------------------------------------------------------------------
# 2.  Flush python logging every second
# ------------------------------------------------------------------
def _start_logger_auto_flush(logger, interval=1.0):
    def _flusher():
        while True:
            for h in logger.handlers:
                h.flush()
            time.sleep(interval)
    t = threading.Thread(target=_flusher, daemon=True)
    t.start()


# ------------------------------------------------------------------
# 3.  Save note on SIGTERM/SIGINT
# ------------------------------------------------------------------
def _install_signal_handlers(logger):
    def _handler(signum, frame):
        logger.error(f'Received signal {signum} – shutting down.')
        for h in logger.handlers:
            h.flush()
        sys.exit(1)
    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT,  _handler)


# ------------------------------------------------------------------
# 4.  At next start, show last OOM killer message (if any)
# ------------------------------------------------------------------
def _print_last_oom():
    try:
        dmesg = subprocess.check_output(['dmesg', '-T', '-l', 'err,crit,alert,emerg']).decode()
        lines = [l for l in dmesg.strip().split('\n') if 'Out of memory' in l or 'Killed process' in l]
        if lines:
            print('=== Possible OOM detected from previous run ===')
            print(lines[-1])
    except Exception:
        pass

def curriculum_factor(epoch, warmup_epochs=2, start=0.6, end=1.0):
    if epoch >= warmup_epochs:
        return end
    alpha = (epoch + 1) / warmup_epochs
    return start + (end - start) * alpha

BUCKETS = [192,224,256,288,320,352,384,416,448,480]
def _bucket_half_len_floor(x_half: int) -> int:
    """
    Floor-bucket in HALF-frames without ever rounding UP.
    If the target is below the smallest bucket, just return the target.
    """
    full = int(x_half * 2)
    if full <= 2:
        return 1
    # Below smallest bucket → do not bucket (avoid overshoot)
    if full < BUCKETS[0]:
        return full // 2
    for b in reversed(BUCKETS):
        if b <= full:
            return b // 2  # return in half-frames
    return BUCKETS[0] // 2

def _nearest_bucket_full_frames(full_frames: int) -> int:
    # Clamp and snap to nearest canonical length (frames)
    full_frames = max(64, min(BUCKETS[-1], int(full_frames)))
    return min(BUCKETS, key=lambda b: abs(b - full_frames))

if __name__=="__main__":
    try:
        main()
    except Exception:
        try:
            logger.error("Unhandled exception occurred:", exc_info=True)
        except Exception:
            log_print("Unhandled exception occurred:")
            traceback.print_exc()
