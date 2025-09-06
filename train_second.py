# load packages

import os
#os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:32")
os.environ.setdefault("BITSANDBYTES_NOWELCOME", "1")

import random
import yaml
import time
import hashlib
import json
from munch import Munch
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
import librosa
import click
import shutil
import traceback
import warnings
warnings.simplefilter('ignore')

from torch.utils.checkpoint import checkpoint

import copy
import os.path as osp
from datetime import timedelta
import psutil

from meldataset import build_dataloader

from Utils.ASR.models import ASRCNN
from Utils.JDC.model import JDCNet
from Utils.PLBERT.util import load_plbert

import atexit, faulthandler, signal, sys, os, time, threading, subprocess, logging
import math
import gc

from models import *
from losses import *
from utils import *

from Modules.slmadv import SLMAdversarialLoss
from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule

from optimizers import build_optimizer

from torch import nn
from torch.nn.utils.rnn import PackedSequence


# simple fix for dataparallel that allows access to class attributes
class MyDataParallel(torch.nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)
        
import logging
from logging import StreamHandler
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = StreamHandler()
handler.setLevel(logging.INFO)
logger.addHandler(handler)

def _scalar(x):
    try:
        import torch
        if torch.is_tensor(x):
            return x.detach().float().mean().item()
    except Exception:
        pass
    try:
        return float(x)
    except Exception:
        return 0.0


import os, psutil, time, torch
from datetime import timedelta

def log_metrics(phase: str, step: int, total: int,
                metrics: dict,
                logger, global_step=None,
                batch_time=None, mem_gb=None, lr=None):
    """
    Pretty-prints and (optionally) TensorBoards a metrics dict.

    phase   : "train" | "val"
    step    : current mini-batch idx  (1-based)
    total   : total mini-batches in epoch
    metrics : {"mel": 0.52, "dur": 1.4, ...}
    """
    core = " | ".join(f"{k}:{v:.3f}" for k, v in metrics.items())
    extras = []
    if batch_time is not None:
        extras.append(f"time:{timedelta(seconds=batch_time)}")
    if mem_gb is not None:
        extras.append(f"mem:{mem_gb:.1f} GB")
    if lr is not None:
        extras.append(f"lr:{lr:.2e}")
    msg = f"{phase.capitalize()} [{step}/{total}] {core}"
    if extras:
        msg += " | " + " ".join(extras)
    logger.info(msg)



def current_mem_gb():
    if torch.cuda.is_available():
        return torch.cuda.memory_reserved() / 1e9
    # fall back to resident set size on CPU
    return psutil.Process(os.getpid()).memory_info().rss / 1e9

# --- simple mem logger using torch.cuda.mem_get_info() ---
def _log_free(tag: str):
    if torch.cuda.is_available():
        free_b, total_b = torch.cuda.mem_get_info()
        free_pct = (free_b / max(1, total_b)) * 100.0
        logger.info(f"[mem] {tag}: free={free_b/1e9:.2f}GB ({free_pct:.1f}%)")

# --- targeted allocator trim points for cudaMallocAsync ---
def _trim_cache(tag: str):
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            logger.info(f"[alloc] trim -> empty_cache() ({tag})")
    except Exception as _e:
        logger.info(f"[alloc] trim failed: {_e}")

# ---- LR scale hook is bound after optimizers are built ----
_base_lrs = None
def _apply_lr_scale(optimizer, scale: float):
    if _base_lrs is None:
        return
    for name, opt in optimizer.optimizers.items():
        blrs = _base_lrs[name]
        for pg, blr in zip(opt.param_groups, blrs):
            pg["lr"] = float(blr) * float(scale)


# -----------------------------
# EMA & GA utilities
# -----------------------------
def _unwrap(m):
    return m.module if isinstance(m, (torch.nn.DataParallel, MyDataParallel)) else m

def _get_diffusion_core(model_dict):
    """
    Returns the *EDM core* that has .sigma_data and is callable as the diffusion loss.
    Works both when wrapped in DP and when not.
    """
    diff = model_dict["diffusion"]
    diff = getattr(diff, "module", diff)                 # unwrap DP if present
    return getattr(diff, "diffusion", diff)              # get inner .diffusion if present

def _named_params(module):
    for n, p in _unwrap(module).named_parameters():
        if p.requires_grad:
            yield n, p

class EMAStore:
    def __init__(self, model_dict, modules, decay=0.999, device=None):
        self.decay = decay
        self.targets = [k for k in modules if k in model_dict]
        self.shadow = {}
        for k in self.targets:
            self.shadow[k] = {n: p.detach().clone()
                              for n, p in _named_params(model_dict[k])}
        self.device = device
        logger.info(f"[ema] init: modules={self.targets} decay={decay}")

    @torch.no_grad()
    def update(self, model_dict):
        for k in self.targets:
            mod = model_dict[k]
            for n, p in _named_params(mod):
                s = self.shadow[k][n]
                s.mul_(self.decay).add_(p.detach(), alpha=(1.0 - self.decay))

    @torch.no_grad()
    def swap_in(self, model_dict):
        """Load EMA weights into live modules; return backups to restore() later."""
        backups = {}
        for k in self.targets:
            mod = _unwrap(model_dict[k])
            backups[k] = {n: p.detach().clone() for n, p in mod.named_parameters() if p.requires_grad}
            for n, p in mod.named_parameters():
                if p.requires_grad:
                    p.copy_(self.shadow[k][n])
        logger.info(f"[ema] swapped IN for eval")
        return backups

    @torch.no_grad()
    def restore(self, model_dict, backups):
        for k, sd in backups.items():
            mod = _unwrap(model_dict[k])
            for n, p in mod.named_parameters():
                if p.requires_grad:
                    p.copy_(sd[n])
        logger.info(f"[ema] restored original weights")

def _set_requires_grad(mod, flag: bool):
    """Toggle requires_grad on a (possibly DP-wrapped) module."""
    if mod is None:
        return
    try:
        for p in _unwrap(mod).parameters():
            p.requires_grad = flag
    except Exception:
        pass


@click.command()
@click.option('-p', '--config_path', default='Configs/config.yml', type=str)
def main(config_path):
    config = yaml.safe_load(open(config_path))
    
    log_dir = config['log_dir']
    if not osp.exists(log_dir): os.makedirs(log_dir, exist_ok=True)

    _redirect_io(log_dir)
    _print_last_oom()

    _start_logger_auto_flush(logger)
    _install_signal_handlers(logger)

    shutil.copy(config_path, osp.join(log_dir, osp.basename(config_path)))

    # write logs
    file_handler = logging.FileHandler(osp.join(log_dir, 'train_second.log'))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(levelname)s:%(asctime)s: %(message)s'))
    logger.addHandler(file_handler)

    # ---- Strict FP32 numerics & stable algorithm choices ----
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    try:
        # FP32 path → prefer mem_efficient + math (flash kernels are off in FP32)
        from torch.backends.cuda import sdp_kernel
        sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=True)
    except Exception:
        pass
    logger.info("[prec] FP32 only | TF32=False | cudnn.benchmark=False | SDPA=mem_efficient")

    batch_size = config.get('batch_size', 10)
    grad_accum = int(config.get('grad_accum', 4))    

    # Stage-2 must use epochs_2nd (fallback to 'epochs' for safety)
    epochs = int(config.get('epochs_2nd', config.get('epochs', 0)))
    if epochs <= 0:
        logger.error(f"[exit] epochs_2nd/epochs not set (>0). epochs={epochs}. Nothing to do.")
        return

    save_freq = config.get('save_freq', 2)
    log_interval = int(config.get('log_interval', 10))
    saving_epoch = config.get('save_freq', 2)

    logger.info(f"[cfg] log_interval={log_interval}")

    data_params = config.get('data_params', None)
    sr = config['preprocess_params'].get('sr', 24000)
    train_path = data_params['train_data']
    val_path = data_params['val_data']
    root_path = data_params['root_path']
    min_length = data_params['min_length']
    OOD_data = data_params['OOD_data']

    max_len = config.get('max_len', 200)
    
    loss_params = Munch(config['loss_params'])
    diff_epoch = loss_params.diff_epoch
    joint_epoch = loss_params.joint_epoch
    
    optimizer_params = Munch(config['optimizer_params'])
    
    t0 = time.time()
    train_list, val_list = get_data_path_list(train_path, val_path)
    logger.info(f"[data] train items={len(train_list)} val items={len(val_list)}")

    tr_cnt, tr_bad = _summarize_speakers(train_list)
    va_cnt, va_bad = _summarize_speakers(val_list)
    log_print(f"[spk] train unique={len(tr_cnt)} bad_defaulted={tr_bad} top={tr_cnt.most_common(5)}", logger)
    log_print(f"[spk]  val  unique={len(va_cnt)} bad_defaulted={va_bad} top={va_cnt.most_common(5)}", logger)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache(); 
        gc.collect()
        _log_free("init")

    # -------------------------
    # Dataset/config signature
    # -------------------------
    def _hash_lines(lines):
        try:
            norm = [str(x).strip() for x in lines]
            return hashlib.sha1("\n".join(sorted(norm)).encode("utf-8")).hexdigest()
        except Exception:
            return "NA"
    def _hash_file(path):
        try:
            with open(path, "rb") as f:
                return hashlib.sha1(f.read()).hexdigest()
        except Exception:
            return "NA"

    manifest_hash = _hash_lines(train_list + val_list)
    config_hash   = _hash_file(config_path)
    spk_ids_train = sorted(tr_cnt.keys())
    stage2_sig = {
        "stage": 2,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "manifest_hash": manifest_hash,
        "config_hash":   config_hash,
        "train_items": len(train_list),
        "val_items":   len(val_list),
        "speakers": {
            "count": int(len(spk_ids_train)),
            "min":   int(min(spk_ids_train) if spk_ids_train else 0),
            "max":   int(max(spk_ids_train) if spk_ids_train else 0),
        },
        "parent_manifest_hash": None,
    }


    # Use CPU as much as possible
    mel_cache_dir = (data_params or {}).get("mel_cache_dir")
    ds_cfg = {"mel_cache_dir": mel_cache_dir} if mel_cache_dir else {}
    nw = min(32, os.cpu_count() or 8)

    logger.info("[data] building train dataloader")
    train_dataloader = build_dataloader(
        train_list, root_path,
        OOD_data=OOD_data, min_length=min_length,
        batch_size=batch_size,
        num_workers=nw,
        prefetch_factor=16,
        persistent_workers=True,
        dataset_config=ds_cfg,
        device=device
    )

    logger.info("[data] building val dataloader")
    val_dataloader = build_dataloader(
        val_list, root_path,
        OOD_data=OOD_data, min_length=min_length,
        batch_size=batch_size,
        validation=True,
        num_workers=nw,
        prefetch_factor=16,
        persistent_workers=True,
        dataset_config=ds_cfg,
        device=device
    )
    
    logger.info(f"[data] dataloaders ready in {time.time()-t0:.1f}s (workers={nw})")
    ref_feat = None

    # load pretrained ASR model
    ASR_config = config.get('ASR_config', False)
    ASR_path = config.get('ASR_path', False)
    text_aligner = load_ASR_models(ASR_path, ASR_config)
    
    # load pretrained F0 model
    F0_path = config.get('F0_path', False)
    pitch_extractor = load_F0_models(F0_path)
    
    # load PL-BERT model
    BERT_path = config.get('PLBERT_dir', False)
    plbert = load_plbert(BERT_path)
    
    # build model
    model_params = recursive_munch(config['model_params'])
    multispeaker = model_params.multispeaker

    t1 = time.time()
    model = build_model(model_params, text_aligner, pitch_extractor, plbert)
    logger.info(f"[build] model dict keys: {list(model.keys())}")

    _ = [model[key].to(device, non_blocking=True) for key in model]
    # param counts
    _pc = {k: sum(p.numel() for p in _unwrap(model[k]).parameters()) for k in model}
    logger.info(f"[build] params (M): " + ", ".join(f"{k}:{v/1e6:.2f}" for k,v in _pc.items()))
    logger.info(f"[build] move+flatten done in {time.time()-t1:.1f}s")

    # Optional: channels_last on decoder (helps memory BW)
    if bool(config.get("decoder_channels_last", False)):
        try:
            dec = getattr(model["decoder"], "module", model["decoder"])
            dec.to(memory_format=torch.channels_last)
            logger.info("[channels_last] applied to decoder")
        except Exception as e:
            logger.info(f"[warn] channels_last not applied: {e}")
    
    # DP OFF by default for a stable baseline; enable with STYLETTS2_USE_DP=1
    USE_DP = bool(int(os.getenv("STYLETTS2_USE_DP", "0")))
    if USE_DP:
        for key in model:
            if key not in ("mpd", "msd", "wd"):
                model[key] = MyDataParallel(model[key])
        logger.info("[dp] DataParallel enabled")
    else:
        logger.info("[dp] DataParallel disabled (baseline)")

    # Refresh cuDNN flat buffers for any RNNs after device moves/wrapping
    def _flatten_rnns(mod):
        for m in mod.modules():
            if isinstance(m, (nn.LSTM, nn.GRU, nn.RNN)):
                try:
                    m.flatten_parameters()
                except Exception:
                    pass
    for k in ("predictor", "bert_encoder", "decoder", "style_encoder", "text_encoder"):
        if k in model:
            _flatten_rnns(getattr(model, k) if hasattr(model, k) else model[k])

    # Best-effort: enable ckpt inside diffusion UNet blocks if the API exists
    try:
        _core = _get_diffusion_core(model)
        enabled = 0
        for m in _core.modules():
            if hasattr(m, "enable_gradient_checkpointing"):
                try:
                    m.enable_gradient_checkpointing()
                    enabled += 1
                except Exception:
                    pass
            elif hasattr(m, "gradient_checkpointing"):
                try:
                    m.gradient_checkpointing = True
                    enabled += 1
                except Exception:
                    pass
        if enabled > 0:
            logger.info(f"[ckpt] diffusion: enabled ckpt on {enabled} sub-modules")
    except Exception as e:
        logger.info(f"[ckpt] diffusion: no gradient-ckpt hooks found ({e})")


    # ---- EMA & GA configs ----
    ema_cfg = Munch(config.get("ema", {
        "enable": True, "decay": 0.999, "modules": ["decoder","style_encoder","predictor","predictor_encoder"],
        "start_epoch": 0, "eval": True, "update_freq": 1
    }))
    logger.info(f"[cfg] EMA: enable={ema_cfg.enable} modules={ema_cfg.modules} decay={ema_cfg.decay} start_epoch={ema_cfg.start_epoch} eval_on_val={ema_cfg.eval}")

    # ---- Adaptive GA (Stage-1 parity) ----
    BASE_GA = int(grad_accum)
    GA_MAX  = int(config.get("ga_max", 8))
    GA_MIN  = int(config.get("ga_min", 2))
    GA_START_HIGH_PCT = float(config.get("ga_start_high_pct", 140.0))  # e.g., 4 -> 6
    FREE_LOW_PCT      = float(config.get("ga_free_low_pct",   8.0))    # drop when below
    FREE_HIGH_PCT     = float(config.get("ga_free_high_pct", 18.0))    # rise when above (after patience)
    GA_GROW_PATIENCE  = int  (config.get("ga_grow_patience",  3))
    GA_GROW_COOLDOWN  = int  (config.get("ga_grow_cooldown",  4))
    start_ga = max(GA_MIN, min(GA_MAX, int(math.ceil(BASE_GA * GA_START_HIGH_PCT / 100.0))))
    D_UPDATE_EVERY    = int(os.getenv("D_UPDATE_EVERY", "1"))
    logger.info(f"[cfg] GA: base={BASE_GA} start={start_ga} range=[{GA_MIN},{GA_MAX}] "
                f"low={FREE_LOW_PCT}% high={FREE_HIGH_PCT}% patience={GA_GROW_PATIENCE} cooldown={GA_GROW_COOLDOWN} "
                f"D_UPDATE_EVERY={D_UPDATE_EVERY}")

    start_epoch = 0
    iters = 0

    load_pretrained = config.get('pretrained_model', '') != '' and config.get('second_stage_load_pretrained', False)
    log_print('load_pretrained %s ...' % load_pretrained, logger)
    
    if not load_pretrained:
        if config.get('first_stage_path', '') != '':
            first_stage_path = osp.join(log_dir, config.get('first_stage_path', 'first_stage.pth'))
            log_print('Loading the first stage model at %s ...' % first_stage_path, logger)

            # Try to read parent signature for provenance
            try:
                _parent = torch.load(first_stage_path, map_location='cpu')
                _psig = _parent.get('signature', None)
                if _psig:
                    stage2_sig["parent_manifest_hash"] = _psig.get("manifest_hash")
                    logger.info(f"[sig] parent manifest_hash={stage2_sig['parent_manifest_hash']}")
                    _pspk = _psig.get("speakers", {})
                    logger.info(f"[sig] parent speakers: count={_pspk.get('count','?')} range=[{_pspk.get('min','?')},{_pspk.get('max','?')}]")
                else:
                    logger.info("[sig] parent signature not present in first-stage checkpoint")
            except Exception as e:
                logger.info(f"[sig] failed to read parent signature: {e}")

            model, _, start_epoch, iters = load_checkpoint(model, 
                None, 
                first_stage_path,
                load_only_params=True,
                ignore_modules=['bert', 'bert_encoder', 'predictor', 'predictor_encoder', 'msd', 'mpd', 'wd', 'diffusion']) # keep starting epoch for tensorboard log

            # these epochs should be counted from the start epoch
            diff_epoch += start_epoch
            joint_epoch += start_epoch
            epochs += start_epoch
            
            model.predictor_encoder = copy.deepcopy(model.style_encoder)
        else:
            raise ValueError('You need to specify the path to the first stage model.') 

    fm_chunks = int(config.get("fm_max_chunks", 8))
    gl = GeneratorLoss(model.mpd, model.msd, fm_max_chunks=fm_chunks, amp_mode="fp32", amp_enabled=False).to(device)
    dl = DiscriminatorLoss(model.mpd, model.msd).to(device)
    wl = WavLMLoss(model_params.slm.model, 
                   model.wd, 
                   sr, 
                   model_params.slm.sr).to(device)

    gl = MyDataParallel(gl)
    dl = MyDataParallel(dl)
    wl = MyDataParallel(wl)
        
    _diff_core = _get_diffusion_core(model)
    train_sampler = DiffusionSampler(
        _diff_core,
        sampler=ADPM2Sampler(),
        sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),
        clamp=False
    )
    
    scheduler_params = {
        "max_lr": optimizer_params.lr,
        "pct_start": float(0),
        "epochs": epochs,
        "steps_per_epoch": len(train_dataloader),
    }
    scheduler_params_dict= {key: scheduler_params.copy() for key in model}
    scheduler_params_dict['bert']['max_lr'] = optimizer_params.bert_lr * 2
    scheduler_params_dict['decoder']['max_lr'] = optimizer_params.ft_lr * 2
    scheduler_params_dict['style_encoder']['max_lr'] = optimizer_params.ft_lr * 2
    
    optimizer = build_optimizer({key: model[key].parameters() for key in model},
                                scheduler_params_dict=scheduler_params_dict, lr=optimizer_params.lr)

    # Write run signature once
    try:
        with open(osp.join(log_dir, "run_signature.json"), "w") as f:
            json.dump(stage2_sig, f, indent=2)
        logger.info(f"[sig] wrote run_signature.json (manifest_hash={manifest_hash[:8]}… parent={str(stage2_sig['parent_manifest_hash'])[:8]}…)")
        if stage2_sig["parent_manifest_hash"] and stage2_sig["parent_manifest_hash"] != manifest_hash:
            logger.info("[sig] NOTE: Stage-2 dataset differs from Stage-1 (hash mismatch)")
    except Exception as e:
        logger.info(f"[sig] failed to write run_signature.json: {e}")


    # Log LR table at epoch 0
    try:
        _lr_table = {k: optimizer.optimizers[k].param_groups[0]['lr'] for k in optimizer.optimizers}
        logger.info("[lr] " + ", ".join(f"{k}:{v:.2e}" for k,v in _lr_table.items()))
    except Exception:
        pass
    
    # adjust BERT learning rate
    for g in optimizer.optimizers['bert'].param_groups:
        g['betas'] = (0.9, 0.99)
        g['lr'] = optimizer_params.bert_lr
        g['initial_lr'] = optimizer_params.bert_lr
        g['min_lr'] = 0
        g['weight_decay'] = 0.01
        
    # adjust acoustic module learning rate
    for module in ["decoder", "style_encoder"]:
        for g in optimizer.optimizers[module].param_groups:
            g['betas'] = (0.0, 0.99)
            g['lr'] = optimizer_params.ft_lr
            g['initial_lr'] = optimizer_params.ft_lr
            g['min_lr'] = 0
            g['weight_decay'] = 1e-4


    # record base LRs for GA LR scaling (Stage-1 parity)
    global _base_lrs
    _base_lrs = {name: [pg["lr"] for pg in opt.param_groups]
                 for name, opt in optimizer.optimizers.items()}
        
    # load models if there is a model
    if load_pretrained:
        model, optimizer, start_epoch, iters = load_checkpoint(model,  optimizer, config['pretrained_model'],
                                    load_only_params=config.get('load_only_params', True))
        
    n_down = model.text_aligner.n_down

    best_loss = float('inf')  # best test loss
    loss_train_record = list([])
    loss_test_record = list([])
    iters = 0
    
    criterion = nn.L1Loss() # F0 loss (regression)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        _log_free("init")
    
    gc.collect()
    
    stft_loss = MultiResolutionSTFTLoss().to(device)
    
    # log_print(f"BERT: {optimizer.optimizers['bert']}", logger)
    # log_print(f"decoder: {optimizer.optimizers['decoder']}", logger)


    start_ds = False
    
    running_std = []
    
    slmadv_params = Munch(config['slmadv_params'])
    slmadv = SLMAdversarialLoss(model, wl, train_sampler, 
                                slmadv_params.min_len, 
                                slmadv_params.max_len,
                                batch_percentage=slmadv_params.batch_percentage,
                                skip_update=slmadv_params.iter, 
                                sig=slmadv_params.sig
                               )

    logger.info(f"[cfg] SLM-ADV: iter={slmadv_params.iter} "
                f"thresh={slmadv_params.thresh} scale={slmadv_params.scale}")


    # ---- EMA store (after all modules exist) ----
    ema = None
    if ema_cfg.enable:
        ema = EMAStore(model, ema_cfg.modules, decay=ema_cfg.decay, device=device)
    
    # ---- Adversarial window cap (bounds MSD/MPD memory) ----
    ADV_MAX_SAMPLES = int(config.get("adv_max_samples", os.getenv("ADV_MAX_SAMPLES", "24000")))  # ~1s @ 24kHz
    logger.info(f"[cfg] adv_max_samples={ADV_MAX_SAMPLES}")

    def _prep_adv_pair(y_real, y_fake, max_samples: int):
        """
        Returns (real, fake) cropped & sanitised for MSD/MPD.
        Input may be [B,T] or [B,1,T]. Output is [B,1,T_crop], plus (Ttot, Tc).
        Mirrors train_first.py semantics to avoid ckpt metadata mismatch.
        """
        # → [B,T]
        if y_real.dim() == 3 and y_real.size(1) == 1: y_real = y_real[:, 0, :]
        if y_fake.dim() == 3 and y_fake.size(1) == 1: y_fake = y_fake[:, 0, :]
        if y_real.dim() != 2 or y_fake.dim() != 2:
            raise RuntimeError(f"adv pair expects [B,T] or [B,1,T]; got real={tuple(y_real.shape)} fake={tuple(y_fake.shape)}")
        Ttot = min(y_real.size(-1), y_fake.size(-1))
        Tc   = min(Ttot, int(max_samples))
        if Tc < Ttot:
            import numpy as _np
            s0 = int(_np.random.randint(0, Ttot - Tc + 1)); s1 = s0 + Tc
            # clone slices so they don't alias checkpointed storages
            y_real = y_real[:, s0:s1].clone()
            y_fake = y_fake[:, s0:s1].clone()
        else:
            # even without cropping, sever aliasing
            y_real = y_real.clone()
            y_fake = y_fake.clone()
        # out-of-place sanitize (no trailing underscores)
        y_real = torch.nan_to_num(y_real, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
        y_fake = torch.nan_to_num(y_fake, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
        return (
            y_real.unsqueeze(1).contiguous(),
            y_fake.unsqueeze(1).contiguous(),
            Ttot, Tc
        )

    # GA state
    acc_steps = start_ga
    ga_safe_count = 0
    ga_cooldown   = GA_GROW_COOLDOWN
    E_BASE = batch_size * max(1, BASE_GA)
    ga_bucket = 0

    # Final sanity check before we start loops
    logger.info(f"[loop] epochs={epochs} start_epoch={start_epoch} steps_per_epoch={len(train_dataloader)}")
    if start_epoch >= epochs:
        logger.warning(f"[exit] start_epoch ({start_epoch}) >= epochs ({epochs}). Nothing to train.")
        return

    logger.info("[train] BEGIN")
    for epoch in range(start_epoch, epochs):
        
        start_time = time.time()
        running_loss = 0
        last_d_marker = "D-"
        slm_skips = 0      # true skips after gate opens (e.g., cadence, None from slmadv)
        slm_gated = 0      # gated steps before epoch >= diff_epoch
        slm_cadence_skips = 0
        slm_fail_skips = 0

        _trim_cache("epoch-start"); 
        gc.collect()

        # reset GA to start_ga each epoch; scale LR with effective batch
        acc_steps = start_ga
        eff_scale = (batch_size * acc_steps) / max(1.0, float(E_BASE))
        _apply_lr_scale(optimizer, eff_scale)
        logger.info(f"[ga] epoch-reset -> {acc_steps} (eff_scale={eff_scale:.2f})")

        _ = [model[key].eval() for key in model]

        model.predictor.train()
        model.bert_encoder.train()
        model.bert.train()
        model.msd.train()
        model.mpd.train()


        if epoch >= diff_epoch and not start_ds:
            logger.info(f"[gate] entering diffusion phase at epoch {epoch}")
            start_ds = True
        if epoch == joint_epoch:
            logger.info(f"[gate] entering joint fine-tune phase at epoch {epoch}")

        logger.info(f"[epoch {epoch}] start ({len(train_dataloader)} steps)")
        if epoch < diff_epoch:
            logger.info(f"[slm-adv] OFF (gated) until epoch >= {diff_epoch} (zero-based). WD will not step; D marker reflects only mpd/msd.")
        else:
            logger.info("[slm-adv] ON. WD steps according to D_UPDATE_EVERY and slmadv cadence.")

        for i, batch in enumerate(train_dataloader):

            # --- lightweight timers for profiling at interval ---
            import time as _t
            _t0 = _t.perf_counter()
            dt_align = dt_text = dt_style = dt_pred = 0.0
            dt_dec_gt = dt_dec = dt_disc = dt_stft = dt_bwd = 0.0

            try:
                # ---- GA boundary bookkeeping ----
                if ga_bucket == 0:
                    # zero ONLY generator-family grads
                    for k in ("bert_encoder","bert","predictor","predictor_encoder","diffusion","style_encoder","decoder"):
                        if k in model:
                            for p in model[k].parameters():
                                if p.grad is not None: p.grad = None
                    # also clear D grads if we accumulate their grads for boundary stepping
                    for k in ("msd","mpd"):
                        if k in model:
                            for p in model[k].parameters():
                                if p.grad is not None: p.grad = None
                ga_bucket += 1
                step_boundary = (ga_bucket % max(1, acc_steps) == 0)

                waves = batch[0]
                batch = [b.to(device, non_blocking=True) for b in batch[1:]]
                texts, input_lengths, ref_texts, ref_lengths, mels, mel_input_length, ref_mels = batch

                with torch.no_grad():
                    mask = length_to_mask(mel_input_length // (2 ** n_down)).to(device)
                    mel_mask = length_to_mask(mel_input_length).to(device)
                    text_mask = length_to_mask(input_lengths).to(texts.device)
                    try:
                        _, _, s2s_attn = model.text_aligner(mels, mask, texts)
                        dt_align = _t.perf_counter() - _t0; _t1 = _t.perf_counter()
                        s2s_attn = s2s_attn.transpose(-1, -2)
                        s2s_attn = s2s_attn[..., 1:]
                        s2s_attn = s2s_attn.transpose(-1, -2)
                    except Exception:
                        continue
                    # build legal attn mask and sanitize + row-renorm
                    attn_mask = (~mask).unsqueeze(-1).expand(mask.shape[0], mask.shape[1], text_mask.shape[-1]).float().transpose(-1, -2)
                    attn_mask = attn_mask.float() * (~text_mask).unsqueeze(-1).expand(text_mask.shape[0], text_mask.shape[1], mask.shape[-1]).float()
                    attn_mask = (attn_mask < 1)
                    s2s_attn = s2s_attn.masked_fill(attn_mask, 0.0)
                    s2s_attn = torch.nan_to_num(s2s_attn, 0.0, 0.0, 0.0)
                    row_sum = s2s_attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)
                    s2s_attn = s2s_attn / row_sum
                    # monotonic fallback
                    mask_ST = mask_from_lens(s2s_attn, input_lengths, mel_input_length // (2 ** n_down))
                    s2s_attn_mono = maximum_path(s2s_attn, mask_ST).float()
                    s2s_attn_mono = torch.nan_to_num(s2s_attn_mono, 0.0, 0.0, 0.0)

                    # encode
                    t_en = model.text_encoder(texts, input_lengths, text_mask)
                    dt_text = _t.perf_counter() - _t1; _t2 = _t.perf_counter()
                    asr = (t_en @ s2s_attn_mono)

                    d_gt = s2s_attn_mono.sum(axis=-1).detach()
                    
                    # compute reference styles (always, if multispeaker)
                    if multispeaker:
                        ref_ss = model.style_encoder(ref_mels.unsqueeze(1))
                        ref_sp = model.predictor_encoder(ref_mels.unsqueeze(1))
                        ref_feat = torch.cat([ref_ss, ref_sp], dim=1)

                # compute the style of the entire utterance (no grads needed; s_trg is detached)
                # doing this under no_grad trims graph/VRAM with zero training signal change
                ss = []; gs = []
                with torch.no_grad():
                    for bib in range(len(mel_input_length)):
                        mel_length = int(mel_input_length[bib].item())
                        mel = mels[bib, :, :mel_input_length[bib]]
                        s = model.predictor_encoder(mel.unsqueeze(0).unsqueeze(1))
                        ss.append(s)
                        s = model.style_encoder(mel.unsqueeze(0).unsqueeze(1))
                        gs.append(s)

                s_dur = torch.stack(ss).squeeze(1)  # global prosodic styles
                gs = torch.stack(gs).squeeze(1) # global acoustic styles
                s_trg = torch.cat([gs, s_dur], dim=-1).detach() # ground truth for denoiser

                # BERT stack tends to be memory hungry → checkpoint both calls
                bert_dur = checkpoint(
                    lambda T, M: model.bert(T, attention_mask=M),
                    texts, (~text_mask).int()
                )
                d_en = checkpoint(lambda X: model.bert_encoder(X),
                                  bert_dur).transpose(-1, -2)
                
                # denoiser training
                if epoch >= diff_epoch:
                    num_steps = np.random.randint(3, 5)
                    
                    if model_params.diffusion.dist.estimate_sigma_data:
                        _edm = _get_diffusion_core(model)
                        sigma_val = s_trg.detach().to('cpu', copy=True).std(dim=-1).mean().item()
                        _edm.sigma_data = sigma_val
                        running_std.append(sigma_val)
                        
                    if multispeaker:
                        s_preds = train_sampler(noise = torch.randn_like(s_trg).unsqueeze(1).to(device), 
                            embedding=bert_dur,
                            embedding_scale=1,
                                    features=ref_feat, # reference from the same speaker as the embedding
                                embedding_mask_proba=0.1,
                                num_steps=num_steps).squeeze(1)
                        _edm = _get_diffusion_core(model)
                        loss_diff = _edm(s_trg.unsqueeze(1), embedding=bert_dur, features=ref_feat).mean() # EDM loss
                        loss_sty = F.l1_loss(s_preds, s_trg.detach()) # style reconstruction loss
                        del s_preds
                    else:
                        s_preds = train_sampler(noise = torch.randn_like(s_trg).unsqueeze(1).to(device), 
                            embedding=bert_dur,
                            embedding_scale=1,
                                embedding_mask_proba=0.1,
                                num_steps=num_steps).squeeze(1)                    

                        _edm = _get_diffusion_core(model)
                        loss_diff = _edm(s_trg.unsqueeze(1), embedding=bert_dur).mean()
                        loss_sty = F.l1_loss(s_preds, s_trg.detach()) # style reconstruction loss
                        del s_preds
                else:
                    loss_sty = 0
                    loss_diff = 0

                # predictor heads (duration/noise) → big enough to benefit from ckpt
                d, p = checkpoint(
                    lambda A, B, C, D, E: model.predictor(A, B, C, D, E),
                    d_en, s_dur, input_lengths, s2s_attn_mono, text_mask
                )

                # free alignments before heavy decoder/adv work
                try:
                    del s2s_attn_mono, s2s_attn, attn_mask
                except Exception:
                    pass
                
                mel_len = min(int(mel_input_length.min().item() / 2 - 1), max_len // 2)
                mel_len_st = int(mel_input_length.min().item() / 2 - 1)
                en = []
                gt = []
                st = []
                p_en = []
                
                wav_cpu = []

                for bib in range(len(mel_input_length)):
                    mel_length = int(mel_input_length[bib].item() / 2)

                    random_start = np.random.randint(0, mel_length - mel_len)
                    en.append(asr[bib, :, random_start:random_start+mel_len])
                    p_en.append(p[bib, :, random_start:random_start+mel_len])
                    gt.append(mels[bib, :, (random_start * 2):((random_start+mel_len) * 2)])
                    
                    y = waves[bib][(random_start * 2) * 300:((random_start+mel_len) * 2) * 300]
                    wav_cpu.append(torch.from_numpy(y))

                    # style reference (better to be different from the GT)
                    random_start = np.random.randint(0, mel_length - mel_len_st)
                    st.append(mels[bib, :, (random_start * 2):((random_start+mel_len_st) * 2)])
                    
                # Keep on CPU (pinned); move cropped/short-lived copies later
                wav = torch.stack(wav_cpu).float().pin_memory()

                en = torch.stack(en)
                p_en = torch.stack(p_en)
                gt = torch.stack(gt).detach()
                st = torch.stack(st).detach()
                
                if gt.size(-1) < 80:
                    continue

                s_inp = (st.unsqueeze(1) if multispeaker else gt.unsqueeze(1))
                # predictor/style encoders feed decoder → checkpoint them
                s_dur = checkpoint(lambda X: model.predictor_encoder(X), s_inp)

                dt_pred = _t.perf_counter() - _t2; _t3 = _t.perf_counter()
                s  = checkpoint(lambda X: model.style_encoder(X), s_inp)
                dt_style = _t.perf_counter() - _t3; _t4 = _t.perf_counter()
                # sanitize style vectors
                s = torch.nan_to_num(s, 0.0, 0.0, 0.0)
                
                with torch.no_grad():
                    F0_real, _, F0 = model.pitch_extractor(gt.unsqueeze(1))
                    F0 = F0.reshape(F0.shape[0], F0.shape[1] * 2, F0.shape[2], 1).squeeze()

                    asr_real = model.text_aligner.get_feature(gt)

                    N_real = log_norm(gt.unsqueeze(1)).squeeze(1)
                    
                    y_rec_gt = wav.unsqueeze(1)
                    y_rec_gt_pred = model.decoder(en, F0_real, N_real, s)
                    dt_dec_gt = _t.perf_counter() - _t4; _t5 = _t.perf_counter()

                    if epoch >= joint_epoch:
                        # ground truth from recording
                        wav = y_rec_gt # use recording since decoder is tuned
                    else:
                        # ground truth from reconstruction
                        wav = y_rec_gt_pred # use reconstruction since decoder is fixed

                F0_fake, N_fake = model.predictor.F0Ntrain(p_en, s_dur)

                # sanitize decoder inputs
                en = torch.nan_to_num(en, 0.0, 0.0, 0.0)
                F0_fake = torch.nan_to_num(F0_fake, 0.0, 0.0, 0.0)
                N_fake = torch.nan_to_num(N_fake, 0.0, 0.0, 0.0)
                
                # decoder is the main hog → checkpoint it (Stage-1 parity: clone inputs/outputs)
                en_c = en.contiguous().clone()
                f0_c = F0_fake.contiguous().clone()
                n_c  = N_fake.contiguous().clone()
                s_c  = s.contiguous().clone()
                y_rec = checkpoint(
                    lambda A, B, C, D: model.decoder(A, B, C, D),
                    en_c, f0_c, n_c, s_c
                )
                # IMPORTANT: sever any alias to checkpointed storage (output)
                y_rec = y_rec.clone().contiguous()
                try:
                    del en_c, f0_c, n_c, s_c
                except Exception:
                    pass
                
                dt_dec = _t.perf_counter() - _t5; _t6 = _t.perf_counter()
                if not torch.isfinite(y_rec).all():
                    logger.info("[train] non-finite decoder output; skipping batch")
                    continue

                loss_F0_rec =  (F.smooth_l1_loss(F0_real, F0_fake)) / 10
                loss_norm_rec = F.smooth_l1_loss(N_real, N_fake)

                # -------- Discriminator path (accumulate; step at boundary) --------
                d_loss = 0.0
                if start_ds:
                    # Enable D grads, disable G grads to avoid retaining G activations during D backward
                    for k in ("mpd", "msd"):
                        if k in model: _set_requires_grad(model[k], True)
                    for k in ("decoder","style_encoder","predictor","predictor_encoder","diffusion","bert","bert_encoder"):
                        if k in model: _set_requires_grad(model[k], False)

                    # Use fully detached, cropped, sanitised reals/fakes for D (Stage-1 parity)
                    _y_real_adv, _y_fake_adv, Ttot, Tc = _prep_adv_pair(wav.detach(), y_rec.detach(), ADV_MAX_SAMPLES)
                    _y_real_adv = _y_real_adv.to(device, non_blocking=True)

                    # --- one-liner: report chosen crop length for D path ---
                    try:
                        if ((i + 1) % max(1, log_interval) == 0):
                            logger.info(f"[adv] D: crop={int(Tc)}/{int(Ttot)} samples (cap={ADV_MAX_SAMPLES})")
                    except Exception:
                        pass

                    d_loss = dl(_y_real_adv, _y_fake_adv).mean().float()
                    if not torch.isfinite(d_loss):
                        logger.info("[skip] non-finite d_loss; skipping D backward for this batch")
                    else:
                        (d_loss / max(1, acc_steps)).backward()
                    dt_disc = _t.perf_counter() - _t6; _t7 = _t.perf_counter()

                    # Restore: disable D grads, enable G grads
                    for k in ("mpd", "msd"):
                        if k in model: _set_requires_grad(model[k], False)
                    for k in ("decoder","style_encoder","predictor","predictor_encoder","diffusion","bert","bert_encoder"):
                        if k in model: _set_requires_grad(model[k], True)


                else:
                    d_loss = 0.0

                # -------- Generator losses (accumulate) --------

                # short-lived GPU copy for STFT; Stage-1 parity squeeze(1)
                _wav_dev = wav.to(device, non_blocking=True)
                loss_mel = stft_loss(
                    y_rec.contiguous().squeeze(1).float(),
                    _wav_dev.float()
                )
                del _wav_dev

                dt_stft = _t.perf_counter() - _t7 if start_ds else _t.perf_counter() - _t6

                if start_ds:
                    # Cropped/sanitised window for G's adv/FM too (bounds MSD/MPD memory)
                    _y_real_adv_g, _y_fake_adv_g, Ttot, Tc = _prep_adv_pair(wav, y_rec, ADV_MAX_SAMPLES)
                    _y_real_adv_g = _y_real_adv_g.to(device, non_blocking=True)

                    # --- one-liner: report chosen crop length for G path ---
                    try:
                        if ((i + 1) % max(1, log_interval) == 0):
                            logger.info(f"[adv] G: crop={int(Tc)}/{int(Ttot)} samples (cap={ADV_MAX_SAMPLES})")
                    except Exception:
                        pass

                    loss_gen_all = gl(_y_real_adv_g, _y_fake_adv_g).mean()
                else:
                    loss_gen_all = 0

                # short-lived GPU copy for SLM (WavLM)
                _wav_slm = wav.to(device, non_blocking=True)

                # _wav_slm is [B,T]; keep as-is. y_rec is [B,1,T] → squeeze channel only.
                loss_lm = wl(_wav_slm.detach(), y_rec.squeeze(1)).mean()
                del _wav_slm

                loss_ce = 0
                loss_dur = 0
                for _s2s_pred, _text_input, _text_length in zip(d, (d_gt), input_lengths):
                    _s2s_pred = _s2s_pred[:_text_length, :]
                    _text_input = _text_input[:_text_length].long()
                    _s2s_trg = torch.zeros_like(_s2s_pred)
                    for p in range(_s2s_trg.shape[0]):
                        _s2s_trg[p, :_text_input[p]] = 1
                    _dur_pred = torch.sigmoid(_s2s_pred).sum(axis=1)

                    loss_dur += F.l1_loss(_dur_pred[1:_text_length-1], 
                                        _text_input[1:_text_length-1])
                    loss_ce += F.binary_cross_entropy_with_logits(_s2s_pred.flatten(), _s2s_trg.flatten())

                loss_ce /= texts.size(0)
                loss_dur /= texts.size(0)

                g_loss = loss_params.lambda_mel * loss_mel + \
                        loss_params.lambda_F0 * loss_F0_rec + \
                        loss_params.lambda_ce * loss_ce + \
                        loss_params.lambda_norm * loss_norm_rec + \
                        loss_params.lambda_dur * loss_dur + \
                        loss_params.lambda_gen * loss_gen_all + \
                        loss_params.lambda_slm * loss_lm + \
                        loss_params.lambda_sty * loss_sty + \
                        loss_params.lambda_diff * loss_diff

                running_loss += loss_mel.item()
                # Guard first: never backprop a non-finite objective
                if not torch.isfinite(g_loss):
                    logger.info("[skip] non-finite g_loss; skipping gradients for this batch")
                    continue
                (g_loss / max(1, acc_steps)).backward()
                dt_bwd = _t.perf_counter() - (_t7 if start_ds else _t6)

                # -------- Step at GA boundary --------
                if step_boundary:
                    # D cadence (optional reduction)
                    do_d_update = ((i // max(1, acc_steps)) % max(1, D_UPDATE_EVERY) == 0)
                    if start_ds and do_d_update:
                        # Clip grads & zero any non-finite grads before D step; then safe-step each D
                        def _clip_and_clean(name, max_norm=1.0):
                            try:
                                torch.nn.utils.clip_grad_norm_(model[name].parameters(), max_norm,
                                                               error_if_nonfinite=False, foreach=False)
                            except Exception as _e:
                                logger.info(f"[clip] {name} ignored ({_e})")
                            bad = False
                            for p in model[name].parameters():
                                g = getattr(p, "grad", None)
                                if g is not None and not torch.isfinite(g).all():
                                    p.grad = None
                                    bad = True
                            return not bad
                        def _safe_step(name):
                            try:
                                optimizer.step(name)
                                return True
                            except Exception as e:
                                logger.info(f"[step-skip] {name}: {type(e).__name__}: {e}")
                                for p in model[name].parameters():
                                    if getattr(p, "grad", None) is not None:
                                        p.grad = None
                                try: torch.cuda.empty_cache()
                                except Exception: pass
                                return False
                        ok_msd = _clip_and_clean('msd')
                        ok_mpd = _clip_and_clean('mpd')
                        if ok_msd: _safe_step('msd')
                        if ok_mpd: _safe_step('mpd')
                        last_d_marker = "D+"
                    else:
                        last_d_marker = "D-"
                    # G family
                    optimizer.step('bert_encoder'); optimizer.step('bert')
                    optimizer.step('predictor');    optimizer.step('predictor_encoder')
                    if epoch >= diff_epoch and 'diffusion' in optimizer.optimizers:
                        optimizer.step('diffusion')
                    if epoch >= joint_epoch:
                        optimizer.step('style_encoder'); optimizer.step('decoder')
                        # mark D and print a compact step map
                        last_d_marker = "D+" if (start_ds and do_d_update) else "D-"

                        wd_step = (start_ds and do_d_update)
                        cad = getattr(slmadv, "skip_update", 1)
                        try:
                            wd_step = wd_step and (((i // max(1, acc_steps)) % max(1, cad)) == 0)
                        except Exception:
                            pass

                        _stepped = {
                            "msd": (start_ds and do_d_update),
                            "mpd": (start_ds and do_d_update),
                            "wd":  wd_step,
                            "bert": True, "bert_enc": True,
                            "pred": True, "pred_enc": True,
                            "diff": (epoch >= diff_epoch and 'diffusion' in optimizer.optimizers),
                            "style": (epoch >= joint_epoch),
                            "dec": (epoch >= joint_epoch)
                        }
                        logger.info("[step] " + " ".join(f"{k}:{'+' if v else '-'}" for k,v in _stepped.items()))

                    # EMA update after real steps
                    if ema and epoch >= ema_cfg.start_epoch and ((i // max(1, acc_steps)) % max(1, ema_cfg.update_freq) == 0):
                        ema.update(model)
                    # ---- GA headroom tuning ----
                    if torch.cuda.is_available():
                        free_b, total_b = torch.cuda.mem_get_info()
                        free_pct = (free_b / max(1, total_b)) * 100.0
                        logger.info(f"[mem] boundary: free={free_b/1e9:.2f}GB ({free_pct:.1f}%), ga={acc_steps}")
                        # cooldown ticks down each boundary
                        if ga_cooldown > 0: ga_cooldown -= 1
                        # DROP when headroom low
                        if free_pct < FREE_LOW_PCT and acc_steps > GA_MIN:
                            acc_steps -= 1
                            eff_scale = (batch_size * acc_steps) / max(1.0, float(E_BASE))
                            _apply_lr_scale(optimizer, eff_scale)
                            logger.info(f"[ga] drop -> {acc_steps} (free={free_pct:.1f}%)")
                            ga_safe_count = 0; ga_cooldown = GA_GROW_COOLDOWN
                        # RISE when headroom consistently high
                        elif (free_pct >= FREE_HIGH_PCT and acc_steps < GA_MAX and ga_cooldown == 0):
                            ga_safe_count += 1
                            if ga_safe_count >= GA_GROW_PATIENCE:
                                acc_steps += 1
                                eff_scale = (batch_size * acc_steps) / max(1.0, float(E_BASE))
                                _apply_lr_scale(optimizer, eff_scale)
                                logger.info(f"[ga] rise -> {acc_steps} (free={free_pct:.1f}%, safe={ga_safe_count})")
                                ga_safe_count = 0; ga_cooldown = GA_GROW_COOLDOWN
                        else:
                            if free_pct < FREE_HIGH_PCT: ga_safe_count = 0
                        # gentle allocator trim when plenty of headroom
                        if free_pct >= (FREE_HIGH_PCT + 7.0):
                            torch.cuda.empty_cache()
                    ga_bucket = 0
            
                    # randomly pick whether to use in-distribution text
                    if np.random.rand() < 0.5:
                        use_ind = True
                    else:
                        use_ind = False

                    if use_ind:
                        ref_lengths = input_lengths
                        ref_texts = texts

                    # Lightweight scalars for logging; lets us free tensors early
                    log_discLM = 0.0
                    log_genLM  = 0.0
                        
                    slm_out = slmadv(
                        i,
                        y_rec_gt,
                        y_rec_gt_pred,
                        waves,
                        mel_input_length,
                        ref_texts,
                        ref_lengths,
                        use_ind,
                        s_trg.detach(),
                        ref_feat if multispeaker else None
                    )

                    if slm_out is None:
                        # stay in the step so logging/heartbeat still prints
                        if not start_ds:
                            slm_gated += 1
                        else:
                            slm_skips += 1
                            cadence = getattr(slmadv, "skip_update", None)
                            if cadence and ((i // max(1, acc_steps)) % max(1, cadence)) != 0:
                                slm_cadence_skips += 1
                            else:
                                slm_fail_skips += 1

                        d_loss_slm, loss_gen_lm, y_pred = 0, 0, None
                        log_discLM, log_genLM = 0.0, 0.0
                    else:
                        d_loss_slm, loss_gen_lm, y_pred = slm_out
                        # SLM generator loss (accumulate; step already handled at boundary)
                        (loss_gen_lm / max(1, acc_steps)).backward()
                        # Keep scalar copies for logs; free big tensors
                        try:
                            log_discLM = float(_scalar(d_loss_slm))
                            log_genLM  = float(_scalar(loss_gen_lm))
                        except Exception:
                            pass
                        y_pred = None  # drop heavy tensor reference

                    # compute the gradient norm
                    total_norm = {}
                    for key in model.keys():
                        total_norm[key] = 0
                        parameters = [p for p in model[key].parameters() if p.grad is not None and p.requires_grad]
                        for p in parameters:
                            param_norm = p.grad.detach().data.norm(2)
                            total_norm[key] += param_norm.item() ** 2
                        total_norm[key] = total_norm[key] ** 0.5

                    # gradient scaling
                    if total_norm['predictor'] > slmadv_params.thresh:
                        for key in model.keys():
                            for p in model[key].parameters():
                                if p.grad is not None:
                                    p.grad *= (1 / total_norm['predictor']) 

                    for p in model.predictor.duration_proj.parameters():
                        if p.grad is not None:
                            p.grad *= slmadv_params.scale

                    for p in model.predictor.lstm.parameters():
                        if p.grad is not None:
                            p.grad *= slmadv_params.scale

                    for p in model.diffusion.parameters():
                        if p.grad is not None:
                            p.grad *= slmadv_params.scale

                    # SLM discriminator loss
                    if d_loss_slm != 0:
                        # update WD without clobbering G grads
                        for p in model['wd'].parameters():
                            if p.grad is not None: p.grad = None
                        d_loss_slm.backward(retain_graph=True)
                        optimizer.step('wd')

                        # free SLM tensors now that grads are consumed
                        try:
                            del d_loss_slm
                            del loss_gen_lm
                        except Exception:
                            pass

                else:
                    d_loss_slm, loss_gen_lm = 0, 0
                    
                iters = iters + 1
                
                # ─ inside the main train loop ─
                if i == 0:
                    # compact early after first successful fwd/bwd
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()

                if (i + 1) % log_interval == 0:
                    total = len(train_dataloader)
                    if not start_ds:
                        logger.info(f"[slm-adv] gated: {slm_gated}/{total} steps so far (diff_epoch={diff_epoch}, current={epoch})")
                    else:
                        logger.info(f"[slm-adv] skipped {slm_skips}/{total} "
                                    f"(cadence={slm_cadence_skips}, fail={slm_fail_skips}) this epoch")

                    batch_time = time.time() - start_time
                    lr_now = optimizer.optimizers['decoder'].param_groups[0]['lr']
                    _log_free("train-boundary")

                    # --- compact, human-friendly line (train_first.py style) ---
                    logger.info(
                        "Epoch [%d/%d], Step [%d/%d], %s | "
                        "Mel Loss: %.5f, Gen Loss: %.5f, Disc Loss: %.5f, "
                        "Dur Loss: %.5f, CE Loss: %.5f, SLM Loss: %.5f | "
                        "GA:%d LR: %.2e"
                        % (
                            epoch + 1, epochs,
                            i + 1, len(train_dataloader),
                            last_d_marker,
                            float(running_loss / log_interval),
                            _scalar(loss_gen_all), _scalar(d_loss),
                            _scalar(loss_dur), _scalar(loss_ce), _scalar(loss_lm),
                            acc_steps, lr_now
                        )
                    )
                    logger.info(f"Time elapsed: {batch_time:.2f} seconds")

                    # --- keep the richer metrics line too (optional) ---
                    log_metrics(
                        phase='train',
                        step=i + 1,
                        total=len(train_dataloader),
                        metrics={
                            'mel':   float(running_loss / log_interval),
                            'disc':  _scalar(d_loss),
                            'dur':   _scalar(loss_dur),
                            'ce':    _scalar(loss_ce),
                            'norm':  _scalar(loss_norm_rec),
                            'F0':    _scalar(loss_F0_rec),
                            'lm':    _scalar(loss_lm),
                            'gen':   _scalar(loss_gen_all),
                            'sty':   _scalar(loss_sty),
                            'diff':  _scalar(loss_diff),
                            'discLM': float(log_discLM),
                            'genLM':  float(log_genLM)
                        },
                        logger=logger,
                        global_step=iters,
                        batch_time=batch_time,
                        mem_gb=current_mem_gb(),
                        lr=lr_now
                    )
                    running_loss = 0.0

                # --- Profile the heavy parts at the same cadence (no spam) ---
                if log_interval > 0 and ((i + 1) % log_interval == 0):
                    try:
                        free_b, total_b = torch.cuda.mem_get_info() if torch.cuda.is_available() else (0, 1)
                        free_gb = free_b / 1e9; free_pct = (free_b / total_b) * 100.0
                        logger.info(
                            "[profile] E%d S%d | align=%.3fs text=%.3fs pred=%.3fs style=%.3fs "
                            "dec_gt=%.3fs dec=%.3fs D=%.3fs stft=%.3fs bwd=%.3fs | free=%.2fGB (%.1f%%)"
                            % (epoch, i + 1, dt_align, dt_text, dt_pred, dt_style,
                            dt_dec_gt, dt_dec, dt_disc, dt_stft, dt_bwd, free_gb, free_pct)
                        )
                    except Exception:
                        pass
            
            
            except torch.cuda.OutOfMemoryError:
                traceback.print_exc()
                _trim_cache("OOM"); 
                gc.collect()
                if acc_steps > GA_MIN:
                    acc_steps = max(GA_MIN, acc_steps // 2); ga_bucket = 0
                    eff_scale = (batch_size * acc_steps) / max(1.0, float(E_BASE))
                    _apply_lr_scale(optimizer, eff_scale)
                    logger.info(f"[ga] OOM: halving acc_steps -> {acc_steps}")
                else:
                    logger.error("[oom] unrecoverable within GA policy; skipping batch")
                continue
                
        loss_test = 0
        loss_align = 0
        loss_f = 0

        _ = [model[key].eval() for key in model]
        ema_backups = None
        if ema and ema_cfg.eval:
            ema_backups = ema.swap_in(model)

        with torch.inference_mode():
            iters_test = 0
            val_start = time.time()
            for batch_idx, batch in enumerate(val_dataloader):
                
                try:
                    waves = batch[0]
                    batch = [b.to(device, non_blocking=True) for b in batch[1:]]
                    texts, input_lengths, ref_texts, ref_lengths, mels, mel_input_length, ref_mels = batch
                    with torch.no_grad():
                        mask = length_to_mask(mel_input_length // (2 ** n_down)).to(device)
                        
                        text_mask = length_to_mask(input_lengths).to(texts.device)
                        _, _, s2s_attn = model.text_aligner(mels, mask, texts)

                        s2s_attn = s2s_attn.transpose(-1, -2)
                        s2s_attn = s2s_attn[..., 1:]
                        s2s_attn = s2s_attn.transpose(-1, -2)

                        # sanitize + row-renorm (Stage-1 parity)
                        attn_mask = (~mask).unsqueeze(-1).expand(mask.shape[0], mask.shape[1], text_mask.shape[-1]).float().transpose(-1, -2)
                        attn_mask = attn_mask.float() * (~text_mask).unsqueeze(-1).expand(text_mask.shape[0], text_mask.shape[1], mask.shape[-1]).float()
                        attn_mask = (attn_mask < 1)
                        s2s_attn = s2s_attn.masked_fill(attn_mask, 0.0)
                        s2s_attn = torch.nan_to_num(s2s_attn, nan=0.0, posinf=0.0, neginf=0.0)
                        row_sum = s2s_attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)
                        s2s_attn = s2s_attn / row_sum
                        mask_ST = mask_from_lens(s2s_attn, input_lengths, mel_input_length // (2 ** n_down))

                        s2s_attn_mono = maximum_path(s2s_attn, mask_ST)

                        # encode
                        t_en = model.text_encoder(texts, input_lengths, text_mask)
                        t_en = torch.nan_to_num(t_en, nan=0.0, posinf=0.0, neginf=0.0)
                        asr      = (t_en @ s2s_attn)
                        asr_mono = (t_en @ s2s_attn_mono)

                        d_gt = s2s_attn_mono.sum(axis=-1).detach()

                    ss = []
                    gs = []

                    for bib in range(len(mel_input_length)):
                        mel_length = int(mel_input_length[bib].item())
                        mel = mels[bib, :, :mel_input_length[bib]]
                        s = model.predictor_encoder(mel.unsqueeze(0).unsqueeze(1))
                        ss.append(s)
                        s = model.style_encoder(mel.unsqueeze(0).unsqueeze(1))
                        gs.append(s)

                    s = torch.stack(ss).squeeze(1)
                    gs = torch.stack(gs).squeeze(1)
                    s_trg = torch.cat([s, gs], dim=-1).detach()

                    bert_dur = model.bert(texts, attention_mask=(~text_mask).int())
                    d_en = model.bert_encoder(bert_dur).transpose(-1, -2)

                    d, p = model.predictor(d_en, s,
                                           input_lengths,
                                           s2s_attn_mono,
                                           text_mask)
                    # alignments no longer needed for val → free now
                    try:
                        del s2s_attn_mono, s2s_attn, attn_mask
                    except Exception:
                        pass


                    # get clips
                    mel_len = int(mel_input_length.min().item() / 2 - 1)
                    en = []
                    en_mono_list = []
                    gt = []
                    p_en = []
                    wav = []
                    wav_cpu = []

                    for bib in range(len(mel_input_length)):
                        mel_length = int(mel_input_length[bib].item() / 2)

                        random_start = np.random.randint(0, mel_length - mel_len)
                        
                        en.append(asr[bib, :, random_start:random_start+mel_len])
                        en_mono_list.append(asr_mono[bib, :, random_start:random_start+mel_len])

                        p_en.append(p[bib, :, random_start:random_start+mel_len])

                        gt.append(mels[bib, :, (random_start * 2):((random_start+mel_len) * 2)])

                        y = waves[bib][(random_start * 2) * 300:((random_start+mel_len) * 2) * 300]
                        wav_cpu.append(torch.from_numpy(y))

                    # Keep CPU-pinned; move a short-lived copy only for STFT
                    wav = torch.stack(wav_cpu).float().pin_memory()

                    en = torch.stack(en)
                    en_mono = torch.stack(en_mono_list)
                    p_en = torch.stack(p_en)
                    gt = torch.stack(gt).detach()

                    s = model.predictor_encoder(gt.unsqueeze(1))
                    s = torch.nan_to_num(s, 0.0, 0.0, 0.0)

                    F0_fake, N_fake = model.predictor.F0Ntrain(p_en, s)

                    loss_dur = 0
                    for _s2s_pred, _text_input, _text_length in zip(d, (d_gt), input_lengths):
                        _s2s_pred = _s2s_pred[:_text_length, :]
                        _text_input = _text_input[:_text_length].long()
                        _s2s_trg = torch.zeros_like(_s2s_pred)
                        for bib in range(_s2s_trg.shape[0]):
                            _s2s_trg[bib, :_text_input[bib]] = 1
                        _dur_pred = torch.sigmoid(_s2s_pred).sum(axis=1)
                        loss_dur += F.l1_loss(_dur_pred[1:_text_length-1], 
                                               _text_input[1:_text_length-1])

                    loss_dur /= texts.size(0)

                    s = model.style_encoder(gt.unsqueeze(1))
                    s = torch.nan_to_num(s, 0.0, 0.0, 0.0)

                    en       = torch.nan_to_num(en,       0.0, 0.0, 0.0)
                    en_mono  = torch.nan_to_num(en_mono,  0.0, 0.0, 0.0)
                    F0_fake  = torch.nan_to_num(F0_fake,  0.0, 0.0, 0.0)
                    N_fake   = torch.nan_to_num(N_fake,   0.0, 0.0, 0.0)

                    y_rec = model.decoder(en, F0_fake, N_fake, s)
                    if not torch.isfinite(y_rec).all():
                        # one FP32 retry on decoder, then monotonic fallback once
                        dec_mod = getattr(model["decoder"], "module", model["decoder"])
                        try:
                            orig_dtype = next(dec_mod.parameters()).dtype
                        except StopIteration:
                            orig_dtype = torch.float32
                        try:
                            dec_mod.to(torch.float32)
                            y_rec = dec_mod(en.float(), F0_fake.float(), N_fake.float(), s.float())
                        finally:
                            dec_mod.to(orig_dtype)
                        if not torch.isfinite(y_rec).all():
                            # use monotonic attention features once
                            y_rec = dec_mod(en_mono.float(), F0_fake.float(), N_fake.float(), s.float())

                            if not torch.isfinite(y_rec).all():
                                # skip batch — don’t poison val average
                                continue

                    _wav_dev = wav.to(device, non_blocking=True)
                    loss_mel = stft_loss(y_rec.contiguous().squeeze(1), _wav_dev.detach())
                    del _wav_dev

                    F0_real, _, F0 = model.pitch_extractor(gt.unsqueeze(1)) 

                    loss_F0 = F.l1_loss(F0_real, F0_fake) / 10

                    loss_test += (loss_mel).mean()
                    loss_align += (loss_dur).mean()
                    loss_f += (loss_F0).mean()

                    iters_test += 1

                                    
                    if batch_idx % 1 == 0:
                        log_metrics(
                            phase='val',
                            step=batch_idx,
                            total=len(val_dataloader),
                            metrics={
                                'mel': (loss_test / max(1, batch_idx + 1)).item(),
                                'dur': (loss_align / max(1, batch_idx + 1)).item(),
                                'F0':  (loss_f     / max(1, batch_idx + 1)).item()
                            },
                            logger=logger,
                            batch_time=time.time() - val_start,
                            mem_gb=current_mem_gb()
                        )
                        val_start = time.time()

                except Exception as e:
                    print(f"run into exception", e)
                    traceback.print_exc()
                    continue

            # light allocator trim after val
            _trim_cache("post-val"); gc.collect()

        log_print(f"Epochs: {epoch + 1}", logger)
        log_print('Validation loss: %.3f, Dur loss: %.3f, F0 loss: %.3f' % (loss_test / iters_test, loss_align / iters_test, loss_f / iters_test) + '\n\n\n', logger)
        log_print('\n\n\n', logger)

        if ema_backups is not None:
            ema.restore(model, ema_backups)
        
        if epoch < joint_epoch:
            # generating reconstruction examples with GT duration
            
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


                    s_dur = model.predictor_encoder(gt.unsqueeze(1))
                    p_en = p[bib, :, :mel_length // 2].unsqueeze(0)

                    F0_fake, N_fake = model.predictor.F0Ntrain(p_en, s_dur)

                    y_pred = model.decoder(en, F0_fake, N_fake, s)

                    # free big temps for previews
                    try:
                        del y_rec
                        del y_pred
                    except Exception:
                        pass

                    if bib >= 5:
                        break
        else:
            # generating sampled speech from text directly
            with torch.no_grad():
                # Build a lightweight, local sampler for previews; free after use
                val_sampler = DiffusionSampler(
                    _get_diffusion_core(model),
                    sampler=ADPM2Sampler(),
                    sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),
                    clamp=False
                )

                # compute reference styles
                ref_feat = None
                if multispeaker and epoch >= diff_epoch:
                    ref_ss = model.style_encoder(ref_mels.unsqueeze(1))
                    ref_sp = model.predictor_encoder(ref_mels.unsqueeze(1))
                    ref_feat = torch.cat([ref_ss, ref_sp], dim=1)
                    
                for bib in range(len(d_en)):
                    if multispeaker and ref_feat is not None:
                        s_pred = val_sampler(
                            noise=torch.randn((1, 256)).unsqueeze(1).to(texts.device),
                            embedding=bert_dur[bib].unsqueeze(0),
                            embedding_scale=1,
                            features=ref_feat[bib].unsqueeze(0),  # reference from same speaker
                            num_steps=5
                        ).squeeze(1)
                    else:
                        s_pred = val_sampler(
                            noise=torch.randn((1, 256)).unsqueeze(1).to(texts.device),
                            embedding=bert_dur[bib].unsqueeze(0),
                            embedding_scale=1,
                            num_steps=5
                        ).squeeze(1)

                    s = s_pred[:, 128:]
                    s_ref = s_pred[:, :128] 

                    d = model.predictor.text_encoder(d_en[bib, :, :input_lengths[bib]].unsqueeze(0), 
                                                     s, input_lengths[bib, ...].unsqueeze(0), text_mask[bib, :input_lengths[bib]].unsqueeze(0))

                    x, _ = model.predictor.lstm(d)
                    duration = model.predictor.duration_proj(x)

                    duration = torch.sigmoid(duration).sum(axis=-1)
                    pred_dur = torch.round(duration.squeeze()).clamp(min=1)

                    pred_dur[-1] += 5

                    pred_aln_trg = torch.zeros(input_lengths[bib], int(pred_dur.sum().data))
                    c_frame = 0
                    for i in range(pred_aln_trg.size(0)):
                        pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
                        c_frame += int(pred_dur[i].data)

                    # encode prosody
                    en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(texts.device))
                    F0_pred, N_pred = model.predictor.F0Ntrain(en, s)
                    out = model.decoder(
                        (t_en[bib, :, :input_lengths[bib]].unsqueeze(0) @ pred_aln_trg.unsqueeze(0).to(texts.device)),
                        F0_pred, N_pred, s_ref.squeeze().unsqueeze(0)
                    )

                    # free per-preview temps
                    try:
                        del s_pred, s, ref, d, x, duration, pred_dur, pred_aln_trg, en, F0_pred, N_pred, out
                    except Exception:
                        pass

                    if bib >= 5:
                        break

                # free the local sampler & trim allocator
                try:
                    del val_sampler
                    gc.collect()
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
                except Exception:
                    pass

        if epoch % saving_epoch == 0 or ((epoch + 1) == epochs):
            cur = (loss_test / iters_test)
            if cur < best_loss:
                best_loss = cur
                logger.info(f"[ckpt] new best_loss={best_loss:.4f} at epoch {epoch}")
            log_print('Saving..', logger)

            state = {
                'net':  {key: model[key].state_dict() for key in model}, 
                'optimizer': optimizer.state_dict(),
                'iters': iters,
                'val_loss': loss_test / iters_test,
                'epoch': epoch,
            }
            state['signature'] = {**stage2_sig, "epoch": epoch + 1, "iters": iters}
            save_path = osp.join(log_dir, 'epoch_2nd_%05d.pth' % epoch)
            torch.save(state, save_path)
            logger.info(f"[ckpt] saved: {save_path}")
            
            # if estimate sigma, save the estimated simga
            if model_params.diffusion.dist.estimate_sigma_data:
                config['model_params']['diffusion']['dist']['sigma_data'] = float(np.mean(running_std))
                
                with open(osp.join(log_dir, osp.basename(config_path)), 'w') as outfile:
                    yaml.dump(config, outfile, default_flow_style=True)

            if ((epoch + 1) == epochs):
                log_print('Saving Second Stage..', logger)
                final_save_path = osp.join(log_dir, config.get('second_stage_path', 'second_stage.pth'))
                torch.save(state, final_save_path)
                shutil.copy(osp.join(log_dir, osp.basename(config_path)),osp.dirname(log_dir))

                logger.info(f"[ckpt] Second Stage saved: {final_save_path}")

            gc.collect(); 
            if torch.cuda.is_available(): 
                torch.cuda.empty_cache()
            logger.info("[train] COMPLETE")

def trace_shapes(model, logger=None):
    def _hook(mod, inp, out):
        def safe_shape(x):
            if isinstance(x, torch.Tensor):
                return tuple(x.shape)
            elif isinstance(x, PackedSequence):
                return f"PackedSequence({tuple(x.data.shape)})"
            else:
                return type(x).__name__

        cin = [safe_shape(x) for x in inp]
        cout = safe_shape(out) if isinstance(out, (torch.Tensor, PackedSequence)) else type(out).__name__

        line = f"{mod.__class__.__name__:20s}  {cin} -> {cout}"
        if logger:
            log_print(line, logger)
        
        print(line)

    for m in model.modules():
        if not isinstance(m, nn.Sequential) and m.__class__.__name__ != 'ModuleList':
            m.register_forward_hook(_hook)

        

# ------------------------------------------------------------------
# 1.  Redirect everything to log_dir/train_stdout.log
# ------------------------------------------------------------------
def _redirect_io(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, 'train_stdout_second.log')
    log_file = open(log_path, 'a', buffering=262144)  # ~256 KiB buffered
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

def _summarize_speakers(lines):
    from collections import Counter
    counts = Counter(); bad = 0
    for ln in lines:
        parts = str(ln).strip().split('|')
        if len(parts) >= 3:
            try:
                sid = int(parts[2])
            except Exception:
                sid = 0; bad += 1
            counts[sid] += 1
    return counts, bad


if __name__=="__main__":
    main()
