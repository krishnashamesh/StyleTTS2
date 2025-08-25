from monotonic_align import maximum_path
from monotonic_align import mask_from_lens
from monotonic_align.core import maximum_path_c
import numpy as np
import torch
import copy
from torch import nn
import torch.nn.functional as F
import torchaudio
import librosa
import matplotlib.pyplot as plt
import os, platform
from munch import Munch
from contextlib import nullcontext

def maximum_path(neg_cent, mask):
  """ Cython optimized version.
  neg_cent: [b, t_t, t_s]
  mask: [b, t_t, t_s]
  """
  device = neg_cent.device
  dtype = neg_cent.dtype
  neg_cent =  np.ascontiguousarray(neg_cent.data.cpu().numpy().astype(np.float32))
  path =  np.ascontiguousarray(np.zeros(neg_cent.shape, dtype=np.int32))

  t_t_max = np.ascontiguousarray(mask.sum(1)[:, 0].data.cpu().numpy().astype(np.int32))
  t_s_max = np.ascontiguousarray(mask.sum(2)[:, 0].data.cpu().numpy().astype(np.int32))
  maximum_path_c(path, neg_cent, t_t_max, t_s_max)
  return torch.from_numpy(path).to(device=device, dtype=dtype)

def get_data_path_list(train_path=None, val_path=None):
    if train_path is None:
        train_path = "Data/train_list.txt"
    if val_path is None:
        val_path = "Data/val_list.txt"

    with open(train_path, 'r', encoding='utf-8', errors='ignore') as f:
        train_list = f.readlines()
    with open(val_path, 'r', encoding='utf-8', errors='ignore') as f:
        val_list = f.readlines()

    return train_list, val_list

def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask+1, lengths.unsqueeze(1))
    return mask

# for norm consistency loss
def log_norm(x, mean=-4, std=4, dim=2):
    """
    normalized log mel -> mel -> norm -> log(norm)
    """
    x = torch.log(torch.exp(x * std + mean).norm(dim=dim))
    return x

def get_image(arrs):
    plt.switch_backend('agg')
    fig = plt.figure()
    ax = plt.gca()
    ax.imshow(arrs)

    return fig

def recursive_munch(d):
    if isinstance(d, dict):
        return Munch((k, recursive_munch(v)) for k, v in d.items())
    elif isinstance(d, list):
        return [recursive_munch(v) for v in d]
    else:
        return d
    
def log_print(message, logger):
    logger.info(message)
    print(message)

# ----------------------------
# Precision helpers
# ----------------------------
def get_precision_cfg(cfg):
    p = (cfg or {}).get("precision", {}) or {}
    mode = str(p.get("mode", "bf16")).lower()
    if mode not in {"fp32", "bf16", "fp16"}:
        mode = "bf16"
    use_autocast = bool(p.get("use_autocast", mode != "fp32"))
    validate_in_fp32 = bool(p.get("validate_in_fp32", True))
    tf32 = bool(p.get("tf32", True))
    return Munch(mode=mode,
                 use_autocast=use_autocast,
                 validate_in_fp32=validate_in_fp32,
                 tf32=tf32)

def amp_context(mode: str = "bf16", enabled: bool = True):
    """Return a context manager for AMP based on config."""
    if (not enabled) or mode == "fp32":
        return nullcontext()
    if mode == "fp16":
        return torch.cuda.amp.autocast(dtype=torch.float16)
    # default: bf16
    return torch.cuda.amp.autocast(dtype=torch.bfloat16)

def _fmt_mem(bytes_val: int) -> str:
    try:
        return f"{bytes_val / (1024**3):.2f} GiB"
    except Exception:
        return str(bytes_val)

def startup_log(accelerator, prec_cfg, cfg, logger):
    """Pretty, single-shot banner printed at start of training."""
    try:
        rank = getattr(accelerator, "process_index", 0)
        world = getattr(accelerator, "num_processes", 1)
        dev = getattr(accelerator, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        mp_effective = getattr(accelerator, "mixed_precision", "unknown")
    except Exception:
        rank, world, dev, mp_effective = 0, 1, torch.device("cpu"), "unknown"

    # GPU info
    gpu_name, total_mem, free_mem = "cpu", None, None
    try:
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(dev)
            gpu_name = props.name
            total_mem = props.total_memory
            try:
                free_mem, tot = torch.cuda.mem_get_info()
                total_mem = tot
            except Exception:
                try:
                    free_mem, tot = torch.cuda.memory.get_info()
                    total_mem = tot
                except Exception:
                    pass
    except Exception:
        pass

    # Flash SDP flags (best effort)
    try:
        sdp = dict(
            flash=getattr(torch.backends.cuda, "flash_sdp_enabled")(),
            mem_efficient=getattr(torch.backends.cuda, "mem_efficient_sdp_enabled")(),
            math=getattr(torch.backends.cuda, "math_sdp_enabled")(),
            flash_available=getattr(torch.backends.cuda, "is_flash_attention_available")(),
        )
    except Exception:
        sdp = None

    try:
        import bitsandbytes as bnb
        bnb_ver = getattr(bnb, "__version__", "unknown")
    except Exception:
        bnb_ver = "not-imported"

    alloc_conf = os.getenv("PYTORCH_CUDA_ALLOC_CONF", "<not set>")

    banner = []
    banner.append("======== RUN START ========")
    banner.append(f"[proc] rank={rank} world={world}")
    banner.append(f"[env]  torch={torch.__version__} cuda={getattr(torch.version, 'cuda', 'n/a')} "
                  f"cudnn={getattr(torch.backends.cudnn, 'version', lambda: 'n/a')()} "
                  f"python={platform.python_version()}")
    if torch.cuda.is_available():
        banner.append(f"[gpu]  device={dev} name='{gpu_name}' "
                      f"free={_fmt_mem(free_mem) if free_mem is not None else 'n/a'} "
                      f"total={_fmt_mem(total_mem) if total_mem is not None else 'n/a'}")
    else:
        banner.append("[gpu]  CUDA not available")
    banner.append(f"[alloc] PYTORCH_CUDA_ALLOC_CONF={alloc_conf}")

    banner.append("[precision] "
                  f"cfg.mode={prec_cfg.mode} cfg.use_autocast={prec_cfg.use_autocast} "
                  f"cfg.validate_in_fp32={prec_cfg.validate_in_fp32} cfg.tf32={prec_cfg.tf32} "
                  f"accelerate.mixed_precision={mp_effective} "
                  f"matmul_precision={torch.get_float32_matmul_precision()} "
                  f"tf32(cublas|cudnn)=({torch.backends.cuda.matmul.allow_tf32}|{torch.backends.cudnn.allow_tf32})")
    if sdp is not None:
        banner.append(f"[sdp]   flash={sdp['flash']} mem_efficient={sdp['mem_efficient']} "
                      f"math={sdp['math']} flash_available={sdp['flash_available']}")
    banner.append(f"[libs]  bitsandbytes={bnb_ver}")

    try:
        bs = cfg.get("batch_size", "n/a")
        ga = cfg.get("grad_accum", "n/a")
        eff = (int(bs) * int(ga) * int(world)) if isinstance(bs, int) and isinstance(ga, int) else "n/a"
        banner.append(f"[train] batch_size={bs} grad_accum={ga} world={world} eff_batch={eff}")
    except Exception:
        pass

    if isinstance(cfg, dict) and "precision" in cfg:
        banner.append(f"[yaml]  precision block: {cfg['precision']}")

    for line in banner:
        log_print(line, logger)
    return True