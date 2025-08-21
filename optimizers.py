#coding:utf-8
import os, sys
import os.path as osp
import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from functools import reduce
from torch.optim import AdamW

import warnings
# bitsandbytes 8-bit optimizers
try:
    import bitsandbytes as bnb
    _HAS_BNB = True
except Exception:
    _HAS_BNB = False



class MultiOptimizer:
    def __init__(self, optimizers={}, schedulers={}):
        self.optimizers = optimizers
        self.schedulers = schedulers
        self.keys = list(optimizers.keys())
        self.param_groups = reduce(lambda x,y: x+y, [v.param_groups for v in self.optimizers.values()])

    def state_dict(self):
        state_dicts = [(key, self.optimizers[key].state_dict())\
                       for key in self.keys]
        return state_dicts

    def load_state_dict(self, state_dict):
        for key, val in state_dict:
            try:
                self.optimizers[key].load_state_dict(val)
            except:
                print("Unloaded %s" % key)

    def step(self, key=None, scaler=None):
        keys = [key] if key is not None else self.keys
        _ = [self._step(key, scaler) for key in keys]

    def _step(self, key, scaler=None):
        if scaler is not None:
            scaler.step(self.optimizers[key])
            scaler.update()
        else:
            self.optimizers[key].step()

    def zero_grad(self, key=None, set_to_none=True):
        if key is not None:
            self.optimizers[key].zero_grad(set_to_none=set_to_none)
        else:
            _ = [self.optimizers[key].zero_grad(set_to_none=set_to_none) for key in self.keys]

    def scheduler(self, *args, key=None):
        if key is not None:
            self.schedulers[key].step(*args)
        else:
            _ = [self.schedulers[key].step(*args) for key in self.keys]

def define_scheduler(optimizer, params):
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=params.get('max_lr', 2e-4),
        epochs=params.get('epochs', 200),
        steps_per_epoch=params.get('steps_per_epoch', 1000),
        pct_start=params.get('pct_start', 0.0),
        div_factor=1,
        final_div_factor=1)

    return scheduler

def build_optimizer(parameters_dict, scheduler_params_dict, lr, impl: str = "torch"):
    
    impl = (os.environ.get("OPTIM_IMPL", impl) or "torch").lower()

    use_fused = hasattr(torch.optim.AdamW, "fused") and torch.cuda.is_available()
    optim = {}
    for key, params in parameters_dict.items():

        # keep text_encoder on standard AdamW for stability
        if key in ("text_encoder", "style_encoder"):
            if use_fused:
                opt = torch.optim.AdamW(params,
                                        lr=lr, weight_decay=1e-4,
                                        betas=(0.0, 0.99), eps=1e-9,
                                        fused=True)
            else:
                opt = torch.optim.AdamW(params,
                                        lr=lr, weight_decay=1e-4,
                                        betas=(0.0, 0.99), eps=1e-9)
                for pg in opt.param_groups:
                    pg.setdefault("foreach", True)
            optim[key] = opt
            continue
        
        if impl == "bnb8bit":
            if not _HAS_BNB:
                warnings.warn("[optim] bitsandbytes not available; falling back to torch AdamW")
                impl = "torch"
            else:
                # A) GPU 8-bit AdamW – biggest VRAM win with minimal speed impact
                min8 = int(os.environ.get("BNB_MIN_8BIT_SIZE", "65536"))
                opt = bnb.optim.AdamW8bit(params, lr=lr, weight_decay=1e-4,
                                          betas=(0.0, 0.99), eps=1e-9,
                                          min_8bit_size=min8)
                optim[key] = opt
                continue

        if impl == "bnb_paged8":
            if not _HAS_BNB:
                warnings.warn("[optim] bitsandbytes not available; falling back to torch AdamW")
                impl = "torch"
            else:
                # B) Paged 8-bit AdamW – optimizer states mostly in host, lower VRAM, some slowdown
                min8 = int(os.environ.get("BNB_MIN_8BIT_SIZE", "65536"))
                opt = bnb.optim.AdamW8bit(params, lr=lr, weight_decay=1e-4,
                                          betas=(0.0, 0.99), eps=1e-9,
                                          min_8bit_size=min8)
                optim[key] = opt
                continue
        # Default: PyTorch AdamW (fused if possible, else foreach path)
        if use_fused:
            opt = torch.optim.AdamW(params,
                                    lr=lr, weight_decay=1e-4,
                                    betas=(0.0, 0.99), eps=1e-9,
                                    fused=True)
        else:
            opt = torch.optim.AdamW(params,
                                    lr=lr, weight_decay=1e-4,
                                    betas=(0.0, 0.99), eps=1e-9)
            for pg in opt.param_groups:
                pg.setdefault("foreach", True)
                
        optim[key] = opt

    schedulers = dict([(key, define_scheduler(opt, scheduler_params_dict[key])) \
                       for key, opt in optim.items()])

    multi_optim = MultiOptimizer(optim, schedulers)
    return multi_optim