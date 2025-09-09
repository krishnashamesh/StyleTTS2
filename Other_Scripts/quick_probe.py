# quick_probe.py  –  run:  python quick_probe.py
import yaml, torch
from munch import munchify
from types import SimpleNamespace
from models import build_model

def dummy_bert(h=768):
    m = torch.nn.Identity()
    m.config = SimpleNamespace(hidden_size=h, max_position_embeddings=512)
    return m

def build_dummy_bert(hidden: int = 768):
    class DummyBert(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = type("cfg", (), {"hidden_size": hidden, "max_position_embeddings": 512})()
        def forward(self, x):
            return torch.zeros(len(x), hidden)
    return DummyBert()

class DummyAligner(torch.nn.Module):
    def forward(self, *a, **kw):  # returns fake durations
        return None

class DummyPitch(torch.nn.Module):
    def forward(self, *a, **kw):  # returns fake f0 curve
        return None

cfg = munchify(yaml.safe_load(open('checkpoints/config.yml'))['model_params'])
nets = build_model(cfg,
                   DummyAligner(),          # aligner
                   DummyPitch(),          # pitch
                   build_dummy_bert())                # BERT stub

print("\n🔍  Sub-modules that look usable for spectrogram diffusion:")
for k, m in nets.items():
    cand = [a for a in dir(m) if any(s in a.lower() for s in
                                     ('infer', 'sample', 'generate'))]
    if cand:
        print(f"  {k:15s} → {cand}")
