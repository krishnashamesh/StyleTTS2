import yaml, torch, onnx, types
from munch import Munch
from models import build_model

YAML  = 'checkpoints/config.yml'
CKPT  = 'checkpoints/latest.pth'
ONNX  = 'styletts2_hybrid10.onnx'
N_STEPS = 10              # 200 for full diffusion

# -------------------------------------------------------------------
# 0 · config tweaks
cfg_raw   = yaml.safe_load(open(YAML))      # full YAML as plain dict
cfg_model = Munch.fromDict(cfg_raw['model_params'])  # deep-convert everything

# the original config has no 'n_timesteps' key; add it:
cfg_model.diffusion.n_timesteps = N_STEPS


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

# -------------------------------------------------------------------
# 1 · build nets on CPU
dummy = torch.nn.Identity()
nets  = build_model(cfg_model, DummyAligner(), DummyPitch(), build_dummy_bert())

print(nets.keys())

state = torch.load(CKPT, map_location='cpu')
for name, module in nets.items():
    if name in state:
        module.load_state_dict(state[name], strict=False)

inner = nets['diffusion'].diffusion          # KDiffusion sampler

print("\nINNER TYPE:", type(inner))
print("   has p_sample_loop:", hasattr(inner, "p_sample_loop"))
print("   has sample       :", hasattr(inner, "sample"))

# If inner is another wrapper, peek one level deeper
for name in ("model", "_model", "sampler", "k_diffusion"):
    obj = getattr(inner, name, None)
    if obj is not None:
        print(f"  › {name}: {type(obj)}  | p_sample_loop:"
              f" {hasattr(obj,'p_sample_loop')}  | sample:"
              f" {hasattr(obj,'sample')}")
        
print("\nCandidate methods on KDiffusion that might be the sampler:")
print([m for m in dir(inner) if any(k in m.lower()
                                    for k in ("sample", "loop", "step", "run", ""))])

if hasattr(inner, "p_sample_loop") and not hasattr(inner, "sample"):
    import types
    inner.sample = types.MethodType(
        lambda self, *a, **kw: self.p_sample_loop(*a, **kw), inner
    )

# -------------------------------------------------------------------
# 2 · thin wrapper to expose a single forward()
class StyleTTS2Wrapper(torch.nn.Module):
    def __init__(self, nets, cfg):
        super().__init__()
        self.nets = torch.nn.ModuleDict(nets)
        self.cfg  = cfg

    def forward(self, tokens):                 # ⇒ (1, samples)
        # ① tokens → mel via diffusion
        mel = self.nets["diffusion"].diffusion.sample(      # ← inner.sample
                 tokens, self.nets, self.cfg, use_teacher_forcing=False)
        wav = self.nets["decoder"](mel)
        return wav

wrapper = StyleTTS2Wrapper(nets, cfg_model)
wrapper.eval()

# -------------------------------------------------------------------
# 3 · dummy input & export
TOK = torch.LongTensor([[1]*32])   # batch=1 · 32 tokens (dynamic)

torch.onnx.export(
    wrapper, (TOK,),
    ONNX,
    input_names  = ['tokens'],
    output_names = ['wav'],
    dynamic_axes = {'tokens': {1: 'T'}, 'wav':  {1: 'samples'}},
    opset_version=17,
)
print('✓ exported', ONNX)
