"""Train a Hybrid-Demucs-style separator to pull guitar out of a mix.

Model   : torchaudio.models.HDemucs  (reference U-Net + band-split transformer)
          forward: (B, C, T) -> (B, num_sources, C, T)
Data    : folders per track, each holding  guitar.wav  and  other.wav  (train),
          plus  mixture.wav  on the valid split (kept for SI-SDR scoring).
Loss    : negative SI-SDR per source; silent references use waveform MSE instead.
Optim   : AdamW + cosine LR, grad-clip, bfloat16 AMP when on CUDA.

Usage:
    # CPU smoke test (tiny model, few steps) -- no GPU needed:
    python train.py --data data/fake --epochs 1 --steps-per-epoch 8 \
        --batch-size 4 --segment-sec 4 --model tiny --device cpu --out runs/smoke

    # Real GPU run (bigger model):
    python train.py --data data/separation --epochs 8 --steps-per-epoch 6000 \
        --batch-size 2 --segment-sec 10 --model small --device cuda --out runs/guitar_v1
"""
from __future__ import annotations
import argparse, json, math, os, time, random
from contextlib import nullcontext
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import soundfile as sf
from torchaudio.models import HDemucs as RefHDemucs

SOURCES = ["guitar", "other"]
# NOTE: torchaudio HDemucs needs even `depth` and nfft >= 1024 to keep its internal
# time-branch encoder/decoder lengths consistent (odd depth / small nfft -> skip-concat crash).
MODEL_SPECS = {
    "tiny":   dict(channels=24, depth=4, nfft=1024),
    "small":  dict(channels=48, depth=6, nfft=2048),
    "medium": dict(channels=64, depth=8, nfft=2048),
}


# --------------------------------------------------------------------------- #
#  SI-SDR (the standard separation metric)
# --------------------------------------------------------------------------- #
def si_sdr(x: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Estimate SI-SDR (dB) of x vs ref over the last dim. We train by minimising -si_sdr.

    SI-SDR = 10*log10(||target||^2 / ||residual||^2), where target is the projection of x
    onto ref. Energies are computed in at least float32 so bf16 autocast can't wreck them.
    Silent (or constant) references have no defined SI-SDR and are rejected.
    """
    if x.shape != ref.shape:
        raise ValueError(f"SI-SDR requires matching shapes, got {x.shape} and {ref.shape}")
    if x.ndim == 0 or x.shape[-1] == 0:
        raise ValueError("SI-SDR requires non-empty audio")
    dtype = torch.promote_types(torch.promote_types(x.dtype, ref.dtype), torch.float32)
    x, ref = x.to(dtype), ref.to(dtype)
    x = x - x.mean(dim=-1, keepdim=True)
    ref = ref - ref.mean(dim=-1, keepdim=True)
    ref_energy = (ref * ref).sum(dim=-1)
    if (ref_energy <= eps).any():
        raise ValueError("SI-SDR is undefined for silent or constant references")
    dot = (x * ref).sum(dim=-1)
    alpha = dot / ref_energy
    target = alpha.unsqueeze(-1) * ref
    residual = x - target
    num = (target * target).sum(dim=-1)
    den = (residual * residual).sum(dim=-1)
    return 10.0 * torch.log10((num + eps) / (den + eps))


def source_loss(x: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Mean negative SI-SDR, using waveform MSE for silent/constant references.

    MSE teaches the model to output silence when that instrument is absent, while
    avoiding an undefined SI-SDR score. Each batch/channel item contributes once.
    """
    if x.shape != ref.shape:
        raise ValueError(f"Source loss requires matching shapes, got {x.shape} and {ref.shape}")
    if x.numel() == 0 or x.ndim == 0:
        raise ValueError("Source loss requires non-empty audio")
    dtype = torch.promote_types(torch.promote_types(x.dtype, ref.dtype), torch.float32)
    x, ref = x.to(dtype), ref.to(dtype)
    centered_ref = ref - ref.mean(dim=-1, keepdim=True)
    active = centered_ref.square().sum(dim=-1) > eps
    losses = (x - ref).square().mean(dim=-1)
    if active.any():
        losses = losses.clone()
        losses[active] = -si_sdr(x[active], ref[active], eps)
    return losses.mean()


def _resample(x: np.ndarray, sr: int, target: int) -> np.ndarray:
    if sr == target:
        return x
    n = int(round(len(x) * target / sr))
    idx = np.linspace(0, len(x) - 1, n)
    i0 = np.floor(idx).astype(int); i1 = np.minimum(i0 + 1, len(x) - 1)
    f = idx - i0
    return (x[i0] * (1 - f) + x[i1] * f).astype(np.float32)


# --------------------------------------------------------------------------- #
#  Dataset: fixed-length segments sampled from (guitar, other) pairs
# --------------------------------------------------------------------------- #
class GuitarDataset(Dataset):
    def __init__(self, root: str, split: str, sr: int, segment_sec: float):
        d = Path(root) / split
        self.tracks = sorted([t for t in d.iterdir() if t.is_dir()])
        if not self.tracks:
            raise ValueError(f"No tracks found in {d}")
        self.sr = sr
        self.segment = int(segment_sec * sr)
        self._cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    def __len__(self):
        # a few random segments per track per epoch
        return len(self.tracks) * 4

    def __getitem__(self, idx):
        name = self.tracks[idx % len(self.tracks)]
        key = name.name
        if key not in self._cache:
            g, sr = sf.read(name / "guitar.wav", dtype="float32")
            o, _ = sf.read(name / "other.wav", dtype="float32")
            if sr != self.sr:
                g = _resample(g, sr, self.sr)
                o = _resample(o, sr, self.sr)
            self._cache[key] = (torch.from_numpy(g).float(), torch.from_numpy(o).float())
        g, o = self._cache[key]
        T = min(g.shape[0], o.shape[0])
        g, o = g[:T], o[:T]
        if T > self.segment:
            st = random.randrange(0, T - self.segment)
            g, o = g[st:st + self.segment], o[st:st + self.segment]
        else:
            pad = self.segment - T
            g = F.pad(g, (0, pad)); o = F.pad(o, (0, pad))
        mix = g + o
        return mix, g, o


# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(model, valid: GuitarDataset, device, seg_sec: float) -> tuple[float, float]:
    """Deterministic full-track evaluation.

    For every valid track, run the model over the whole mixture.wav in back-to-back
    `seg_sec` chunks (last chunk zero-padded), stitch the estimates, and score SI-SDR
    against the full guitar.wav / other.wav. Returns mean SI-SDR in dB as (guitar, other).
    Silent references are excluded per source, with counts reported. At least one
    non-silent reference per source is required to report both validation scores.
    """
    if not valid.tracks:
        raise ValueError("Validation set contains no tracks")
    model.eval()
    seg = int(seg_sec * valid.sr)
    if seg < 1:
        raise ValueError("Evaluation segment must contain at least one sample")
    totals = [0.0, 0.0]
    counts = [0, 0]
    for trk in valid.tracks:
        g, sr = sf.read(trk / "guitar.wav", dtype="float32")
        o, _ = sf.read(trk / "other.wav", dtype="float32")
        mix_path = trk / "mixture.wav"
        mix = sf.read(mix_path, dtype="float32")[0] if mix_path.exists() else g + o
        if sr != valid.sr:
            g, o, mix = (_resample(a, sr, valid.sr) for a in (g, o, mix))
        T = min(len(g), len(o), len(mix))
        if T == 0:
            raise ValueError(f"Validation track contains empty audio: {trk}")
        mix_t = torch.from_numpy(np.ascontiguousarray(mix[:T])).to(device)
        est = torch.zeros(len(SOURCES), T, device=device)
        for st in range(0, T, seg):
            chunk = mix_t[st:st + seg]
            n = chunk.shape[0]
            if n < seg:
                chunk = F.pad(chunk, (0, seg - n))
            out = model(chunk[None, None])             # (1, S, 1, seg)
            est[:, st:st + n] = out[0, :, 0, :n].float()
        g_t = torch.from_numpy(np.ascontiguousarray(g[:T])).to(device)
        o_t = torch.from_numpy(np.ascontiguousarray(o[:T])).to(device)
        # float64 for the energy sums: tracks are millions of samples long
        for i, ref in enumerate((g_t.double(), o_t.double())):
            if (ref - ref.mean()).square().sum().item() <= 1e-8:
                continue
            totals[i] += si_sdr(est[i].double(), ref).item()
            counts[i] += 1
    for name, n in zip(SOURCES, counts):
        if n == 0:
            raise ValueError(f"No non-silent {name} references in validation set")
        if n < len(valid.tracks):
            print(f"[eval] {name}: excluded {len(valid.tracks) - n} silent reference tracks")
    return totals[0] / counts[0], totals[1] / counts[1]


def make_model(spec: str, device: torch.device):
    spec = MODEL_SPECS[spec]
    m = RefHDemucs(sources=SOURCES, audio_channels=1, **spec).to(device)
    print(f"[model] HDemucs({spec}) params={sum(p.numel() for p in m.parameters())/1e6:.1f}M on {device}")
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="root with train/ and valid/")
    ap.add_argument("--out", default="runs/latest")
    ap.add_argument("--model", default="small", choices=list(MODEL_SPECS))
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--steps-per-epoch", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--segment-sec", type=float, default=10.0)
    ap.add_argument("--sr", type=int, default=16000, help="everything resampled to this")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--eval-every", type=int, default=0, help="0 = only at end")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    device = torch.device(a.device)
    torch.manual_seed(a.seed); random.seed(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    print(f"[env] torch={torch.__version__} cuda={torch.cuda.is_available()} device={device}")
    if device.type == "cuda":
        p = torch.cuda.get_device_properties(0)
        print(f"[env] gpu={p.name} vram={p.total_memory/1e9:.2f}GB")

    train = GuitarDataset(a.data, "train", a.sr, a.segment_sec)
    valid = GuitarDataset(a.data, "valid", a.sr, a.segment_sec)
    print(f"[data] train tracks={len(train.tracks)}  valid tracks={len(valid.tracks)}  "
          f"segment={int(a.segment_sec*a.sr)} samples")
    print(f"[data] samples per epoch ~ {len(train)} (batch={a.batch_size}, steps/epoch={a.steps_per_epoch})")

    model = make_model(a.model, device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    max_steps = a.epochs * a.steps_per_epoch
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, max_steps), eta_min=a.lr * 0.01)
    use_amp = (device.type == "cuda") and torch.cuda.is_bf16_supported()

    loader = DataLoader(train, batch_size=a.batch_size, shuffle=True,
                        num_workers=0, drop_last=True)

    step = 0; t0 = time.time(); best = -float("inf")
    print(f"[loop] {max_steps} steps total, amp={use_amp}")

    for ep in range(a.epochs):
        model.train()
        it = iter(loader); it_exhausted = False
        for k in range(a.steps_per_epoch):
            if not it_exhausted:
                try:
                    mix, g, o = next(it)
                except StopIteration:
                    it = iter(loader); it_exhausted = True
                    mix, g, o = next(it)
            else:
                mix, g, o = next(iter(loader))

            mix = mix.to(device).unsqueeze(1)   # (B,1,T)
            g = g.to(device).unsqueeze(1); o = o.to(device).unsqueeze(1)

            cm = torch.autocast(device_type=device.type, dtype=torch.bfloat16) if use_amp else nullcontext()
            with cm:
                est = model(mix)                # (B,S,1,T)
                est_g = est[:, 0]               # (B,1,T), matching g
                est_o = est[:, 1]               # (B,1,T), matching o
                loss = source_loss(est_g, g) + source_loss(est_o, o)

            loss.backward()
            if a.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip)
            opt.step(); opt.zero_grad(set_to_none=True); sched.step()
            step += 1

            if step % a.log_every == 0:
                qps = step / max(1e-6, time.time() - t0)
                print(f"  ep{ep} step {step}/{max_steps} loss={loss.item():.4f} "
                      f"lr={opt.param_groups[0]['lr']:.2e} qps={qps:.2f} ({time.time()-t0:.0f}s)")

            if a.eval_every and step % a.eval_every == 0:
                vg, vo = evaluate(model, valid, device, a.segment_sec)
                best = max(best, vg)
                print(f"    [eval@{step}] valid SI-SDR guitar={vg:.3f} dB  other={vo:.3f} dB  "
                      f"(best guitar={best:.3f})")
                model.train()

    vg, vo = evaluate(model, valid, device, a.segment_sec)
    print(f"[end] valid SI-SDR guitar={vg:.3f} dB  other={vo:.3f} dB  "
          f"steps={step}  time={time.time()-t0:.0f}s")

    ckpt = out / "model.pt"
    torch.save({"model": model.state_dict(), "spec": a.model, "sources": SOURCES,
                 "steps": step, "valid_sdr_db": vg, "valid_sdr_other_db": vo}, ckpt)
    print(f"[ckpt] saved {ckpt}")
    with open(out / "train.json", "w") as f:
        json.dump({"args": vars(a), "valid_sdr_db": vg, "valid_sdr_other_db": vo, "steps": step,
                    "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"},
                    f, indent=2)


if __name__ == "__main__":
    main()
