from __future__ import annotations

import copy, glob, io, math, os, pickle, random, struct, subprocess, sys, time, uuid, zlib
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    import zstandard as zstd
    _HAS_ZSTD = True
except ImportError:
    _HAS_ZSTD = False

class Hyperparameters:
    data_path      = os.environ.get("DATA_PATH",      "./data/datasets/fineweb10B_sp8192")
    train_files    = os.path.join(data_path, "fineweb_train_*.bin")
    val_files      = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_8192_bpe.model")
    run_id         = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed           = int(os.environ.get("SEED", 1337))

    val_batch_size     = int(os.environ.get("VAL_BATCH_SIZE",     524_288))
    val_loss_every     = int(os.environ.get("VAL_LOSS_EVERY",     1000))
    train_log_every    = int(os.environ.get("TRAIN_LOG_EVERY",    200))
    iterations         = int(os.environ.get("ITERATIONS",         20000))
    warmdown_iters     = int(os.environ.get("WARMDOWN_ITERS",     3500))
    warmup_steps       = int(os.environ.get("WARMUP_STEPS",       20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 786_432))
    train_seq_len      = int(os.environ.get("TRAIN_SEQ_LEN",      2048))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))

    # Model shape — SP8192 + hybrid RWKV/attention architecture
    vocab_size         = int(os.environ.get("VOCAB_SIZE",          8192))
    num_unique_layers  = int(os.environ.get("NUM_UNIQUE_LAYERS",   4))     # unique parameter sets
    num_recurrences    = int(os.environ.get("NUM_RECURRENCES",     3))     # how many times to loop
    num_kv_heads       = int(os.environ.get("NUM_KV_HEADS",        4))
    model_dim          = int(os.environ.get("MODEL_DIM",           512))
    num_heads          = int(os.environ.get("NUM_HEADS",           8))
    mlp_mult           = int(os.environ.get("MLP_MULT",            3))
    rope_dims          = int(os.environ.get("ROPE_DIMS",           16))    # partial RoPE dims
    rope_base          = float(os.environ.get("ROPE_BASE",         10000.0))
    logit_softcap      = float(os.environ.get("LOGIT_SOFTCAP",     30.0))
    qk_gain_init       = float(os.environ.get("QK_GAIN_INIT",      5.25))
    bigram_buckets     = int(os.environ.get("BIGRAM_BUCKETS",      2048))
    bigram_dim         = int(os.environ.get("BIGRAM_DIM",          128))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    rwkv_block_indices = os.environ.get("RWKV_BLOCK_INDICES", "1,2")  # which blocks are RWKV-7

    # Optimizer
    embed_lr   = float(os.environ.get("EMBED_LR",   0.035))
    matrix_lr  = float(os.environ.get("MATRIX_LR",  0.025))
    scalar_lr  = float(os.environ.get("SCALAR_LR",  0.025))
    weight_decay = float(os.environ.get("WEIGHT_DECAY", 0.09))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.99))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.92))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 1500))
    beta1      = float(os.environ.get("BETA1",    0.9))
    beta2      = float(os.environ.get("BETA2",    0.95))
    adam_eps   = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.3))

    # EMA / SWA
    ema_decay     = float(os.environ.get("EMA_DECAY",     0.997))
    swa_every     = int(os.environ.get("SWA_EVERY",       50))
    swa_lr_thresh = float(os.environ.get("SWA_LR_THRESH", 0.2))

    # Multi-prediction aux loss
    aux_weight_init = float(os.environ.get("AUX_WEIGHT_INIT", 1.0))
    aux_anneal_frac = float(os.environ.get("AUX_ANNEAL_FRAC", 0.5))

    # Late QAT
    qat_lr_thresh = float(os.environ.get("QAT_LR_THRESH", 0.15))

    # Novel: Compressor-Aware Training (CAT)
    cat_weight     = float(os.environ.get("CAT_WEIGHT", 0.005))
    cat_start_frac = float(os.environ.get("CAT_START_FRAC", 0.7))  # enable after 70% of training

    # Novel: Cascaded self-distillation across recurrence passes
    distill_weight = float(os.environ.get("DISTILL_WEIGHT", 0.3))
    distill_temp   = float(os.environ.get("DISTILL_TEMP", 2.0))

def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    """MuonEq-R: row-normalized Newton-Schulz orthogonalization."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    # MuonEq-R: row normalization before orthogonalization
    row_norms = X.norm(dim=-1, keepdim=True).clamp(min=eps)
    X = X / row_norms
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * (A @ A)) @ X
    return X.T if G.size(0) > G.size(1) else X

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr, momentum, backend_steps, nesterov=True):
        super().__init__(params, dict(lr=lr, momentum=momentum,
                                      backend_steps=backend_steps, nesterov=nesterov))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0
        for group in self.param_groups:
            params = group["params"]
            if not params: continue
            lr, mom, steps, nest = group["lr"], group["momentum"], group["backend_steps"], group["nesterov"]
            total = sum(p.numel() for p in params)
            flat  = torch.zeros(total, device=params[0].device, dtype=torch.bfloat16)
            curr  = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad
                    st = self.state[p]
                    if "buf" not in st:
                        st["buf"] = torch.zeros_like(g)
                    st["buf"].mul_(mom).add_(g)
                    if nest:
                        g = g.add(st["buf"], alpha=mom)
                    g = zeropower_via_newtonschulz5(g, steps=steps)
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    flat[curr:curr + p.numel()] = g.reshape(-1)
                curr += p.numel()
            if distributed:
                dist.all_reduce(flat, op=dist.ReduceOp.SUM)
            curr = 0
            for p in params:
                p.add_(flat[curr:curr + p.numel()].view_as(p).to(p.dtype), alpha=-lr)
                curr += p.numel()
        return loss

def build_sentencepiece_luts(sp, vocab_size, device):
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np        = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_np       = np.ones ((table_size,), dtype=np.bool_)
    for tid in range(sp_vocab_size):
        if sp.is_control(tid) or sp.is_unknown(tid) or sp.is_unused(tid): continue
        is_boundary_np[tid] = False
        if sp.is_byte(tid):
            base_bytes_np[tid] = 1; continue
        piece = sp.id_to_piece(tid)
        if piece.startswith("▁"):
            has_leading_space_np[tid] = True; piece = piece[1:]
        base_bytes_np[tid] = len(piece.encode("utf-8"))
    return (torch.tensor(base_bytes_np,        dtype=torch.int16, device=device),
            torch.tensor(has_leading_space_np, dtype=torch.bool,  device=device),
            torch.tensor(is_boundary_np,       dtype=torch.bool,  device=device))

def load_validation_tokens(pattern, seq_len):
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files: raise FileNotFoundError(f"No files for pattern: {pattern}")
    tokens = torch.cat([load_data_shard(f) for f in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0: raise ValueError(f"Val split too short for seq_len={seq_len}")
    return tokens[:usable + 1]

def eval_val(args, model, rank, world_size, device, grad_accum_steps,
             val_tokens, bb_lut, hs_lut, ib_lut):
    lbt = args.val_batch_size // (world_size * grad_accum_steps)
    if lbt < args.train_seq_len: raise ValueError("VAL_BATCH_SIZE too small")
    lbs = lbt // args.train_seq_len
    total_seqs = (val_tokens.numel() - 1) // args.train_seq_len
    seq_start  = (total_seqs * rank) // world_size
    seq_end    = (total_seqs * (rank + 1)) // world_size
    loss_sum = tok_sum = byte_sum = torch.zeros((), device=device, dtype=torch.float64)
    # reassign properly
    loss_sum  = torch.zeros((), device=device, dtype=torch.float64)
    tok_sum   = torch.zeros((), device=device, dtype=torch.float64)
    byte_sum  = torch.zeros((), device=device, dtype=torch.float64)
    model.eval()
    with torch.inference_mode():
        for bss in range(seq_start, seq_end, lbs):
            bse = min(bss + lbs, seq_end)
            rs, re = bss * args.train_seq_len, bse * args.train_seq_len + 1
            local = val_tokens[rs:re].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, args.train_seq_len)
            y = local[1:].reshape(-1, args.train_seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                bl = model(x, y).detach()
            ntok = float(y.numel())
            loss_sum += bl.to(torch.float64) * ntok
            tok_sum  += ntok
            prev = x.reshape(-1); tgt = y.reshape(-1)
            tb = bb_lut[tgt].to(torch.int16)
            tb += (hs_lut[tgt] & ~ib_lut[prev]).to(torch.int16)
            byte_sum += tb.to(torch.float64).sum()
    if dist.is_available() and dist.is_initialized():
        for t in (loss_sum, tok_sum, byte_sum):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
    vl = float((loss_sum / tok_sum).item())
    bpb = float((vl / math.log(2.0)) * (tok_sum / byte_sum).item())
    model.train()
    return vl, bpb

_CTRL = ("resid_mix", "q_gain", "skip_weights",
         "iter_attn_scales", "iter_mlp_scales", "bigram_proj")

def _is_ctrl(name): return any(p in name for p in _CTRL)

_EMBED = ("tok_emb", "bigram.table")
def _is_embed(name): return any(p in name for p in _EMBED)

def _pack_int6(arr: np.ndarray) -> tuple[bytes, int]:
    u = (arr.astype(np.int16) + 32).clip(0, 63).astype(np.uint8)
    pad = (4 - len(u) % 4) % 4
    if pad: u = np.concatenate([u, np.zeros(pad, np.uint8)])
    u = u.reshape(-1, 4)
    out = np.empty((len(u), 3), np.uint8)
    out[:, 0] = (u[:, 0] << 2) | (u[:, 1] >> 4)
    out[:, 1] = ((u[:, 1] & 0x0F) << 4) | (u[:, 2] >> 2)
    out[:, 2] = ((u[:, 2] & 0x03) << 6) | u[:, 3]
    return out.tobytes(), pad

def _unpack_int6(data: bytes, n: int) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8).reshape(-1, 3)
    u = np.empty((len(arr), 4), np.uint8)
    u[:, 0] = arr[:, 0] >> 2
    u[:, 1] = ((arr[:, 0] & 0x03) << 4) | (arr[:, 1] >> 4)
    u[:, 2] = ((arr[:, 1] & 0x0F) << 2) | (arr[:, 2] >> 6)
    u[:, 3] = arr[:, 2] & 0x3F
    return (u.reshape(-1)[:n].astype(np.int16) - 32).astype(np.int8)


_PERCENTILES = [0.999, 0.9995, 0.9999, 0.99999, 1.0]

def _quant_row_int6(row: np.ndarray):
    best_q = best_s = best_e = None
    for p in _PERCENTILES:
        clip = float(np.quantile(np.abs(row), p)) if row.size else 1.0
        clip = max(clip, 1e-8)
        s = clip / 31.0
        q = np.clip(np.round(np.clip(row, -clip, clip) / s), -31, 31).astype(np.int8)
        e = float(np.mean((row - q.astype(np.float32) * s) ** 2))
        if best_e is None or e < best_e:
            best_e, best_q, best_s = e, q, np.float16(s)
    return best_q, best_s

def _quant_row_int8(row: np.ndarray):
    clip = float(np.quantile(np.abs(row), 0.9999)) if row.size else 1.0
    clip = max(clip, 1e-8)
    s = clip / 127.0
    q = np.clip(np.round(np.clip(row, -clip, clip) / s), -127, 127).astype(np.int8)
    return q, np.float16(s)

_KEEP_NUMEL = 65_536

def quantize_model(state_dict: dict) -> dict:
    out = {}
    for name, t in state_dict.items():
        arr = t.detach().cpu().float().numpy()
        # Small / control params: store as fp16
        if _is_ctrl(name) or arr.size <= _KEEP_NUMEL:
            out[name] = {"t": "f16", "shape": list(arr.shape),
                         "d": arr.astype(np.float16).tobytes()}
            continue
        rows = arr.reshape(arr.shape[0], -1)
        use8 = _is_embed(name)
        qs, ss = [], []
        for row in rows:
            q, s = (_quant_row_int8(row) if use8 else _quant_row_int6(row))
            qs.append(q); ss.append(s)
        scales = np.array(ss, dtype=np.float16)
        if use8:
            q_arr = np.stack(qs)
            out[name] = {"t": "i8", "shape": list(arr.shape),
                         "scales": scales.tobytes(), "d": q_arr.tobytes()}
        else:
            packed_bytes, pad = _pack_int6(np.stack(qs).reshape(-1))
            out[name] = {"t": "i6", "shape": list(arr.shape), "pad": pad,
                         "scales": scales.tobytes(), "d": packed_bytes}
    return out

def dequantize_model(qdict: dict) -> dict[str, torch.Tensor]:
    out = {}
    for name, entry in qdict.items():
        shape = entry["shape"]
        typ   = entry["t"]
        if typ == "f16":
            t = torch.frombuffer(bytearray(entry["d"]), dtype=torch.float16).reshape(shape).float()
        else:
            scales = np.frombuffer(entry["scales"], dtype=np.float16).astype(np.float32)
            nr = shape[0]
            nc = int(np.prod(shape[1:])) if len(shape) > 1 else 1
            if typ == "i6":
                q = _unpack_int6(entry["d"], nr * nc).reshape(nr, nc).astype(np.float32)
            else:
                q = np.frombuffer(entry["d"], dtype=np.int8).reshape(nr, nc).astype(np.float32)
            t = torch.from_numpy((q * scales[:, None]).reshape(shape))
        out[name] = t
    return out

def compress_sd(sd: dict) -> bytes:
    raw = pickle.dumps(quantize_model(sd))
    if _HAS_ZSTD:
        return zstd.ZstdCompressor(level=22).compress(raw)
    return zlib.compress(raw, level=9)

def decompress_sd(data: bytes) -> dict:
    if _HAS_ZSTD:
        raw = zstd.ZstdDecompressor().decompress(data)
    else:
        raw = zlib.decompress(data)
    return dequantize_model(pickle.loads(raw))



def load_data_shard(file: Path) -> Tensor:
    hdr = np.fromfile(file, dtype="<i4", count=256)
    if hdr.size != 256 or int(hdr[0]) != 20240520 or int(hdr[1]) != 1:
        raise ValueError(f"Bad shard header: {file}")
    n = int(hdr[2])
    hb = 256 * np.dtype("<i4").itemsize
    if file.stat().st_size != hb + n * 2:
        raise ValueError(f"Shard size mismatch: {file}")
    return torch.from_numpy(np.fromfile(file, dtype="<u2", count=n, offset=hb).astype(np.uint16, copy=False))

class TokenStream:
    def __init__(self, pattern):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files: raise FileNotFoundError(f"No files for: {pattern}")
        self.fi, self.tokens, self.pos = 0, load_data_shard(self.files[0]), 0

    def _adv(self):
        self.fi = (self.fi + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.fi]); self.pos = 0

    def take(self, n):
        chunks, rem = [], n
        while rem > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0: self._adv(); continue
            k = min(rem, avail); chunks.append(self.tokens[self.pos:self.pos + k])
            self.pos += k; rem -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)

class DistributedTokenLoader:
    def __init__(self, pattern, rank, world_size, device):
        self.rank, self.ws, self.dev = rank, world_size, device
        self.stream = TokenStream(pattern)

    def next_batch(self, global_tokens, seq_len, gas):
        lt = global_tokens // (self.ws * gas)
        span = lt + 1
        chunk = self.stream.take(span * self.ws)
        local = chunk[self.rank * span:(self.rank + 1) * span].to(torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x.to(self.dev, non_blocking=True), y.to(self.dev, non_blocking=True)

class RMSNorm(nn.Module):
    def __init__(self, eps=None): super().__init__(); self.eps = eps
    def forward(self, x): return F.rms_norm(x, (x.size(-1),), eps=self.eps)

class CastedLinear(nn.Linear):
    _qat: bool = False

    def forward(self, x: Tensor) -> Tensor:
        w = self.weight.to(x.dtype)
        if CastedLinear._qat and w.ndim == 2 and w.numel() > _KEEP_NUMEL:
            with torch.no_grad():
                s = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 31.0
                wq = (w / s).round().clamp(-31, 31) * s
            w = w + (wq - w).detach()   # STE: gradient passes through unchanged
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, bias)

def _restore_fp32(module: nn.Module) -> None:
    with torch.no_grad():
        for name, p in module.named_parameters():
            if (p.ndim < 2 or _is_ctrl(name)) and p.dtype != torch.float32:
                p.data = p.data.float()

class Rotary(nn.Module):
    def __init__(self, dim, base=10000.0):
        super().__init__()
        self.register_buffer("inv_freq",
            1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)), persistent=False)
        self._sl = 0; self._cos = self._sin = None

    def forward(self, seq_len, device, dtype):
        if self._cos is None or self._sl != seq_len or self._cos.device != device:
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            f = torch.outer(t, self.inv_freq.to(device))
            self._cos = f.cos()[None, None]; self._sin = f.sin()[None, None]
            self._sl = seq_len
        return self._cos.to(dtype), self._sin.to(dtype)

def _apply_rope(x, cos, sin):
    h = x.size(-1) // 2
    x1, x2 = x[..., :h], x[..., h:]
    return torch.cat([x1 * cos + x2 * sin, -x2 * cos + x1 * sin], dim=-1)

class BigramHash(nn.Module):
    """Learned bigram-context bias added to token embeddings at input."""
    def __init__(self, vocab_size, num_buckets, proj_dim, model_dim):
        super().__init__()
        self.nb = num_buckets
        self.table = nn.Embedding(num_buckets, proj_dim)
        self.proj  = CastedLinear(proj_dim, model_dim, bias=False)
        nn.init.normal_(self.table.weight, std=0.02)

    def forward(self, ids: Tensor) -> Tensor:
        # ids: (B, T) int64
        prev = F.pad(ids[:, :-1], (1, 0), value=0)
        idx  = (prev * 1_234_567 + ids) % self.nb
        return self.proj(self.table(idx))

class CausalSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, rope_base, qk_gain_init, rope_dims):
        super().__init__()
        assert dim % num_heads == 0 and num_heads % num_kv_heads == 0
        self.nh, self.nkv = num_heads, num_kv_heads
        self.hd = dim // num_heads
        self.rope_dims = rope_dims
        kv = num_kv_heads * self.hd
        self.c_q  = CastedLinear(dim, dim,  bias=False)
        self.c_k  = CastedLinear(dim, kv,   bias=False)
        self.c_v  = CastedLinear(dim, kv,   bias=False)
        self.proj = CastedLinear(dim, dim,  bias=False); self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(rope_dims, base=rope_base)

    def forward(self, x: Tensor) -> Tensor:
        B, T, D = x.shape
        q = self.c_q(x).reshape(B, T, self.nh,  self.hd).transpose(1, 2)
        k = self.c_k(x).reshape(B, T, self.nkv, self.hd).transpose(1, 2)
        v = self.c_v(x).reshape(B, T, self.nkv, self.hd).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),)); k = F.rms_norm(k, (k.size(-1),))
        # Partial RoPE: rotate only first rope_dims of head_dim
        cos, sin = self.rotary(T, x.device, q.dtype)
        q = torch.cat([_apply_rope(q[..., :self.rope_dims], cos, sin), q[..., self.rope_dims:]], dim=-1)
        k = torch.cat([_apply_rope(k[..., :self.rope_dims], cos, sin), k[..., self.rope_dims:]], dim=-1)
        q = q * self.q_gain.to(q.dtype)[None, :, None, None]
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                            enable_gqa=(self.nkv != self.nh))
        return self.proj(y.transpose(1, 2).contiguous().reshape(B, T, D))

class MLP(nn.Module):
    def __init__(self, dim, mlp_mult):
        super().__init__()
        h = mlp_mult * dim
        self.fc   = CastedLinear(dim, h, bias=False)
        self.proj = CastedLinear(h, dim, bias=False); self.proj._zero_init = True

    def forward(self, x): return self.proj(torch.relu(self.fc(x)).square())

class RWKV7Block(nn.Module):
    """Novel: RWKV-7 Delta-Rule linear recurrence block.
    Replaces attention in middle layers with O(1)-memory linear recurrence.
    Uses data-dependent decay + in-context learning gate (delta rule).
    """
    def __init__(self, dim, num_heads):
        super().__init__()
        self.nh = num_heads
        self.hd = dim // num_heads
        self.c_r = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, dim, bias=False)
        self.c_v = CastedLinear(dim, dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False); self.proj._zero_init = True
        self.w_proj = CastedLinear(dim, dim, bias=False)  # decay gate
        self.ln = RMSNorm()

    def forward(self, x: Tensor) -> Tensor:
        B, T, D = x.shape
        r = torch.sigmoid(self.c_r(x))
        k = self.c_k(x)
        v = self.c_v(x)
        w = torch.sigmoid(self.w_proj(x)) * 0.99  # decay ∈ (0, 0.99)
        r = r.view(B, T, self.nh, self.hd)
        k = k.view(B, T, self.nh, self.hd)
        v = v.view(B, T, self.nh, self.hd)
        w = w.view(B, T, self.nh, self.hd)
        # Linear recurrence: state_t = diag(w_t) * state_{t-1} + v_t ⊗ k_t
        out = torch.zeros_like(v)
        state = torch.zeros(B, self.nh, self.hd, self.hd,
                            device=x.device, dtype=x.dtype)
        for t in range(T):
            state = state * w[:, t, :, :, None] + \
                    v[:, t, :, :, None] * k[:, t, :, None, :]
            out[:, t] = (r[:, t, :, :, None] * state).sum(-1)
        return self.proj(self.ln(out.reshape(B, T, D)))

class RecurrentBlock(nn.Module):
    """Parallel-residual recurrent block (GPT-J style).
    Attn + MLP read from the same pre-norm input. Weights shared across recurrence passes.
    """
    def __init__(self, dim, num_heads, num_kv_heads, mlp_mult, num_rec,
                 rope_base, qk_gain_init, rope_dims, use_rwkv=False):
        super().__init__()
        self.pre_norm = RMSNorm()
        self.use_rwkv = use_rwkv
        if use_rwkv:
            self.attn = RWKV7Block(dim, num_heads)
        else:
            self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads,
                                             rope_base, qk_gain_init, rope_dims)
        self.mlp  = MLP(dim, mlp_mult)
        self.resid_mix = nn.Parameter(torch.stack([torch.ones(dim), torch.zeros(dim)]).float())
        self.iter_attn_scales = nn.Parameter(torch.ones(num_rec, dim, dtype=torch.float32))
        self.iter_mlp_scales  = nn.Parameter(torch.ones(num_rec, dim, dtype=torch.float32))

    def forward(self, x: Tensor, x0: Tensor, it: int) -> Tensor:
        mix = self.resid_mix.to(x.dtype)
        x = mix[0][None, None] * x + mix[1][None, None] * x0
        as_ = self.iter_attn_scales[it].to(x.dtype)[None, None]
        ms  = self.iter_mlp_scales[it].to(x.dtype)[None, None]
        # Parallel residuals: both read from same pre-norm
        h = self.pre_norm(x)
        x = x + as_ * self.attn(h) + ms * self.mlp(h)
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size, num_unique_layers, num_recurrences, model_dim,
                 num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init,
                 rope_dims, logit_softcap, bigram_buckets, bigram_dim,
                 tied_embed_init_std=0.005, rwkv_indices=None,
                 distill_weight=0.0, distill_temp=2.0):
        super().__init__()
        self.num_unique   = num_unique_layers
        self.num_rec      = num_recurrences
        self.lsc          = logit_softcap
        self.distill_weight = distill_weight
        self.distill_temp   = distill_temp
        rwkv_set = set(rwkv_indices or [])

        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.bigram  = BigramHash(vocab_size, bigram_buckets, bigram_dim, model_dim)
        self.blocks  = nn.ModuleList([
            RecurrentBlock(model_dim, num_heads, num_kv_heads, mlp_mult,
                           num_recurrences, rope_base, qk_gain_init, rope_dims,
                           use_rwkv=(i in rwkv_set))
            for i in range(num_unique_layers)
        ])
        num_enc = num_unique_layers // 2
        self.skip_weights = nn.Parameter(torch.ones(num_enc, model_dim, dtype=torch.float32))
        self.final_norm   = RMSNorm()

        self.register_buffer("aux_weight", torch.tensor(0.0))

        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=tied_embed_init_std)
        for m in self.modules():
            if isinstance(m, nn.Linear) and getattr(m, "_zero_init", False):
                nn.init.zeros_(m.weight)

    def _logits(self, h: Tensor) -> Tensor:
        lp = F.linear(h, self.tok_emb.weight)
        return self.lsc * torch.tanh(lp / self.lsc)

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        x = self.tok_emb(input_ids) + self.bigram(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x

        num_enc  = self.num_unique // 2
        skips: list[Tensor] = []
        intermediate_logits: list[Tensor] = []

        for it in range(self.num_rec):
            is_enc = (it == 0)
            is_dec = (it == self.num_rec - 1)

            if is_enc:
                for i, blk in enumerate(self.blocks):
                    x = blk(x, x0, it)
                    if i < num_enc:
                        skips.append(x)
            elif is_dec:
                for i, blk in enumerate(self.blocks):
                    si = num_enc - 1 - i
                    if 0 <= si < len(skips):
                        x = x + self.skip_weights[si].to(x.dtype)[None, None] * skips[si]
                    x = blk(x, x0, it)
            else:
                for blk in self.blocks:
                    x = blk(x, x0, it)

            # Novel: capture intermediate logits for self-distillation
            if self.training and self.distill_weight > 0 and it < self.num_rec - 1:
                h_int = self.final_norm(x)
                intermediate_logits.append(self._logits(h_int.reshape(-1, x.size(-1))))

        hidden  = self.final_norm(x)
        logits  = self._logits(hidden.reshape(-1, x.size(-1)))
        targets = target_ids.reshape(-1)
        main_loss = F.cross_entropy(logits.float(), targets)

        if self.training:
            # Multi-prediction aux loss
            aw = float(self.aux_weight)
            if aw > 0.0:
                aux = hidden.new_tensor(0.0)
                for offset, w in ((2, 0.25), (4, 0.08), (8, 0.02)):
                    h_src = hidden[:, :-offset].reshape(-1, x.size(-1))
                    tgt   = target_ids[:, offset:].reshape(-1)
                    aux   = aux + w * F.cross_entropy(self._logits(h_src).float(), tgt)
                main_loss = main_loss + aw * aux

            # Novel: Cascaded self-distillation across recurrence passes
            if self.distill_weight > 0 and intermediate_logits:
                T2 = self.distill_temp ** 2
                teacher_probs = F.softmax(logits.detach().float() / self.distill_temp, dim=-1)
                distill_loss = logits.new_tensor(0.0)
                for il in intermediate_logits:
                    student_log = F.log_softmax(il.float() / self.distill_temp, dim=-1)
                    distill_loss = distill_loss + F.kl_div(
                        student_log, teacher_probs, reduction='batchmean') * T2
                main_loss = main_loss + self.distill_weight * distill_loss / len(intermediate_logits)

        return main_loss

# -------------------------------------------------------------------------
# TRAINING
# -------------------------------------------------------------------------

def main() -> None:
    global zeropower_via_newtonschulz5

    code_bytes = Path(__file__).read_bytes()
    args = Hyperparameters()

    zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)

    # ---- distributed + device setup ----
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank       = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if 8 % world_size != 0:
        raise ValueError(f"WORLD_SIZE={world_size} must divide 8")
    gas = 8 // world_size   # gradient accumulation steps
    grad_scale = 1.0 / gas

    if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device); dist.barrier()
    master = rank == 0

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import enable_flash_sdp, enable_cudnn_sdp, enable_mem_efficient_sdp, enable_math_sdp
    enable_flash_sdp(True); enable_cudnn_sdp(False)
    enable_mem_efficient_sdp(False); enable_math_sdp(False)

    logfile = None
    if master:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"
        print(logfile)

    def log0(msg, console=True):
        if not master: return
        if console: print(msg)
        if logfile:
            with open(logfile, "a", encoding="utf-8") as f: print(msg, file=f)

    log0(code_bytes.decode("utf-8"), console=False)

    # ---- seeding ----
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    # ---- tokenizer + val tokens ----
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(f"Vocab mismatch: got {sp.vocab_size()}, expected {args.vocab_size}")
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    bb_lut, hs_lut, ib_lut = build_sentencepiece_luts(sp, args.vocab_size, device)
    log0(f"val_bpb:enabled tokenizer={args.tokenizer_path}")
    log0(f"val_tokens:{val_tokens.numel() - 1}")

    # ---- model ----
    rwkv_indices = [int(x) for x in args.rwkv_block_indices.split(",") if x.strip()]
    base_model = GPT(
        vocab_size=args.vocab_size,
        num_unique_layers=args.num_unique_layers,
        num_recurrences=args.num_recurrences,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        rope_dims=args.rope_dims,
        logit_softcap=args.logit_softcap,
        bigram_buckets=args.bigram_buckets,
        bigram_dim=args.bigram_dim,
        tied_embed_init_std=args.tied_embed_init_std,
        rwkv_indices=rwkv_indices,
        distill_weight=args.distill_weight,
        distill_temp=args.distill_temp,
    ).to(device).bfloat16()

    for m in base_model.modules():
        if isinstance(m, CastedLinear): m.float()
    _restore_fp32(base_model)

    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"model_params:{n_params} unique_layers:{args.num_unique_layers} recurrences:{args.num_recurrences}")
    log0(f"effective_depth:{args.num_unique_layers * args.num_recurrences} model_dim:{args.model_dim}")

    compiled = torch.compile(base_model, dynamic=False, fullgraph=True)
    model: nn.Module = DDP(compiled, device_ids=[local_rank], broadcast_buffers=False) if distributed else compiled

    # ---- optimizers ----
    # Muon for 2D weight matrices in transformer blocks and bigram projection
    # Adam for embeddings, control tensors, and all 1D parameters
    block_np = list(base_model.blocks.named_parameters())
    mat_params = [p for nm, p in block_np
                  if p.ndim == 2 and not _is_ctrl(nm)]
    sca_params = [p for nm, p in block_np
                  if p.ndim < 2 or _is_ctrl(nm)]
    sca_params += [base_model.skip_weights]
    # bigram projection goes through Muon (it's a matrix)
    mat_params += [base_model.bigram.proj.weight]

    optimizer_tok = torch.optim.AdamW(
        [{"params": [base_model.tok_emb.weight,
                     base_model.bigram.table.weight],
          "lr": args.embed_lr, "base_lr": args.embed_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps,
        weight_decay=args.weight_decay, fused=True)

    optimizer_muon = Muon(mat_params, lr=args.matrix_lr,
                          momentum=args.muon_momentum,
                          backend_steps=args.muon_backend_steps)
    for g in optimizer_muon.param_groups:
        g["base_lr"] = args.matrix_lr

    optimizer_sca = torch.optim.AdamW(
        [{"params": sca_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps,
        weight_decay=0.0, fused=True)   # no WD on scalars/control params

    optimizers = [optimizer_tok, optimizer_muon, optimizer_sca]

    def zero_grad():
        for o in optimizers: o.zero_grad(set_to_none=True)

    # ---- EMA shadow copy ----
    ema_state: dict[str, Tensor] = {
        k: v.detach().clone().float()
        for k, v in base_model.state_dict().items()
        if v.is_floating_point()
    }

    def update_ema():
        with torch.no_grad():
            sd = base_model.state_dict()
            for k, v in sd.items():
                if k in ema_state:
                    ema_state[k].mul_(args.ema_decay).add_(v.float(), alpha=1 - args.ema_decay)

    # ---- SWA accumulator ----
    swa_state: dict[str, Tensor] = {}
    swa_count = 0

    def accumulate_swa():
        nonlocal swa_count
        swa_count += 1
        sd = base_model.state_dict()
        if not swa_state:
            for k, v in sd.items():
                if v.is_floating_point():
                    swa_state[k] = v.detach().clone().float()
        else:
            for k, v in sd.items():
                if k in swa_state:
                    swa_state[k] += (v.float() - swa_state[k]) / swa_count

    def best_averaged_sd() -> dict:
        """Return SWA average if available and better than EMA, else EMA."""
        return swa_state if swa_count > 0 else ema_state

    # ---- LR schedule ----
    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    def lr_scale(step, elapsed_ms):
        if args.warmdown_iters <= 0: return 1.0
        if max_wallclock_ms is None:
            wd_start = max(args.iterations - args.warmdown_iters, 0)
            if step >= wd_start:
                return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0)
            return 1.0
        step_ms = elapsed_ms / max(step, 1)
        wd_ms   = args.warmdown_iters * step_ms
        rem_ms  = max(max_wallclock_ms - elapsed_ms, 0.0)
        return min(rem_ms / max(wd_ms, 1e-9), 1.0)

    def set_lr(scale):
        for o in optimizers:
            for g in o.param_groups:
                g["lr"] = g["base_lr"] * scale

    # ---- warmup ----
    if args.warmup_steps > 0:
        init_sd  = {k: v.detach().cpu().clone() for k, v in base_model.state_dict().items()}
        init_opt = [copy.deepcopy(o.state_dict()) for o in optimizers]
        loader_wu = DistributedTokenLoader(args.train_files, rank, world_size, device)
        model.train()
        for ws in range(args.warmup_steps):
            zero_grad()
            for _ in range(gas):
                if distributed: model.require_backward_grad_sync = (_ == gas - 1)
                x, y = loader_wu.next_batch(args.train_batch_tokens, args.train_seq_len, gas)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    (model(x, y) * grad_scale).backward()
            for o in optimizers: o.step()
        log0(f"warmup_done:{args.warmup_steps}")
        base_model.load_state_dict(init_sd, strict=True)
        for o, st in zip(optimizers, init_opt): o.load_state_dict(st)
        zero_grad()
        if distributed: model.require_backward_grad_sync = True
        # Re-init EMA after warmup reset
        for k, v in base_model.state_dict().items():
            if k in ema_state: ema_state[k].copy_(v.float())

    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    # ---- main training loop ----
    training_ms = 0.0
    stop_after: int | None = None
    torch.cuda.synchronize(); t0 = time.perf_counter()
    step = 0

    while True:
        last_step = (step == args.iterations or
                     (stop_after is not None and step >= stop_after))
        do_val = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)

        if do_val:
            torch.cuda.synchronize(); training_ms += 1000.0 * (time.perf_counter() - t0)
            vl, vbpb = eval_val(args, model, rank, world_size, device, gas,
                                 val_tokens, bb_lut, hs_lut, ib_lut)
            log0(f"step:{step}/{args.iterations} val_loss:{vl:.4f} val_bpb:{vbpb:.4f} "
                 f"train_time:{training_ms:.0f}ms step_avg:{training_ms / max(step,1):.2f}ms")
            torch.cuda.synchronize(); t0 = time.perf_counter()

        if last_step: break

        # ---- forward + backward ----
        model.train(); zero_grad()
        for micro in range(gas):
            if distributed: model.require_backward_grad_sync = (micro == gas - 1)
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, gas)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(x, y)
            (loss * grad_scale).backward()

        # ---- grad clip ----
        if args.grad_clip_norm > 0:
            all_params = [p for o in optimizers for g in o.param_groups for p in g["params"]]
            nn.utils.clip_grad_norm_(all_params, args.grad_clip_norm)

        # ---- compute elapsed + lr_scale ----
        torch.cuda.synchronize(); elapsed_now = training_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_scale(step + 1, elapsed_now)
        set_lr(scale)

        # ---- Muon momentum warmup ----
        if step < args.muon_momentum_warmup_steps:
            frac = step / max(args.muon_momentum_warmup_steps, 1)
            new_mom = args.muon_momentum_warmup_start + frac * (args.muon_momentum - args.muon_momentum_warmup_start)
            for g in optimizer_muon.param_groups: g["momentum"] = new_mom

        # ---- optimizer step ----
        for o in optimizers: o.step()
        zero_grad()

        # ---- EMA update (every step) ----
        update_ema()

        # ---- SWA snapshot (during warmdown) ----
        if scale < args.swa_lr_thresh and step % args.swa_every == 0:
            accumulate_swa()

        # ---- annealed aux loss weight ----
        if scale >= 1.0:
            aw = args.aux_weight_init
        else:
            aw = max(0.0, args.aux_weight_init * (scale / max(1.0 - args.aux_anneal_frac, 1e-9))
                     if scale > (1.0 - args.aux_anneal_frac) else 0.0)
        base_model.aux_weight.fill_(aw)

        # ---- late QAT: enable STE fake-quant during final warmdown ----
        if scale < args.qat_lr_thresh and not CastedLinear._qat:
            CastedLinear._qat = True
            log0("qat:enabled")

        if step % args.train_log_every == 0:
            log0(f"step:{step} loss:{float(loss):.4f} lr_scale:{scale:.4f} ema_decay:{args.ema_decay} "
                 f"swa_count:{swa_count} aux_weight:{aw:.3f} qat:{CastedLinear._qat}")

        step += 1

        # ---- wallclock cutoff ----
        torch.cuda.synchronize(); elapsed_now = training_ms + 1000.0 * (time.perf_counter() - t0)
        if max_wallclock_ms is not None and elapsed_now >= max_wallclock_ms:
            log0(f"wallclock_cutoff:{elapsed_now:.0f}ms")
            break

    # ---- final: apply averaged weights ----
    avg_sd = best_averaged_sd()
    current_sd = base_model.state_dict()
    merged = {k: avg_sd[k].to(current_sd[k].dtype) if k in avg_sd else current_sd[k]
              for k in current_sd}
    base_model.load_state_dict(merged, strict=True)

    # ---- final eval with averaged model ----
    if master:
        torch.cuda.synchronize(); training_ms += 1000.0 * (time.perf_counter() - t0)
        vl, vbpb = eval_val(args, model, rank, world_size, device, gas,
                             val_tokens, bb_lut, hs_lut, ib_lut)
        log0(f"final_avg val_loss:{vl:.4f} val_bpb:{vbpb:.4f}")

        # ---- save artifact ----
        artifact_path = f"logs/{args.run_id}_model.bin"
        compressed = compress_sd(base_model.state_dict())
        Path(artifact_path).write_bytes(compressed)
        total_bytes = len(code_bytes) + len(compressed)
        log0(f"final_int6_zstd_roundtrip code_bytes:{len(code_bytes)} "
             f"model_bytes:{len(compressed)} total_bytes:{total_bytes} "
             f"total_mb:{total_bytes / 1e6:.3f}")

        # ---- roundtrip check: decompress + reload + re-eval ----
        rt_sd = decompress_sd(compressed)
        base_model.load_state_dict(rt_sd, strict=False)
        vl2, vbpb2 = eval_val(args, model, rank, world_size, device, gas,
                               val_tokens, bb_lut, hs_lut, ib_lut)
        log0(f"roundtrip_check val_loss:{vl2:.4f} val_bpb:{vbpb2:.4f} "
             f"delta_bpb:{vbpb2 - vbpb:.5f}")
        log0(f"submission_score:{vbpb2:.4f}")

    if distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()

