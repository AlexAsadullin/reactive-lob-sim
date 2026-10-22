"""Conditional autoregressive order-flow model (GRU over tokenised events).

At step i the input is the sum of the four field embeddings of event i plus a linear map of its conditioning
vector. A GRU summarises the history in h_i, and four heads predict event i+1 in a chained factorisation:

p(type, dt, offset, size | h) = p(type | h) p(dt | h, type) p(offset | h, type) p(size | h, type),

so offset, size and timing can depend on the event type. Training minimises the summed cross-entropy
(the offset term is masked for market orders, which have no offset). The design follows the finding of
LOB-Bench (Nagy et al. 2025) that autoregressive message models are the strongest generator family; it is
kept small enough to train on a CPU in seconds.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import structlog
import torch
from torch import nn

from reactive_lob_sim.generative.tokenizer import Tokenizer, TokenStream
from reactive_lob_sim.params import MKT_BUY, MKT_SELL, N_EVENT_TYPES
from reactive_lob_sim.settings import GenerativeSettings

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class ModelShape:
    n_offsets: int
    n_sizes: int
    n_dts: int
    cond_dim: int
    embed_dim: int
    hidden_dim: int
    num_layers: int


class OrderFlowGRU(nn.Module):
    def __init__(self, shape: ModelShape) -> None:
        super().__init__()
        self.shape = shape
        e, h = shape.embed_dim, shape.hidden_dim
        self.emb_type = nn.Embedding(N_EVENT_TYPES, e)
        self.emb_off = nn.Embedding(shape.n_offsets, e)
        self.emb_size = nn.Embedding(shape.n_sizes, e)
        self.emb_dt = nn.Embedding(shape.n_dts, e)
        self.cond = nn.Linear(shape.cond_dim, e)
        self.gru = nn.GRU(e, h, num_layers=shape.num_layers, batch_first=True)
        self.head_type = nn.Linear(h, N_EVENT_TYPES)
        self.next_type = nn.Embedding(N_EVENT_TYPES, h)
        self.head_dt = nn.Linear(h, shape.n_dts)
        self.head_off = nn.Linear(h, shape.n_offsets)
        self.head_size = nn.Linear(h, shape.n_sizes)

    def embed(
        self, kind: torch.Tensor, off: torch.Tensor, size: torch.Tensor, dt: torch.Tensor, cond: torch.Tensor
    ) -> torch.Tensor:
        out: torch.Tensor = (
            self.emb_type(kind) + self.emb_off(off) + self.emb_size(size) + self.emb_dt(dt) + self.cond(cond)
        )
        return out

    def forward(
        self,
        kind: torch.Tensor,
        off: torch.Tensor,
        size: torch.Tensor,
        dt: torch.Tensor,
        cond: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out, hidden_out = self.gru(self.embed(kind, off, size, dt, cond), hidden)
        return out, hidden_out

    def conditional_heads(
        self, h: torch.Tensor, next_kind: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = torch.tanh(h + self.next_type(next_kind))
        return self.head_dt(z), self.head_off(z), self.head_size(z)


def _loss(model: OrderFlowGRU, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    out, _ = model(
        batch["kind"][:, :-1],
        batch["off"][:, :-1],
        batch["size"][:, :-1],
        batch["dt"][:, :-1],
        batch["cond"][:, :-1],
    )
    nk = batch["kind"][:, 1:]
    ce = nn.functional.cross_entropy
    loss_type = ce(model.head_type(out).flatten(0, 1), nk.flatten())
    dt_logits, off_logits, size_logits = model.conditional_heads(out, nk)
    loss_dt = ce(dt_logits.flatten(0, 1), batch["dt"][:, 1:].flatten())
    loss_size = ce(size_logits.flatten(0, 1), batch["size"][:, 1:].flatten())
    has_offset = ((nk != MKT_BUY) & (nk != MKT_SELL)).flatten()
    off_all = ce(off_logits.flatten(0, 1), batch["off"][:, 1:].flatten(), reduction="none")
    loss_off = (off_all * has_offset).sum() / has_offset.sum().clamp(min=1)
    return loss_type + loss_dt + loss_size + loss_off


def _windows(stream: TokenStream, seq_len: int) -> dict[str, torch.Tensor]:
    n = (stream.kind.shape[0] - 1) // seq_len
    if n < 1:
        raise ValueError("token stream shorter than one training window")
    span = n * seq_len + 1
    idx = np.arange(n)[:, None] * seq_len + np.arange(seq_len + 1)[None, :]
    idx = np.minimum(idx, span - 1)
    return {
        "kind": torch.as_tensor(stream.kind[idx]),
        "off": torch.as_tensor(stream.offset[idx]),
        "size": torch.as_tensor(stream.size[idx]),
        "dt": torch.as_tensor(stream.dt[idx]),
        "cond": torch.as_tensor(stream.cond[idx]),
    }


@dataclass(frozen=True)
class TrainReport:
    epochs: int
    train_loss: list[float]
    valid_loss: float
    n_train_windows: int
    n_valid_windows: int


def train_model(
    tok: Tokenizer, stream: TokenStream, cfg: GenerativeSettings, seed: int, valid_fraction: float = 0.1
) -> tuple[OrderFlowGRU, TrainReport]:
    """Train on windows of ``seq_len + 1`` tokens; the last ``valid_fraction`` of windows is held out."""
    torch.manual_seed(seed)
    torch.set_num_threads(cfg.torch_threads)
    shape = ModelShape(
        tok.n_offsets, tok.n_sizes, tok.n_dts, tok.cond_dim, cfg.embed_dim, cfg.hidden_dim, cfg.num_layers
    )
    model = OrderFlowGRU(shape)
    data = _windows(stream, cfg.seq_len)
    n = data["kind"].shape[0]
    n_valid = max(int(n * valid_fraction), 1) if n > 1 else 0
    train = {k: v[: n - n_valid] for k, v in data.items()}
    valid = {k: v[n - n_valid :] for k, v in data.items()}
    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    gen = torch.Generator().manual_seed(seed)
    losses = []
    n_train = train["kind"].shape[0]
    for epoch in range(cfg.epochs):
        model.train()
        perm = torch.randperm(n_train, generator=gen)
        total, count = 0.0, 0
        for start in range(0, n_train, cfg.batch_size):
            sel = perm[start : start + cfg.batch_size]
            loss = _loss(model, {k: v[sel] for k, v in train.items()})
            opt.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call]  # torch ships Tensor.backward untyped
            opt.step()
            total += loss.item() * sel.numel()
            count += sel.numel()
        losses.append(total / max(count, 1))
        log.info("generative_epoch", epoch=epoch, loss=losses[-1])
    model.eval()
    with torch.no_grad():
        valid_loss = float(_loss(model, valid)) if n_valid else float("nan")
    return model, TrainReport(cfg.epochs, losses, valid_loss, n_train, n_valid)


def save_model(model: OrderFlowGRU, tok: Tokenizer, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), directory / "model.pt")
    meta = {
        "shape": asdict(model.shape),
        "tokenizer": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in asdict(tok).items()},
    }
    (directory / "model.json").write_text(json.dumps(meta, indent=2))


def load_model(directory: Path) -> tuple[OrderFlowGRU, Tokenizer]:
    meta = json.loads((directory / "model.json").read_text())
    model = OrderFlowGRU(ModelShape(**meta["shape"]))
    model.load_state_dict(torch.load(directory / "model.pt", weights_only=True))
    model.eval()
    t = meta["tokenizer"]
    tok = Tokenizer(
        t["offset_clip"],
        np.asarray(t["size_edges"]),
        np.asarray(t["size_values"]),
        np.asarray(t["dt_edges"]),
        np.asarray(t["vol_edges"]),
        t["funding_mean"],
        t["funding_std"],
        t["oi_std"],
        t["vol_window_s"],
        t["dt_floor_s"],
    )
    return model, tok
