from typing import Optional

import torch
import torch.nn.functional as F


def contrastive_loss(z_prod: torch.Tensor, z_templ: torch.Tensor, temperature: float) -> torch.Tensor:
    # Normalize embeddings for cosine similarity
    z_prod = F.normalize(z_prod, dim=-1)
    z_templ = F.normalize(z_templ, dim=-1)
    logits = z_prod @ z_templ.t()
    logits = logits / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    loss_p2t = F.cross_entropy(logits, labels)
    loss_t2p = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_p2t + loss_t2p)


def listwise_rank_loss(
    logits: torch.Tensor,
    pos_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # logits: [B, B] where row i scores templates for product i
    log_probs = F.log_softmax(logits, dim=-1)
    bsz = logits.size(0)
    if pos_mask is None:
        pos_mask = torch.eye(bsz, device=logits.device, dtype=torch.bool)
    losses = []
    for i in range(bsz):
        pos_idx = pos_mask[i].nonzero(as_tuple=False).squeeze(-1)
        if pos_idx.numel() == 0:
            continue
        losses.append(-log_probs[i, pos_idx].mean())
    if not losses:
        return torch.tensor(0.0, device=logits.device)
    return torch.stack(losses).mean()


def mlm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    # logits: [B, L, V], labels: [B, L] with -100 for ignore
    return F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)


def kl_divergence(log_probs: torch.Tensor, ref_log_probs: torch.Tensor) -> torch.Tensor:
    # KL(pi || pref) = sum pi * (log pi - log pref)
    probs = log_probs.exp()
    return (probs * (log_probs - ref_log_probs)).sum(dim=-1).mean()
