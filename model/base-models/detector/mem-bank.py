import torch
import torch.nn.functional as F

class MemoryBank:
    def __init__(self, bank_size=4096, feat_dim=256, temp=0.07):
        self.bank_size = bank_size
        self.feat_dim = feat_dim
        self.temp = temp
        # float32 CPU bank
        self.bank = torch.randn(bank_size, feat_dim, dtype=torch.float32)
        self.label_bank = torch.zeros(bank_size, dtype=torch.long)
        self.ptr = 0

    def enqueue(self, feats: torch.Tensor, labels: torch.Tensor):
        b = feats.size(0)
        idx = (self.ptr + torch.arange(b, device=feats.device)) % self.bank_size
        idx_cpu = idx.cpu()
        feats_cpu = feats.detach().cpu().to(self.bank.dtype)
        labels_cpu = labels.detach().cpu().long()
        self.bank[idx_cpu] = feats_cpu
        self.label_bank[idx_cpu] = labels_cpu
        self.ptr = (self.ptr + b) % self.bank_size

    def contrastive_loss(self, feats: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        feats:  [N,D]  on GPU
        labels: [N]    long on GPU
        """
        # 1) bring bank → GPU & same dtype
        bank_feats  = self.bank.to(feats.device).to(feats.dtype)      # [M,D]
        bank_labels = self.label_bank.to(feats.device)               # [M]

        # 2) normalize
        feats_norm = F.normalize(feats, dim=1)        # [N,D]
        bank_norm  = F.normalize(bank_feats, dim=1)   # [M,D]

        # 3) cosine sims and scale by temperature
        logits = feats_norm @ bank_norm.t() / self.temp  # [N,M]

        # 4) for each query i, positives = bank_indices with same label
        labels_q = labels.unsqueeze(1)           # [N,1]
        labels_b = bank_labels.unsqueeze(0)      # [1,M]
        pos_mask = labels_q.eq(labels_b)         # [N,M], boolean

        # 5) InfoNCE:  -log( ∑_{j∈Pos_i} exp(logits[i,j]) / ∑_{k} exp(logits[i,k]) )
        exp_logits = logits.exp()                # [N,M]
        # numerator: sum over positives (if none, we’ll skip those anchors)
        pos_exp = (exp_logits * pos_mask.float()).sum(dim=1)    # [N]
        all_exp = exp_logits.sum(dim=1)                        # [N]

        # to avoid division by zero when no positives: mask them out
        # compute loss_i only if pos_exp>0
        nonzero_pos = pos_exp > 0
        loss = torch.zeros_like(pos_exp)
        loss[nonzero_pos] = - (pos_exp[nonzero_pos] / all_exp[nonzero_pos]).log()

        # 6) return mean over valid anchors
        return loss[nonzero_pos].mean()
