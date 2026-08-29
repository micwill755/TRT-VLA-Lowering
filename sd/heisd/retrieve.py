import torch
import torch.nn.functional as F


class ActionBank:
    def __init__(self, max_entries=2048):
        self.max_entries = max_entries
        self.keys = []
        self.chunks = []

    def add(self, lm_hidden: torch.Tensor, actions: torch.Tensor):
        key = F.normalize(lm_hidden.float().mean(dim=1).squeeze(0), dim=-1)
        self.keys.append(key.detach().cpu())
        self.chunks.append(actions.detach().cpu())
        if len(self.keys) > self.max_entries:
            self.keys.pop(0)
            self.chunks.pop(0)

    def query(self, lm_hidden: torch.Tensor):
        if not self.keys:
            return None, 0.0

        q = F.normalize(lm_hidden.float().mean(dim=1).squeeze(0), dim=-1).cpu()
        sims = torch.stack([F.cosine_similarity(q, k, dim=0) for k in self.keys])
        i = int(sims.argmax())
        return self.chunks[i], float(sims[i])
