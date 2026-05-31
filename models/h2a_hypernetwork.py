import torch
import torch.nn as nn


class DualBranchHyperNetwork(nn.Module):
    def __init__(
        self,
        num_clients: int,
        num_feature_layers: int,
        embed_dim: int,
        shared_hidden_dim: int,
        branch_hidden_dim: int,
    ):
        super().__init__()
        if num_clients <= 0:
            raise ValueError("num_clients must be positive.")
        if num_feature_layers <= 0:
            raise ValueError("num_feature_layers must be positive.")

        self.num_clients = num_clients
        self.num_feature_layers = num_feature_layers

        self.embedding = nn.Embedding(num_clients + 1, embed_dim)
        self.shared = nn.Sequential(
            nn.Linear(embed_dim, shared_hidden_dim),
            nn.ReLU(),
        )
        self.h_p = nn.Sequential(
            nn.Linear(shared_hidden_dim, branch_hidden_dim),
            nn.ReLU(),
            nn.Linear(branch_hidden_dim, num_clients * num_feature_layers),
        )
        self.h_r = nn.Sequential(
            nn.Linear(shared_hidden_dim, branch_hidden_dim),
            nn.LayerNorm(branch_hidden_dim),
            nn.ReLU(),
            nn.Linear(branch_hidden_dim, branch_hidden_dim),
            nn.LayerNorm(branch_hidden_dim),
            nn.ReLU(),
            nn.Linear(branch_hidden_dim, num_clients * num_feature_layers),
        )

    def forward(self, client_id):
        if not torch.is_tensor(client_id):
            client_id = torch.tensor([client_id], dtype=torch.long, device=self.embedding.weight.device)
        else:
            client_id = client_id.to(device=self.embedding.weight.device, dtype=torch.long).reshape(-1)

        emb = self.embedding(client_id)
        z = self.shared(emb)
        logits_p = self.h_p(z).reshape(-1, self.num_clients, self.num_feature_layers)
        logits_r = self.h_r(z).reshape(-1, self.num_clients, self.num_feature_layers)

        if logits_p.size(0) == 1:
            return logits_p.squeeze(0), logits_r.squeeze(0), z.squeeze(0)
        return logits_p, logits_r, z
