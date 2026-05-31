import copy
import torch
from models.cnn_lstm import CNNLSTMModel
from utils.aggregation import fedavg


def is_head_param(name: str, head_prefixes, head_exact_names=None) -> bool:
    head_exact_names = head_exact_names or []
    if name in head_exact_names:
        return True
    return any(name.startswith(prefix + ".") for prefix in head_prefixes)


class FedServer:
    def __init__(self, input_dim: int, cfg):
        self.cfg = copy.deepcopy(cfg)
        self.global_model = CNNLSTMModel(
            input_dim=input_dim,
            output_dim=self.cfg.data.horizon,
            cfg=self.cfg.model,
        ).to(self.cfg.train.device)
        self.global_rc = None

    def get_global_state(self):
        return copy.deepcopy(self.global_model.state_dict())

    def set_global_state(self, state_dict):
        self.global_model.load_state_dict(copy.deepcopy(state_dict))

    def get_global_rc(self):
        if self.global_rc is None:
            return None
        return self.global_rc.clone().detach()

    def get_head_prefixes(self):
        return getattr(self.cfg.federated, "head_param_prefixes", ["fc1", "fc2"])

    def get_head_exact_names(self):
        return getattr(self.cfg.federated, "head_param_exact_names", [])

    def aggregate(self, client_updates):
        sample_counts = [item["num_samples"] for item in client_updates]
        use_head_personalization = getattr(self.cfg.federated, "use_head_personalization", False)
        head_prefixes = self.get_head_prefixes()
        head_exact_names = self.get_head_exact_names()

        if use_head_personalization:
            old_global_state = self.get_global_state()
            total_samples = float(sum(sample_counts))
            if total_samples <= 0:
                raise ValueError("Cannot aggregate model with non-positive sample count.")

            new_global_state = {}
            for name, old_tensor in old_global_state.items():
                if not is_head_param(name, head_prefixes, head_exact_names):
                    avg_tensor = old_tensor.detach().clone() * 0.0
                    for item, n in zip(client_updates, sample_counts):
                        tensor = item["state_dict"][name].to(device=old_tensor.device, dtype=old_tensor.dtype)
                        avg_tensor += tensor * (float(n) / total_samples)
                    new_global_state[name] = avg_tensor
                    continue

                numerator = torch.zeros_like(old_tensor, dtype=old_tensor.dtype)
                denominator = torch.zeros_like(old_tensor, dtype=old_tensor.dtype)
                for item, n in zip(client_updates, sample_counts):
                    tensor = item["state_dict"][name].to(device=old_tensor.device, dtype=old_tensor.dtype)
                    head_mask = item.get("head_mask")
                    if head_mask is None or name not in head_mask:
                        mask = torch.zeros_like(old_tensor, dtype=old_tensor.dtype)
                    else:
                        mask = head_mask[name].to(device=old_tensor.device, dtype=old_tensor.dtype)
                    collab_mask = 1.0 - mask
                    numerator += tensor * collab_mask * float(n)
                    denominator += collab_mask * float(n)

                new_tensor = old_tensor.detach().clone()
                valid = denominator > 0
                new_tensor[valid] = numerator[valid] / denominator[valid]
                new_global_state[name] = new_tensor
        else:
            state_dicts = [item["state_dict"] for item in client_updates]
            new_global_state = fedavg(state_dicts, sample_counts)

        self.set_global_state(new_global_state)

        rc_updates = [
            item for item in client_updates
            if "local_rc" in item and item["local_rc"] is not None
        ]
        if len(rc_updates) == 0:
            return

        total_samples = float(sum(item["num_samples"] for item in rc_updates))
        if total_samples <= 0:
            raise ValueError("Cannot aggregate RC with non-positive sample count.")

        global_rc = None
        for item in rc_updates:
            local_rc = item["local_rc"].detach().cpu().float()
            weight = float(item["num_samples"]) / total_samples
            if global_rc is None:
                global_rc = torch.zeros_like(local_rc)
            global_rc += local_rc * weight

        self.global_rc = global_rc
