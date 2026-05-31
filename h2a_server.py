import copy
from contextlib import nullcontext
from collections import OrderedDict
from typing import Dict, List, Tuple

import torch

from models.cnn_lstm import CNNLSTMModel
from models.h2a_hypernetwork import DualBranchHyperNetwork


class H2AServer:
    def __init__(self, input_dim: int, cfg, num_clients: int):
        self.cfg = copy.deepcopy(cfg)
        self.num_clients = int(num_clients)
        if self.num_clients <= 0:
            raise ValueError("num_clients must be positive.")

        self.device = torch.device(self.cfg.train.device)
        init_model = CNNLSTMModel(
            input_dim=input_dim,
            output_dim=self.cfg.data.horizon,
            cfg=self.cfg.model,
        ).to(self.device)
        init_state = self.clone_state_dict(init_model.state_dict())

        self.param_types: Dict[str, str] = {}
        self.param_to_layer_idx: Dict[str, int] = {}
        self.feature_layer_names: List[str] = []
        self.feature_param_names: List[str] = []
        self._build_param_mappings(init_state)

        self.client_states = {
            client_id: self.clone_state_dict(init_state)
            for client_id in range(1, self.num_clients + 1)
        }
        self.pending_client_states = {}
        self.round_snapshot = None
        self.round = 0
        self.importance_matrix = torch.eye(self.num_clients, dtype=torch.float32)

        self.hypernet = DualBranchHyperNetwork(
            num_clients=self.num_clients,
            num_feature_layers=len(self.feature_layer_names),
            embed_dim=self.cfg.federated.h2a_embed_dim,
            shared_hidden_dim=self.cfg.federated.h2a_shared_hidden_dim,
            branch_hidden_dim=self.cfg.federated.h2a_branch_hidden_dim,
        ).to(self.device)
        self.hyper_optimizer = torch.optim.Adam(
            self.hypernet.parameters(),
            lr=self.cfg.federated.h2a_meta_lr,
        )

    @staticmethod
    def clone_state_dict(state_dict):
        cloned = OrderedDict()
        for name, value in state_dict.items():
            if torch.is_tensor(value):
                cloned[name] = value.detach().cpu().clone()
            else:
                cloned[name] = copy.deepcopy(value)
        return cloned

    def classify_param(self, name: str) -> str:
        feature_matches = [
            prefix for prefix in self.cfg.federated.h2a_feature_param_prefixes
            if name.startswith(prefix + ".")
        ]
        head_matches = [
            prefix for prefix in self.cfg.federated.h2a_head_param_prefixes
            if name.startswith(prefix + ".")
        ]

        if feature_matches and head_matches:
            raise ValueError(
                f"H2A parameter {name} matches both feature prefixes "
                f"{feature_matches} and head prefixes {head_matches}."
            )
        if feature_matches:
            return "feature"
        if head_matches:
            return "head"

        policy = getattr(self.cfg.federated, "h2a_unmatched_param_policy", "error").lower()
        if policy in ("feature", "head"):
            return policy
        if policy != "error":
            raise ValueError(
                "h2a_unmatched_param_policy must be one of: error, feature, head."
            )
        raise ValueError(
            f"H2A parameter {name} does not match feature or head prefixes. "
            "Set h2a_unmatched_param_policy to 'feature' or 'head' if this is intended."
        )

    def _matched_feature_prefix(self, name: str):
        matches = [
            prefix for prefix in self.cfg.federated.h2a_feature_param_prefixes
            if name.startswith(prefix + ".")
        ]
        if not matches:
            return None
        return max(matches, key=len)

    def _fallback_layer_name(self, name: str) -> str:
        return name.split(".", 1)[0] if "." in name else name

    def _build_param_mappings(self, state_dict):
        configured_layers = []
        fallback_layers = []

        for name, tensor in state_dict.items():
            param_type = self.classify_param(name)
            self.param_types[name] = param_type
            if param_type != "feature":
                continue
            if not torch.is_tensor(tensor) or not torch.is_floating_point(tensor):
                continue

            matched_prefix = self._matched_feature_prefix(name)
            layer_name = matched_prefix if matched_prefix is not None else self._fallback_layer_name(name)
            if matched_prefix is not None:
                if layer_name not in configured_layers:
                    configured_layers.append(layer_name)
            elif layer_name not in fallback_layers:
                fallback_layers.append(layer_name)

        prefix_order = list(self.cfg.federated.h2a_feature_param_prefixes)
        configured_layers = sorted(
            configured_layers,
            key=lambda layer: prefix_order.index(layer) if layer in prefix_order else len(prefix_order),
        )
        self.feature_layer_names = configured_layers + fallback_layers
        if len(self.feature_layer_names) == 0:
            raise ValueError("H2A found no floating feature parameters to aggregate.")

        layer_to_idx = {name: idx for idx, name in enumerate(self.feature_layer_names)}
        for name, tensor in state_dict.items():
            if self.param_types[name] != "feature":
                continue
            if not torch.is_tensor(tensor) or not torch.is_floating_point(tensor):
                continue
            matched_prefix = self._matched_feature_prefix(name)
            layer_name = matched_prefix if matched_prefix is not None else self._fallback_layer_name(name)
            self.param_to_layer_idx[name] = layer_to_idx[layer_name]
            self.feature_param_names.append(name)

    def get_feature_layer_index(self, param_name: str) -> int:
        if param_name not in self.param_to_layer_idx:
            raise KeyError(f"{param_name} is not a floating H2A feature parameter.")
        return self.param_to_layer_idx[param_name]

    def select_reference_clients(self, target_client_id: int) -> List[int]:
        self._validate_client_id(target_client_id)
        reference_mode = getattr(self.cfg.federated, "h2a_reference_mode", "adaptive").lower()
        if reference_mode == "fixed":
            return self._select_fixed_reference_clients(target_client_id)
        if reference_mode != "adaptive":
            raise ValueError("h2a_reference_mode must be 'adaptive' or 'fixed'.")
        return self._select_adaptive_reference_clients(target_client_id)

    def _select_fixed_reference_clients(self, target_client_id: int) -> List[int]:
        fixed_refs = getattr(self.cfg.federated, "h2a_fixed_ref_client_ids", {})
        configured_refs = fixed_refs.get(target_client_id)
        if configured_refs is None:
            configured_refs = fixed_refs.get(str(target_client_id))

        if configured_refs is None:
            missing_policy = getattr(self.cfg.federated, "h2a_fixed_missing_policy", "adaptive").lower()
            if missing_policy == "adaptive":
                return self._select_adaptive_reference_clients(target_client_id)
            if missing_policy == "error":
                raise ValueError(
                    f"h2a_reference_mode='fixed' but client {target_client_id} "
                    "has no h2a_fixed_ref_client_ids entry."
                )
            raise ValueError("h2a_fixed_missing_policy must be 'adaptive' or 'error'.")

        refs = [target_client_id]
        for ref_id in configured_refs:
            ref_id = int(ref_id)
            self._validate_client_id(ref_id)
            if ref_id == target_client_id or ref_id in refs:
                continue
            refs.append(ref_id)
        return refs

    def _select_adaptive_reference_clients(self, target_client_id: int) -> List[int]:
        if getattr(self.cfg.federated, "h2a_warmup_use_all_clients", False) and self.round <= 1:
            return [target_client_id] + [
                cid for cid in range(1, self.num_clients + 1) if cid != target_client_id
            ]

        num_refs = min(
            max(1, int(getattr(self.cfg.federated, "h2a_num_refs", 1))),
            self.num_clients,
        )
        row = self.importance_matrix[target_client_id - 1].detach().cpu()
        ranked = sorted(
            range(1, self.num_clients + 1),
            key=lambda cid: (-float(row[cid - 1].item()), cid),
        )
        refs = [target_client_id]
        for cid in ranked:
            if cid == target_client_id:
                continue
            refs.append(cid)
            if len(refs) >= num_refs:
                break
        return refs

    def compute_model_distance(self, state_a, state_b) -> float:
        distance = 0.0
        for name in self.feature_param_names:
            tensor_a = state_a[name]
            tensor_b = state_b[name]
            if not (torch.is_tensor(tensor_a) and torch.is_tensor(tensor_b)):
                continue
            diff = tensor_a.detach().cpu().float() - tensor_b.detach().cpu().float()
            distance += float(torch.sum(diff * diff).item())
        return distance

    def compute_alpha(self, target_client_id: int, ref_client_ids: List[int], state_snapshot) -> Tuple[float, float]:
        target_state = state_snapshot[target_client_id]
        distances = [
            self.compute_model_distance(target_state, state_snapshot[ref_id])
            for ref_id in ref_client_ids
        ]
        distance_avg = sum(distances) / max(len(distances), 1)
        gamma = float(getattr(self.cfg.federated, "h2a_gamma", 0.5))
        alpha = torch.sigmoid(torch.tensor(gamma * distance_avg, dtype=torch.float32)).item()
        return float(alpha), float(distance_avg)

    def build_personalized_state(self, target_client_id: int, training: bool = True):
        self._validate_client_id(target_client_id)
        state_snapshot = self.round_snapshot if training and self.round_snapshot is not None else self.client_states
        ref_client_ids = self.select_reference_clients(target_client_id)
        if target_client_id not in ref_client_ids:
            ref_client_ids = [target_client_id] + [cid for cid in ref_client_ids if cid != target_client_id]

        alpha, distance_avg = self.compute_alpha(target_client_id, ref_client_ids, state_snapshot)
        context = nullcontext() if training else torch.no_grad()

        with context:
            logits_p, logits_r, _ = self.hypernet(target_client_id)
            ref_indices = torch.tensor(
                [cid - 1 for cid in ref_client_ids],
                dtype=torch.long,
                device=self.device,
            )
            logits_p_ref = logits_p.index_select(0, ref_indices)
            logits_r_ref = logits_r.index_select(0, ref_indices)
            weights_p = torch.softmax(logits_p_ref, dim=0)
            weights_r = torch.softmax(logits_r_ref, dim=0)
            alpha_tensor = torch.tensor(alpha, dtype=weights_p.dtype, device=self.device)
            weights = (1.0 - alpha_tensor) * weights_p + alpha_tensor * weights_r

            graph_state = OrderedDict()
            detached_state = OrderedDict()
            target_state = state_snapshot[target_client_id]

            for name, target_tensor in target_state.items():
                if not torch.is_tensor(target_tensor):
                    graph_state[name] = copy.deepcopy(target_tensor)
                    detached_state[name] = copy.deepcopy(target_tensor)
                    continue

                if not torch.is_floating_point(target_tensor):
                    cloned = target_tensor.detach().cpu().clone()
                    graph_state[name] = cloned.clone()
                    detached_state[name] = cloned
                    continue

                param_type = self.param_types[name]
                if param_type == "feature":
                    layer_idx = self.get_feature_layer_index(name)
                    aggregated = None
                    for ref_pos, ref_id in enumerate(ref_client_ids):
                        ref_tensor = state_snapshot[ref_id][name].to(
                            device=self.device,
                            dtype=target_tensor.dtype,
                        )
                        term = ref_tensor * weights[ref_pos, layer_idx]
                        aggregated = term if aggregated is None else aggregated + term
                    graph_state[name] = aggregated
                    detached_state[name] = aggregated.detach().cpu().clone()
                elif param_type == "head":
                    local_head = target_tensor.detach().to(self.device).clone()
                    graph_state[name] = local_head
                    detached_state[name] = target_tensor.detach().cpu().clone()
                else:
                    raise ValueError(f"Unsupported H2A parameter type for {name}: {param_type}")

        weights_cpu = weights.detach().cpu().clone()
        self_pos = ref_client_ids.index(target_client_id)
        info = {
            "round": self.round,
            "client_id": target_client_id,
            "ref_client_ids": list(ref_client_ids),
            "alpha": alpha,
            "distance_avg": distance_avg,
            "gamma": float(getattr(self.cfg.federated, "h2a_gamma", 0.5)),
            "weights": weights_cpu,
            "self_weights": weights_cpu[self_pos].clone(),
            "feature_layer_names": list(self.feature_layer_names),
        }
        return graph_state, detached_state, info

    def update_importance(self, target_client_id: int, ref_client_ids: List[int], weights):
        self._validate_client_id(target_client_id)
        if not torch.is_tensor(weights):
            weights = torch.tensor(weights, dtype=torch.float32)
        weights = weights.detach().cpu().float()
        self_pos = ref_client_ids.index(target_client_id)
        self_weights = weights[self_pos]
        row_idx = target_client_id - 1

        for ref_pos, ref_id in enumerate(ref_client_ids):
            self._validate_client_id(ref_id)
            col_idx = ref_id - 1
            delta = torch.mean(weights[ref_pos] - self_weights).item()
            self.importance_matrix[row_idx, col_idx] += float(delta)

    def meta_update(self, target_client_id: int, graph_state, delta_state):
        self._validate_client_id(target_client_id)
        meta_loss = None
        for name in self.feature_param_names:
            if name not in graph_state or name not in delta_state:
                continue
            graph_tensor = graph_state[name]
            if not torch.is_tensor(graph_tensor) or not torch.is_floating_point(graph_tensor):
                continue
            delta_tensor = delta_state[name].detach().to(
                device=graph_tensor.device,
                dtype=graph_tensor.dtype,
            )
            term = torch.sum(graph_tensor * delta_tensor)
            meta_loss = term if meta_loss is None else meta_loss + term

        if meta_loss is None:
            return 0.0

        self.hyper_optimizer.zero_grad()
        meta_loss.backward()
        self.hyper_optimizer.step()
        return float(meta_loss.detach().cpu().item())

    def begin_round(self):
        self.round += 1
        self.round_snapshot = {
            cid: self.clone_state_dict(state)
            for cid, state in self.client_states.items()
        }
        self.pending_client_states = {}

    def queue_client_state(self, target_client_id: int, state_dict):
        self._validate_client_id(target_client_id)
        self.pending_client_states[target_client_id] = self.clone_state_dict(state_dict)

    def commit_round(self):
        for client_id, state in self.pending_client_states.items():
            self.client_states[client_id] = self.clone_state_dict(state)
        self.pending_client_states = {}
        self.round_snapshot = None

    def get_eval_state(self, target_client_id: int):
        _, detached_state, _ = self.build_personalized_state(target_client_id, training=False)
        return detached_state

    def state_dict(self):
        return {
            "round": self.round,
            "client_states": {
                cid: self.clone_state_dict(state)
                for cid, state in self.client_states.items()
            },
            "importance_matrix": self.importance_matrix.detach().cpu().clone(),
            "hypernet_state_dict": {
                name: tensor.detach().cpu().clone()
                for name, tensor in self.hypernet.state_dict().items()
            },
            "hyper_optimizer_state_dict": copy.deepcopy(self.hyper_optimizer.state_dict()),
            "feature_layer_names": list(self.feature_layer_names),
            "feature_param_names": list(self.feature_param_names),
            "param_types": copy.deepcopy(self.param_types),
            "param_to_layer_idx": copy.deepcopy(self.param_to_layer_idx),
        }

    def load_state_dict(self, payload):
        self.round = int(payload.get("round", 0))
        self.client_states = {
            int(cid): self.clone_state_dict(state)
            for cid, state in payload["client_states"].items()
        }
        self.importance_matrix = payload["importance_matrix"].detach().cpu().float().clone()
        self.hypernet.load_state_dict(payload["hypernet_state_dict"])
        if "hyper_optimizer_state_dict" in payload and payload["hyper_optimizer_state_dict"] is not None:
            self.hyper_optimizer.load_state_dict(payload["hyper_optimizer_state_dict"])
            self._move_optimizer_state_to_device()
        self.pending_client_states = {}
        self.round_snapshot = None

    def _move_optimizer_state_to_device(self):
        for state in self.hyper_optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(self.device)

    def _validate_client_id(self, client_id: int):
        if client_id < 1 or client_id > self.num_clients:
            raise ValueError(f"client_id must be in [1, {self.num_clients}], got {client_id}.")
