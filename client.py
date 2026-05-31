import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from models.cnn_lstm import CNNLSTMModel
from utils.data_utils import drop_duplicate_timestamps, prepare_client_data, inverse_transform_array
from utils.metrics import calc_metrics


def get_loss_fn(loss_name: str):
    loss_name = loss_name.lower()
    if loss_name == "mse":
        return nn.MSELoss()
    if loss_name == "mae":
        return nn.L1Loss()
    raise ValueError(f"不支持的损失函数: {loss_name}")


def get_optimizer(optimizer_name: str, model: nn.Module, lr: float):
    optimizer_name = optimizer_name.lower()
    if optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    raise ValueError(f"不支持的优化器: {optimizer_name}")


def run_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    train: bool,
    global_rc=None,
    rc_lambda: float = 0.0,
    return_details: bool = False,
):
    if train:
        model.train()
    else:
        model.eval()

    total_loss_sum = 0.0
    task_loss_sum = 0.0
    rc_loss_sum = 0.0
    count = 0
    preds_all = []
    trues_all = []
    use_rc = train and global_rc is not None and rc_lambda > 0

    with torch.enable_grad() if train else torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            if train:
                optimizer.zero_grad()

            if use_rc:
                pred, feat = model(batch_x, return_feature=True)
                task_loss = criterion(pred, batch_y)
                batch_rc = feat.mean(dim=0)
                rc_target = global_rc.to(device)
                rc_loss = torch.mean((batch_rc - rc_target) ** 2)
                loss = task_loss + rc_lambda * rc_loss
            else:
                pred = model(batch_x)
                task_loss = criterion(pred, batch_y)
                rc_loss = torch.zeros((), device=device)
                loss = task_loss

            if train:
                loss.backward()
                optimizer.step()

            total_loss_sum += loss.item() * batch_x.size(0)
            task_loss_sum += task_loss.item() * batch_x.size(0)
            rc_loss_sum += rc_loss.item() * batch_x.size(0)
            count += batch_x.size(0)

            preds_all.append(pred.detach().cpu().numpy())
            trues_all.append(batch_y.detach().cpu().numpy())

    preds_all = np.concatenate(preds_all, axis=0)
    trues_all = np.concatenate(trues_all, axis=0)
    avg_total_loss = total_loss_sum / max(count, 1)
    avg_task_loss = task_loss_sum / max(count, 1)
    avg_rc_loss = rc_loss_sum / max(count, 1)
    avg_weighted_rc_loss = rc_lambda * avg_rc_loss if use_rc else 0.0
    avg_rc_to_task_ratio = avg_weighted_rc_loss / (avg_task_loss + 1e-12) if use_rc else 0.0

    if return_details:
        details = {
            "task_loss": avg_task_loss,
            "rc_loss": avg_rc_loss,
            "weighted_rc_loss": avg_weighted_rc_loss,
            "total_loss": avg_total_loss,
            "rc_to_task_ratio": avg_rc_to_task_ratio,
        }
        return avg_total_loss, preds_all, trues_all, details

    return avg_total_loss, preds_all, trues_all


def compute_local_rc(model, loader, device):
    model.eval()
    sum_feat = None
    count = 0

    with torch.no_grad():
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)
            _, feat = model(batch_x, return_feature=True)

            batch_sum = feat.sum(dim=0).detach().cpu()
            if sum_feat is None:
                sum_feat = batch_sum
            else:
                sum_feat += batch_sum
            count += feat.size(0)

    if count == 0:
        raise RuntimeError("Cannot compute local RC from an empty loader.")

    return (sum_feat / count).float()


def is_head_param(name: str, head_prefixes, head_exact_names=None) -> bool:
    head_exact_names = head_exact_names or []
    if name in head_exact_names:
        return True
    return any(name.startswith(prefix + ".") for prefix in head_prefixes)


def get_head_param_names(state_dict, head_prefixes, head_exact_names=None):
    return [
        name for name in state_dict.keys()
        if is_head_param(name, head_prefixes, head_exact_names)
    ]


def clone_state_to_cpu(state_dict):
    return {k: v.detach().cpu().clone() for k, v in state_dict.items()}


def build_personalized_state(global_state_dict, local_head_state, head_mask, head_prefixes, head_exact_names=None):
    state = copy.deepcopy(global_state_dict)
    if local_head_state is None or head_mask is None:
        return state

    for name in get_head_param_names(global_state_dict, head_prefixes, head_exact_names):
        if name not in local_head_state or name not in head_mask:
            continue
        global_tensor = global_state_dict[name]
        local_tensor = local_head_state[name].to(device=global_tensor.device, dtype=global_tensor.dtype)
        mask = head_mask[name].to(device=global_tensor.device, dtype=global_tensor.dtype)
        state[name] = local_tensor * mask + global_tensor * (1.0 - mask)

    return state


def build_top_tau_mask(importance_dict, tau):
    if len(importance_dict) == 0:
        return {}

    mask_dict = {
        name: torch.zeros_like(value.detach().cpu(), dtype=torch.float32)
        for name, value in importance_dict.items()
    }
    total_numel = sum(value.numel() for value in importance_dict.values())

    if tau <= 0 or total_numel == 0:
        return mask_dict
    if tau >= 1:
        return {
            name: torch.ones_like(value.detach().cpu(), dtype=torch.float32)
            for name, value in importance_dict.items()
        }

    flat_importance = torch.cat([
        value.detach().cpu().float().reshape(-1)
        for value in importance_dict.values()
    ])
    k = max(1, int(round(float(tau) * total_numel)))
    top_indices = torch.topk(flat_importance, k=k, largest=True).indices
    flat_mask = torch.zeros(total_numel, dtype=torch.float32)
    flat_mask[top_indices] = 1.0

    offset = 0
    for name, value in importance_dict.items():
        numel = value.numel()
        mask_dict[name] = flat_mask[offset:offset + numel].reshape(value.shape).clone()
        offset += numel

    return mask_dict


class FederatedClient:
    def __init__(self, client_id: int, data_path: str, cfg):
        self.client_id = client_id
        self.client_name = f"client_{client_id}"
        self.data_path = data_path
        self.cfg = copy.deepcopy(cfg)
        self.device = torch.device(self.cfg.train.device)
        self.data = prepare_client_data(data_path, self.cfg)
        self.feature_cols = self.data["feature_cols"]
        self.local_head_state = None
        self.head_mask = None
        self.head_importance_ema = None

    def build_model(self):
        model = CNNLSTMModel(
            input_dim=len(self.feature_cols),
            output_dim=self.cfg.data.horizon,
            cfg=self.cfg.model,
        ).to(self.device)
        return model

    def get_head_prefixes(self):
        return getattr(self.cfg.federated, "head_param_prefixes", ["fc1", "fc2"])

    def get_head_exact_names(self):
        return getattr(self.cfg.federated, "head_param_exact_names", [])

    def has_personalization(self):
        return self.local_head_state is not None and self.head_mask is not None

    def make_personalized_state(self, global_state_dict):
        use_head_personalization = getattr(self.cfg.federated, "use_head_personalization", False)
        if use_head_personalization and self.has_personalization():
            return build_personalized_state(
                global_state_dict,
                self.local_head_state,
                self.head_mask,
                self.get_head_prefixes(),
                self.get_head_exact_names(),
            )
        return copy.deepcopy(global_state_dict)

    def export_personalization(self):
        return {
            "local_head_state": copy.deepcopy(self.local_head_state),
            "head_mask": copy.deepcopy(self.head_mask),
        }

    def import_personalization(self, payload):
        if payload is None:
            self.local_head_state = None
            self.head_mask = None
            return
        self.local_head_state = copy.deepcopy(payload.get("local_head_state"))
        self.head_mask = copy.deepcopy(payload.get("head_mask"))

    def local_update(
        self,
        global_state_dict,
        local_epochs: int,
        global_rc=None,
        rc_lambda: float = 0.0,
        enable_head_personalization=None,
        update_head_mask: bool = True,
    ):
        model = self.build_model()
        use_head_personalization_cfg = getattr(self.cfg.federated, "use_head_personalization", False)
        if enable_head_personalization is None:
            use_head_personalization = bool(use_head_personalization_cfg)
        else:
            use_head_personalization = bool(use_head_personalization_cfg and enable_head_personalization)
        tau = getattr(self.cfg.federated, "head_personalization_tau", 0.0)
        head_prefixes = self.get_head_prefixes()
        head_exact_names = self.get_head_exact_names()

        if use_head_personalization:
            init_state = self.make_personalized_state(global_state_dict)
        else:
            init_state = copy.deepcopy(global_state_dict)
        init_state_cpu = clone_state_to_cpu(init_state)
        model.load_state_dict(init_state)

        head_init = {}
        for name, tensor in model.state_dict().items():
            if is_head_param(name, head_prefixes, head_exact_names):
                head_init[name] = tensor.detach().cpu().clone()

        criterion = get_loss_fn(self.cfg.train.loss_name)
        optimizer = get_optimizer(self.cfg.train.optimizer_name, model, self.cfg.train.lr)

        train_losses = []
        train_details = []
        val_losses = []

        for _ in range(local_epochs):
            train_loss, _, _, details = run_one_epoch(
                model=model,
                loader=self.data["train_loader"],
                criterion=criterion,
                optimizer=optimizer,
                device=self.device,
                train=True,
                global_rc=global_rc,
                rc_lambda=rc_lambda,
                return_details=True,
            )
            val_loss, _, _ = run_one_epoch(
                model=model,
                loader=self.data["val_loader"],
                criterion=criterion,
                optimizer=None,
                device=self.device,
                train=False,
            )
            train_losses.append(train_loss)
            train_details.append(details)
            val_losses.append(val_loss)

        last_train_details = train_details[-1]
        train_task_loss = float(last_train_details["task_loss"])
        final_state = clone_state_to_cpu(model.state_dict())
        delta_state = {}
        for name, tensor in final_state.items():
            if (
                torch.is_tensor(tensor)
                and torch.is_floating_point(tensor)
                and name in init_state_cpu
                and torch.is_tensor(init_state_cpu[name])
            ):
                delta_state[name] = tensor.detach().cpu() - init_state_cpu[name].detach().cpu()
        local_rc = compute_local_rc(model, self.data["train_loader"], self.device)
        head_param_names = get_head_param_names(final_state, head_prefixes, head_exact_names)
        num_head_params = int(sum(final_state[name].numel() for name in head_param_names))
        num_personalized_head_params = 0
        personalized_head_ratio = 0.0
        head_importance_mean = 0.0
        importance = None
        raw_importance = None
        use_ema = bool(getattr(self.cfg.federated, "use_head_importance_ema", False))
        ema_beta = float(getattr(self.cfg.federated, "head_importance_ema_beta", 0.8))

        if use_head_personalization:
            raw_importance = {}
            for name in head_param_names:
                delta = delta_state[name]
                raw_importance[name] = ((delta * final_state[name]) ** 2).detach().cpu().float()

            if use_ema:
                if self.head_importance_ema is None:
                    self.head_importance_ema = {
                        name: raw_importance[name].detach().cpu().float().clone()
                        for name in raw_importance
                    }
                else:
                    new_ema = {}
                    for name in raw_importance:
                        current = raw_importance[name].detach().cpu().float()
                        previous = self.head_importance_ema.get(name)
                        if previous is None or previous.shape != current.shape:
                            new_ema[name] = current.clone()
                        else:
                            new_ema[name] = ema_beta * previous.float() + (1.0 - ema_beta) * current
                    self.head_importance_ema = new_ema

                importance = {
                    name: value.detach().cpu().float().clone()
                    for name, value in self.head_importance_ema.items()
                }
            else:
                importance = {
                    name: value.detach().cpu().float().clone()
                    for name, value in raw_importance.items()
                }

            if len(importance) > 0:
                flat_importance = torch.cat([value.reshape(-1).float() for value in importance.values()])
                head_importance_mean = float(flat_importance.mean().item())

            if update_head_mask or self.head_mask is None:
                self.head_mask = build_top_tau_mask(importance, tau)
            self.local_head_state = {
                name: final_state[name].clone()
                for name in head_param_names
            }
            current_head_mask = self.head_mask if self.head_mask is not None else {}
            num_personalized_head_params = int(sum(mask.sum().item() for mask in current_head_mask.values()))
            personalized_head_ratio = (
                num_personalized_head_params / num_head_params
                if num_head_params > 0 else 0.0
            )
        else:
            self.head_mask = None
            self.local_head_state = None

        return {
            "state_dict": final_state,
            "delta_state": delta_state,
            "num_samples": self.data["train_samples"],
            "local_rc": local_rc,
            "head_mask": copy.deepcopy(self.head_mask),
            "head_importance": copy.deepcopy(importance),
            "head_raw_importance": copy.deepcopy(raw_importance),
            "head_param_names": copy.deepcopy(head_param_names),
            "head_prefixes": copy.deepcopy(head_prefixes),
            "head_exact_names": copy.deepcopy(head_exact_names),
            "use_head_personalization": use_head_personalization,
            "use_head_personalization_effective": use_head_personalization,
            "head_mask_updated": bool(update_head_mask and use_head_personalization),
            "use_head_importance_ema": use_ema if use_head_personalization else False,
            "head_importance_ema_beta": ema_beta if (use_head_personalization and use_ema) else None,
            "num_head_params": num_head_params,
            "num_personalized_head_params": num_personalized_head_params,
            "personalized_head_ratio": personalized_head_ratio,
            "head_importance_mean": head_importance_mean,
            "train_loss": train_task_loss,
            "train_task_loss": train_task_loss,
            "train_rc_loss": float(last_train_details["rc_loss"]),
            "train_weighted_rc_loss": float(last_train_details["weighted_rc_loss"]),
            "train_total_loss": float(last_train_details["total_loss"]),
            "train_rc_to_task_ratio": float(last_train_details["rc_to_task_ratio"]),
            "val_loss": float(val_losses[-1]),
        }

    def predict_split(self, global_state_dict, split_name: str = "test", use_personalized_head=None):
        if split_name not in ["train", "val", "test"]:
            raise ValueError("split_name 只支持 train / val / test")

        model = self.build_model()
        if use_personalized_head is None:
            use_personalized_head = getattr(self.cfg.federated, "use_head_personalization", False)
        if use_personalized_head:
            state = self.make_personalized_state(global_state_dict)
        else:
            state = copy.deepcopy(global_state_dict)
        model.load_state_dict(state)

        criterion = get_loss_fn(self.cfg.train.loss_name)
        loader = self.data[f"{split_name}_loader"]
        ts_matrix = self.data[f"{split_name}_timestamps"]

        loss, pred_scaled, true_scaled = run_one_epoch(
            model=model,
            loader=loader,
            criterion=criterion,
            optimizer=None,
            device=self.device,
            train=False,
        )

        y_scaler = self.data["y_scaler"]
        pred_real = inverse_transform_array(y_scaler, pred_scaled)
        true_real = inverse_transform_array(y_scaler, true_scaled)

        result = pd.DataFrame({
            "timestamp": pd.to_datetime(ts_matrix[:, 0]),
        })

        for step in range(self.cfg.data.horizon):
            result[f"y_true_step_{step + 1}"] = true_real[:, step]
            result[f"y_pred_step_{step + 1}"] = pred_real[:, step]

        result = result.sort_values("timestamp").reset_index(drop=True)
        if split_name == "test":
            result = drop_duplicate_timestamps(result, timestamp_col="timestamp", keep="first")
        return result, float(loss)

    def evaluate_split(self, global_state_dict, split_name: str = "test", use_personalized_head=None):
        pred_df, loss = self.predict_split(
            global_state_dict,
            split_name=split_name,
            use_personalized_head=use_personalized_head,
        )

        true_cols = [f"y_true_step_{i + 1}" for i in range(self.cfg.data.horizon)]
        pred_cols = [f"y_pred_step_{i + 1}" for i in range(self.cfg.data.horizon)]

        metrics = calc_metrics(
            pred_df[true_cols].values.reshape(-1),
            pred_df[pred_cols].values.reshape(-1),
        )

        return pred_df, metrics, float(loss)
