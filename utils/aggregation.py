from collections import OrderedDict


def fedavg(state_dicts, sample_counts):
    if len(state_dicts) == 0:
        raise ValueError("state_dicts 为空，无法执行 FedAvg。")
    if len(state_dicts) != len(sample_counts):
        raise ValueError("state_dicts 与 sample_counts 长度不一致。")

    total_samples = float(sum(sample_counts))
    if total_samples <= 0:
        raise ValueError("样本数之和必须大于 0。")

    avg_state = OrderedDict()
    first_state = state_dicts[0]

    for key in first_state.keys():
        avg_state[key] = first_state[key].clone().detach() * 0.0

    for state, n in zip(state_dicts, sample_counts):
        weight = float(n) / total_samples
        for key in avg_state.keys():
            avg_state[key] += state[key].detach().clone() * weight

    return avg_state
