"""Deterministic, training-only sequence variance models for Phase 4.

Both models estimate zero-mean Gaussian conditional variance. The state-space
model is a small stable diagonal linear recurrence, not a Mamba implementation.
Input windows must already obey the experiment's forecast-origin timing rules.
"""

import math
from collections.abc import Mapping

import numpy as np
import torch
from torch import nn


_LOG_VARIANCE_BOUND = 12.0
_SCHEMA_VERSION = 1


class _GRUVariance(nn.Module):
    def __init__(self, n_inputs, hidden_size):
        super().__init__()
        self.recurrent = nn.GRU(n_inputs, hidden_size, batch_first=True,
                                dtype=torch.float64, device="cpu")
        self.head = nn.Linear(hidden_size, 1, dtype=torch.float64, device="cpu")

    def forward(self, values):
        _, state = self.recurrent(values)
        raw = self.head(state[-1]).squeeze(-1)
        return _LOG_VARIANCE_BOUND * torch.tanh(raw / _LOG_VARIANCE_BOUND)


class _DiagonalStateSpace(nn.Module):
    def __init__(self, n_inputs, hidden_size):
        super().__init__()
        self.log_decay = nn.Parameter(torch.linspace(-2.0, 4.0, hidden_size,
                                                     dtype=torch.float64, device="cpu"))
        self.input_projection = nn.Linear(n_inputs, hidden_size, dtype=torch.float64,
                                          device="cpu")
        self.head = nn.Linear(hidden_size, 1, dtype=torch.float64, device="cpu")

    def decay(self):
        # Keep strict stability even if sigmoid rounds an extreme logit to 1.
        return torch.sigmoid(self.log_decay).clamp(max=1.0 - torch.finfo(torch.float64).eps)

    def forward(self, values):
        state = values.new_zeros((values.shape[0], self.log_decay.numel()))
        projected = self.input_projection(values)
        decay = self.decay()
        for step in range(values.shape[1]):
            state = decay * state + projected[:, step, :]
        raw = self.head(state).squeeze(-1)
        return _LOG_VARIANCE_BOUND * torch.tanh(raw / _LOG_VARIANCE_BOUND)


def _make_model(architecture, n_inputs, hidden_size):
    if architecture == "gru":
        return _GRUVariance(n_inputs, hidden_size)
    if architecture == "ssm":
        return _DiagonalStateSpace(n_inputs, hidden_size)
    raise ValueError("Architecture must be 'gru' or 'ssm'")


def _integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}")
    return int(value)


def _features(values):
    values = np.asarray(values, dtype=np.float64)
    if (values.ndim != 3 or values.shape[1] < 1 or values.shape[2] < 1
            or np.isinf(values).any()):
        raise ValueError("Features must have shape [observations, time, features] with finite values or NaN")
    return values


def _expanded(values, medians):
    return np.concatenate((np.where(np.isnan(values), medians, values),
                           np.isnan(values).astype(np.float64)), axis=-1)


def _preprocessor(values):
    flattened = values.reshape(-1, values.shape[-1])
    medians = np.array([np.median(column[np.isfinite(column)]) if np.isfinite(column).any() else 0.0
                        for column in flattened.T])
    expanded = _expanded(flattened, medians)
    with np.errstate(over="ignore", invalid="ignore"):
        centers = expanded.mean(axis=0)
        scales = expanded.std(axis=0)
    scales[scales <= np.finfo(np.float64).eps] = 1.0
    if not np.isfinite(centers).all() or not np.isfinite(scales).all():
        raise ValueError("Training features produce nonfinite preprocessing statistics")
    return medians, centers, scales


def _transform(values, medians, centers, scales):
    with np.errstate(over="ignore", invalid="ignore"):
        transformed = (_expanded(values, medians) - centers) / scales
    if not np.isfinite(transformed).all():
        raise ValueError("Feature transformation produced nonfinite values")
    return torch.from_numpy(np.ascontiguousarray(transformed))


def _deterministic_cpu():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def fit_sequence_variance(X, y, *, architecture, hidden_size, epochs=40,
                          learning_rate=0.01, seed=1729):
    """Fit a GRU or diagonal state-space variance model to complete windows.

    Medians, centers, and scales are fitted across training window rows only.
    One missing indicator is appended per input feature; an entirely missing
    feature has median zero. The target is divided by its training RMS. The
    full-batch Adam objective is ``.5 * mean(log(h) + (y/RMS)**2 / h)``.
    ``log(h) = 12*tanh(raw/12)`` smoothly bounds the normalized variance away
    from numerical zero and overflow. No validation data, shuffling, early
    stopping, checkpoint selection, or target mean fitting is performed.

    Returns JSON-safe metadata and the final-epoch CPU float64 state_dict.
    Loss history includes the initial loss and one post-update loss per epoch,
    measured in normalized target units (the Gaussian constant is omitted).
    """
    X = _features(X)
    y = np.asarray(y, dtype=np.float64)
    if y.ndim != 1 or len(y) != len(X) or len(y) < 2 or not np.isfinite(y).all():
        raise ValueError("Sequence training requires at least two aligned finite targets")
    hidden_size = _integer(hidden_size, "Hidden size")
    epochs = _integer(epochs, "Epochs")
    seed = _integer(seed, "Seed", minimum=0)
    if seed > 2**63 - 1:
        raise ValueError("Seed must be no greater than 2**63 - 1")
    if isinstance(learning_rate, (bool, np.bool_)) or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Learning rate must be finite and positive")
    with np.errstate(over="ignore", invalid="ignore"):
        second_moment = float(np.mean(y * y))
    if not math.isfinite(second_moment) or second_moment <= 0:
        raise ValueError("Training targets require a finite positive second moment")
    target_scale = math.sqrt(second_moment)
    medians, centers, scales = _preprocessor(X)
    values = _transform(X, medians, centers, scales)
    squares = torch.from_numpy(np.ascontiguousarray((y / target_scale) ** 2))
    _deterministic_cpu()
    # Preserve the caller's RNG state while fixing initialization locally.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = _make_model(architecture, X.shape[-1] * 2, hidden_size)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate),
                                     betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0,
                                     foreach=False)

        def objective():
            log_variance = model(values)
            return 0.5 * (log_variance + squares * torch.exp(-log_variance)).mean()

        with torch.no_grad():
            history = [float(objective())]
        if not math.isfinite(history[0]):
            raise RuntimeError("Sequence model initial loss is nonfinite")
        for _ in range(epochs):
            optimizer.zero_grad(set_to_none=True)
            loss = objective()
            loss.backward()
            if not all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                       for parameter in model.parameters()):
                raise RuntimeError("Sequence model produced a nonfinite or missing gradient")
            optimizer.step()
            with torch.no_grad():
                history.append(float(objective()))
            if not math.isfinite(history[-1]):
                raise RuntimeError("Sequence model training loss is nonfinite")
        checkpoint = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        if not all(torch.isfinite(value).all() for value in checkpoint.values()):
            raise RuntimeError("Sequence model checkpoint contains nonfinite parameters")

    metadata = {
        "schema_version": _SCHEMA_VERSION, "architecture": architecture,
        "hidden_size": hidden_size, "n_features": X.shape[-1], "sequence_length": X.shape[1],
        "training_observations": len(y), "preprocessing_rows": X.shape[0] * X.shape[1],
        "medians": medians.tolist(), "centers": centers.tolist(), "scales": scales.tolist(),
        "missing_indicators": True, "target_scale": target_scale,
        "log_variance_bound": _LOG_VARIANCE_BOUND, "mean": "fixed_zero",
        "seed": seed, "epochs": epochs, "learning_rate": float(learning_rate),
        "optimizer": "Adam", "optimizer_betas": [0.9, 0.999], "optimizer_epsilon": 1e-8,
        "weight_decay": 0.0, "batch_size": len(y), "optimizer_steps": epochs,
        "checkpoint_epoch": epochs, "checkpoint_selection": "final_epoch",
        "loss_initial": history[0], "loss_final": history[-1], "loss_history": history,
        "loss_units": "training_RMS_normalized_Gaussian_QNLL_without_constant",
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "dtype": "float64", "device": "cpu", "threads": 1,
        "deterministic_algorithms": True,
    }
    return metadata, checkpoint


def predict_sequence_variance(X, metadata, checkpoint):
    """Restore a final checkpoint and predict with frozen training transforms."""
    X = _features(X)
    if not isinstance(metadata, Mapping) or metadata.get("schema_version") != _SCHEMA_VERSION:
        raise ValueError("Unsupported sequence model metadata schema")
    try:
        n_features = _integer(metadata["n_features"], "Feature count")
        sequence_length = _integer(metadata["sequence_length"], "Sequence length")
        hidden_size = _integer(metadata["hidden_size"], "Hidden size")
        medians = np.asarray(metadata["medians"], dtype=np.float64)
        centers = np.asarray(metadata["centers"], dtype=np.float64)
        scales = np.asarray(metadata["scales"], dtype=np.float64)
        target_scale = float(metadata["target_scale"])
        architecture = metadata["architecture"]
    except (KeyError, TypeError, OverflowError) as exc:
        raise ValueError("Incomplete or invalid sequence model metadata") from exc
    if X.shape[1:] != (sequence_length, n_features):
        raise ValueError("Prediction sequence length or feature count differs from training")
    if (medians.shape != (n_features,) or centers.shape != (2 * n_features,)
            or scales.shape != (2 * n_features,) or not np.isfinite(medians).all()
            or not np.isfinite(centers).all() or not np.isfinite(scales).all() or np.any(scales <= 0)
            or not math.isfinite(target_scale) or target_scale <= 0
            or metadata.get("log_variance_bound") != _LOG_VARIANCE_BOUND
            or metadata.get("missing_indicators") is not True
            or metadata.get("mean") != "fixed_zero"):
        raise ValueError("Invalid sequence model preprocessing or variance metadata")
    _deterministic_cpu()
    with torch.random.fork_rng(devices=[]):
        model = _make_model(architecture, 2 * n_features, hidden_size)
    expected = model.state_dict()
    if not isinstance(checkpoint, Mapping) or set(checkpoint) != set(expected):
        raise ValueError("Checkpoint keys do not match the sequence model architecture")
    for name, value in checkpoint.items():
        if (not isinstance(value, torch.Tensor) or value.device.type != "cpu"
                or value.dtype != torch.float64 or value.shape != expected[name].shape
                or not torch.isfinite(value).all()):
            raise ValueError(f"Invalid checkpoint tensor: {name}")
    model.load_state_dict(checkpoint, strict=True)
    model.eval()
    values = _transform(X, medians, centers, scales)
    if not len(X):
        return np.empty(0, dtype=np.float64)
    with torch.no_grad():
        normalized = torch.exp(model(values)).numpy()
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        variance = normalized * np.square(np.float64(target_scale))
    if not np.isfinite(variance).all() or not (variance > 0).all():
        raise RuntimeError("Sequence prediction produced a nonpositive or nonfinite variance")
    return variance
