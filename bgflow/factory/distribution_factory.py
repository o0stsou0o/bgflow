
import torch
import bgflow as bg
from bgflow.bgflow.distribution.normal import NormalDistribution, TruncatedNormalDistribution
from bgflow.bgflow.distribution.distributions import UniformDistribution

__all__ = ["make_distribution"]

# === Prior Factory ===


def make_distribution(distribution_type, shape, **kwargs):
    factory = DISTRIBUTION_FACTORIES[distribution_type]
    return factory(shape=shape, **kwargs)


def _make_uniform_distribution(shape, device=None, dtype=None, **kwargs):
    defaults = {
        "low": torch.zeros(shape),
        "high": torch.ones(shape)
    }
    defaults.update(kwargs)
    for key in defaults:
        if isinstance(defaults[key], torch.Tensor):
            defaults[key] = defaults[key].to(device=device, dtype=dtype)
    return UniformDistribution(**defaults)


def _make_normal_distribution(shape, device=None, dtype=None, **kwargs):
    defaults = {
        "dim": shape,
        "mean": torch.zeros(shape),
    }
    defaults.update(kwargs)
    for key in defaults:
        if isinstance(defaults[key], torch.Tensor):
            defaults[key] = defaults[key].to(device=device, dtype=dtype)
    return NormalDistribution(**defaults)


def _make_truncated_normal_distribution(shape, device=None, dtype=None, **kwargs):
    defaults = {
        "mu": torch.zeros(shape),
        "sigma": torch.ones(shape),
    }
    defaults.update(kwargs)
    for key in defaults:
        if isinstance(defaults[key], torch.Tensor):
            defaults[key] = defaults[key].to(device=device, dtype=dtype)
    return TruncatedNormalDistribution(**defaults)


DISTRIBUTION_FACTORIES = {
    UniformDistribution: _make_uniform_distribution,
    NormalDistribution: _make_normal_distribution,
    TruncatedNormalDistribution: _make_truncated_normal_distribution
}

