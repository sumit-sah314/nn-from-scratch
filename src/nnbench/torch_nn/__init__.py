"""PyTorch reference implementations (same architectures, built-in autograd/optimizers)."""
from .models import build_model, make_optimizer

__all__ = ["build_model", "make_optimizer"]
