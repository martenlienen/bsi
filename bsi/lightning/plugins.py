from contextlib import contextmanager

from lightning.fabric.accelerators import CUDAAccelerator
from lightning.pytorch.plugins.precision import MixedPrecision


class TrainOnlyAMP(MixedPrecision):
    """AMP that is only active for the training step."""

    def __init__(self, precision):
        device = "cuda" if CUDAAccelerator.is_available() else "cpu"
        super().__init__(precision, device)

    @contextmanager
    def val_step_context(self):
        yield

    @contextmanager
    def test_step_context(self):
        yield

    @contextmanager
    def predict_step_context(self):
        yield
