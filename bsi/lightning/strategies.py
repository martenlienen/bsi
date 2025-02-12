import logging

from lightning.pytorch.accelerators import CUDAAccelerator
from lightning.pytorch.strategies import DDPStrategy

log = logging.getLogger(__name__)


class NoWrappingDDPStrategy(DDPStrategy):
    """A DDP strategy that does not wrap the model in DistributedDataParallel.

    This lets us deal with DDP ourselves and combine it correctly with EMA and
    torch.compile.
    """

    @staticmethod
    def applicable():
        return CUDAAccelerator.is_available() and CUDAAccelerator.auto_device_count() > 1

    def configure_ddp(self):
        hook = getattr(self.model, "configure_ddp", None)
        if hook is None:
            name = self.model.__class__.name
            log.warning(
                f"Using NoWrappingDDPStrategy but {name} is not DDP aware. "
                "Falling back to normal DDP!"
            )
            return super().configure_ddp()
        else:
            log.info("Running model configure_ddp hook")
            hook()
