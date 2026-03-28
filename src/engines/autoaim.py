"""Auto-aim engine process."""

# import iceoryx2 as iox2

from src.core.engine import Engine
from src.subsystems.display import display
from src.subsystems.selection import SelectingWith3DModule
from src.subsystems.vision import ClassicalDetectorModule
from src.types.autoaim import AutoAimContext
from src.types.ipc import ImageMessage, LogMessage
from src.subsystems.estimation.module import RadiiEstimatorModule

"""
# TODO: make context_type just accept the concept
class SimpleAutoAimEngine(Engine[AutoAimContext]):
   # Use classical cv to find panels and 3D info to select the best target.

    def __init__(self):
        self.ctx = AutoAimContext()
        self.detection = ClassicalDetectorModule(self.ctx)
        self.selection = SelectingWith3DModule(self.ctx)
        super().__init__(
            modules=[
                self.detection,
                self.selection,
            ],
            context_type=AutoAimContext,
        )

    def initialize(self):  # noqa: D102
        self.node = (
            iox2.NodeBuilder.new()
            .name(iox2.NodeName.new("AutoAimProcess"))
            .create(iox2.ServiceType.Ipc)
        )

        self.service = (
            self.node.service_builder(iox2.ServiceName.new("LogService"))
            .publish_subscribe(LogMessage)
            .open_or_create()
        )
        self.publisher = self.service.publisher_builder().create()
        self.count = 0

    def execute(self):  # noqa: D102
        # FIXME: this is creating an new instance every time. not good.
        self.detection.run(self.ctx)
        self.selection.run(self.ctx)

        # TODO: add config control for display
        self.sample = self.publisher.loan_uninit().write_payload(
            LogMessage(
                log_level=b"I",
                process_name=b"AutoAimProcess",
                message=f"Your mom is this big: {self.count}".encode("utf-8"),
            )
        )
        self.sample.send()
        self.count += 1

        display.show_windows()

"""

# NOTE: Assume that detections/panels are all coming from the same robot
class AdvancedAutoAimEngine(Engine[AutoAimContext]):
    """Use more advanced techniques to find panels and select targets."""
    def __init__(self):
        self.ctx = AutoAimContext()
        self.detection = ClassicalDetectorModule(self.ctx)
        self.estimation = RadiiEstimatorModule(self.ctx)
        super().__init__(
            modules=[
                self.detection,
                self.estimation,
            ],
            context_type=AutoAimContext,
        )

    def initialize(self):  # noqa: D102
        pass

    def execute(self):  # noqa: D102
        self.detection.run(self.ctx)
        self.estimation.run(self.ctx)
        display.show_windows()
