"""Auto-aim engine process."""

import iceoryx2 as iox2

from src.core.engine import Engine
from src.core.process import Process
from src.subsystems.display import display
from src.subsystems.selection import SelectingWith3DModule
from src.subsystems.vision import ClassicalDetectorModule
from src.types.autoaim import AutoAimContext
from src.types.ipc import ImageMessage, LogMessage


class AutoAimProcess(Process):  # noqa: D101
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
        self.image_service = (
            self.node.service_builder(iox2.ServiceName.new("ImageService"))
            .publish_subscribe(ImageMessage)
            .open_or_create()
        )
        self.image_publisher = self.image_service.publisher_builder().create()
        self.publisher = self.service.publisher_builder().create()
        self.count = 0
        self.ctx = AutoAimContext()

    def execute(self):  # noqa: D102
        ClassicalDetectorModule().run(self.ctx)
        SelectingWith3DModule().run(self.ctx)

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


class SimpleAutoAimEngine(Engine[AutoAimContext]):
    """Use classical cv to find panels and 3D info to select the best target."""

    def __init__(self):
        super().__init__(
            modules=[
                ClassicalDetectorModule(),
                SelectingWith3DModule(),
            ]
        )
        self.aim = AutoAimProcess()

    def start(self):  # noqa: D102
        self.aim.start()
