
from src.core.engine import Engine
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
        self.publisher = self.service.publisher_builder().create()
        self.count = 0
        self.ctx = AutoAimContext()

    def execute(self):  # noqa: D102
        ClassicalDetectorModule().run(self.ctx)

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




class Rerun (Engine):
    def __init__(self):
        super().__init__(
            modules=[
                ClassicalDetectorModule(),
            ],
            context_type=AutoAimContext
        )
        self.aim = AutoAimProcess()

    def start(self):  # noqa: D102
        self.aim.start()
