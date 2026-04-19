
import iceoryx2 as iox2

from src.core.engine import Engine
from src.subsystems.selection import SelectingWith3DModule
from src.subsystems.vision import ClassicalDetectorModule
from src.types.autoaim import AutoAimContext
from src.subsystems.visualization import Simulation

class RerunEngine(Engine[AutoAimContext]):
    """Use classical cv to find panels and 3D info to select the best target."""

    def __init__(self):
        self.ctx = AutoAimContext()
        self.detection = ClassicalDetectorModule(self.ctx)
        self.selection = SelectingWith3DModule(self.ctx)
        self.simulate = Simulation(self.ctx)
        super().__init__(
            modules=[
                self.detection,
                self.selection,

            ],
            context_type=AutoAimContext,
        )

#make process
    def initialize(self):  # noqa: D102
        #these are here for now not sure if I'm supposed to get these from somewhere else
        self.detection.run(self.ctx)
        self.selection.run(self.ctx)

        self.node = (
            iox2.NodeBuilder.new()
            .name(iox2.NodeName.new("RerunProcess"))
            .create(iox2.ServiceType.Ipc)
        )

        self.service = (
            self.node.service_builder(iox2.ServiceName.new("LogService"))
            .publish_subscribe(LogMessage)
            .open_or_create()
        )

        self.publisher = self.service.publisher_builder().create()
        self.count = 0

#run process
    def execute(self):  # noqa: D102
        #message thing
        self.node.wait(iox2.Duration.from_millis(500))
        self.sample = self.publisher.loan_uninit().write_payload(
            LogMessage(
                log_level=b"I",
                process_name=b"RerunProcess",
                message=f"there are messages being sent here cool: {self.count}".encode("utf-8"),
            )
        )
        self.sample.send()
        self.count += 1

    def stop(self):
        pass

 