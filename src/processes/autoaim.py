"""TODO."""
from src.pipelines.aim.classic_vision import classic_aim_pipeline
from src.subsystems.display import display
from src.pipelines.aim.context import AutoAimContext
from src.core.process import Process
import iceoryx2 as iox2
from src.ipc.log import LogMessage

context = AutoAimContext()

class AutoAimProcess(Process):
    def __init__(self):
        super().__init__()
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

    def execute(self):
        classic_aim_pipeline.run(context)
        #TODO: add config control for display
        self.sample = self.publisher.loan_uninit().write_payload(
            LogMessage(
                log_level=b'I',
                process_name=b'AutoAimProcess',
                message=f'Your mom is this big: {self.count}'.encode('utf-8')
            )
        )
        self.sample.send()
        self.count += 1

        display.show_windows()