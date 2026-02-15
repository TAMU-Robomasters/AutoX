
from src.core.process import Process
from src.ipc.log import LogMessage
import iceoryx2 as iox2


class CountProcess(Process):
    def __init__(self):
        super().__init__()
        self.node = (
            iox2.NodeBuilder.new()
            .name(iox2.NodeName.new("CountProcess"))
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
        self.node.wait(iox2.Duration.from_millis(500))
        self.sample = self.publisher.loan_uninit().write_payload(
            LogMessage(
                log_level=b'I',
                process_name=b'CountProcess',
                message=f'Count: {self.count}'.encode('utf-8')
            )
        )
        self.sample.send()
        self.count += 1 
