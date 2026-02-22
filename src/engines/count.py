"""A simple engine that counts and sends log messages until the engine is stopped."""

import iceoryx2 as iox2

from src.core.engine import Engine
from src.core.process import Process
from src.types.ipc import LogMessage
from src.types.null import NullContext


class CountProcess(Process):  # noqa: D101
    def initialize(self):  # noqa: D102
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

    def execute(self):  # noqa: D102
        self.node.wait(iox2.Duration.from_millis(500))
        self.sample = self.publisher.loan_uninit().write_payload(
            LogMessage(
                log_level=b"I",
                process_name=b"CountProcess",
                message=f"Count: {self.count}".encode("utf-8"),
            )
        )
        self.sample.send()
        self.count += 1


class CountEngine(Engine[NullContext]):
    """Count and send log messages until the engine is stopped."""

    def __init__(self):
        super().__init__(
            modules=[
                # No modules for this simple engine
            ],
            context_type=NullContext
        )
        self.count_process = CountProcess()

    def start(self):  # noqa: D102
        self.count_process.start()
