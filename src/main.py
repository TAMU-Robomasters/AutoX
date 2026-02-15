"""Main thing for real."""


#! check if this has multi threading
from sympy.core.evalf import LG10
from sympy.core.tests.test_priority import l
from src.toolbox.autoboot_check import throw_if_autoboot_is_already_running
from src.toolbox.globals import config




def main():
    """Main function to run the armor detection and processing loop."""
    if config.mode != "production":
        throw_if_autoboot_is_already_running()
    from src.processes.autoaim import AutoAimProcess
    auto_aim: AutoAimProcess = AutoAimProcess()

    from src.processes.count import CountProcess
    count_process: CountProcess = CountProcess()

    count_process.start()
    auto_aim.start()

    import iceoryx2 as iox2
    from src.ipc.log import LogMessage
    iox2.set_log_level_from_env_or(iox2.LogLevel.Info)
    log_node = iox2.NodeBuilder.new().create(iox2.ServiceType.Ipc)

    service = (
        log_node.service_builder(iox2.ServiceName.new("LogService"))
        .publish_subscribe(LogMessage)
        .open_or_create()
    )

    logger_subscriber = service.subscriber_builder().create()

    while True:
        while True:
            sample = logger_subscriber.receive()
            if sample is not None:
                data = sample.payload()
                print("received log message:", data.contents)
            else:
                break

