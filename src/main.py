"""Main thing for real."""

#! check if this has multi threading
from src.toolbox.autoboot_check import throw_if_autoboot_is_already_running
from src.toolbox.globals import config


def main():
    """Main function to run the armor detection and processing loop."""
    if config.mode != "production":
        throw_if_autoboot_is_already_running()
    from src.engines.autoaim import SimpleAutoAimEngine

    auto_aim: SimpleAutoAimEngine = SimpleAutoAimEngine()

    from src.engines.count import CountEngine

    count_process: CountEngine = CountEngine()

    count_process.start()
    auto_aim.start()

    import iceoryx2 as iox2

    from src.types.ipc import LogMessage

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
