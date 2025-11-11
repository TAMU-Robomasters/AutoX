from ctypes import Structure, c_uint8, c_float, c_bool
import serial
from time import time

from super_map import LazyDict

from toolbox.globals import path_to, config, print, runtime
from subsystems.video_stream import video_stream

# 
# config
# 
serial_port  = config.communication.serial_port
baudrate     = config.communication.serial_baudrate

def setup_serial_port():
    print('')  # spacer
    if not serial_port:
        print('[Communication]: Port=None so no communication')
        return None  # disable port
    else:
        print(f'[Communication]: Port={serial_port}')
        try:
            return serial.Serial(
                serial_port,
                baudrate=baudrate,
                timeout=.05,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE
            )
        except Exception as error:
            import subprocess
            # very bad hack but it works
            # FIXME
            subprocess.run(["bash", "-c", f"sudo -S chmod 777 '{serial_port}' <<<  \"$(cat \"$HOME/.pass\")\" ", ])
            return setup_serial_port()  # recursion until it works


#
# initialize
#
port = setup_serial_port()


# C++ struct
class MessageToEmbedded(Structure):
    _pack_ = 1
    _fields_ = [
        ("magic_number", c_uint8),
        ("X", c_float),
        ("Y", c_float),
        ("Z", c_float),
        ("capture_delay", c_uint8),
        ("status", c_uint8),
    ]

message_to_embedded = MessageToEmbedded(ord('a'), 0.0, 0.0, 0.0, 0, 0)
# 
# main
#
def when_aiming_refreshes(panels, start_time):
    global port, last_x, last_y, last_z
    capture_delay = int((time()-start_time) * 1000)
    # capture_delay = min(int(monotonic()*1000 - capture_time), 255) # max 255 ms delay
    # TODO: figure out how to get a good capture delay value automatically

    # Sending XYZ position (meters), time since frame capture, and status of target relative to front of camera plane
    if panels is None or len(panels) < 1:
        message_to_embedded.X = 0
        message_to_embedded.Y = 0
        message_to_embedded.Z = 0
        message_to_embedded.status = 0
    else:
        biggest_panel = max(panels, key=lambda q: q.area)
        message_to_embedded.X = float(biggest_panel.tvec[0]) / 1E2 # cm to m
        message_to_embedded.Y = float(biggest_panel.tvec[2]) / 1E2
        message_to_embedded.Z = float(-biggest_panel.tvec[1]) / 1E2
        message_to_embedded.status = 1
    message_to_embedded.capture_delay = capture_delay
    if config.debug:
        print(f'''msg({f"X:{message_to_embedded.X:.4f}".rjust(7)}, {f"Y:{message_to_embedded.Y:.4f}".rjust(7)}, {f"Z:{message_to_embedded.Z:.4f}".rjust(7)}, {f"delay:{message_to_embedded.capture_delay}"}ms")''')

    try:
        port.write(bytes(message_to_embedded))
    except Exception as error:
        print(f"\n[Communication]: error when writing over UART: {error}")
        port = setup_serial_port()  # attempt re-setup

    
# overwrite function if port is None
if port is None:
    def when_aiming_refreshes(panels, start_time):
        pass  # do nothing intentionally