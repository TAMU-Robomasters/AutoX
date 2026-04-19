
import rerun as rr
import av
from scipy.spatial.transform import Rotation as rot
import random
import time

from src.core.module import Module, real
from src.types.autoaim import AutoAimContext

class Simulation(Module[AutoAimContext]):

    def __init__(self, context: AutoAimContext):
        super().__init__(
            name="rerun_simulation",
            context=context,
            inputs=["panels"],
            outputs=[],
        )

    def init_stuff(self) -> None:
        rr.init("rerun_fake_data_test", spawn=False, recording_id="pleaseeework")
        
        #computer stuff needs to be sent to tailscale ip
        laptopIP = "10.245.54.24"
        rr.connect_grpc(f"rerun+http://{laptopIP}:9876/proxy")

        rr.set_time("stable_time", duration=0)
        rr.log("logs", rr.TextLog("please work"))
        #self.laptopCamera()
        #self.pipelineInit()
        #self.logCameras()
    
    def ending(self):
        time.sleep(100)
        print("completeee")
        rr.disconnect()

    def fakePanels(self):
        points = []
        for i in range(10):
            x = round(random.uniform(-2,2),2)
            y = round(random.uniform(-2,2),2)
            z = round(random.uniform(-2,2),2)
            points.append([x,y,z])
        return points

    @real()
    def _run_simulate(self, panels):
        self.logEverythingElse(panels)
        '''for f in self.input_container.decode(video=0):
            f.pict_type = av.video.frame.PictureType.NONE
            for packet in self.stream.encode(f):
                if packet.pts is None:
                    continue
                rr.set_time("stable_time", duration=float(packet.pts * packet.time_base))
                for c in range(4):
                    rr.log(f"bot/cam{c+1}", rr.VideoStream.from_fields(sample=bytes(packet)))
            panels = AutoAimContext.panels
            self.logEverythingElse(panels)'''
        
        for i in range(400):
            time = i*0.01
            rr.set_time("stable_time", duration=time)

    def laptopCamera(self):
        self.input_container = av.open("/dev/video0", format="v4l2")
        istr = self.input_container.streams.video[0]
        self.fps = 30
        self.duration_s = 4
        self.width = istr.width
        self.height = istr.height
        self.codec = rr.VideoCodec.H264

        self.formats = {rr.VideoCodec.H265: "hevc", rr.VideoCodec.H264: "h264"}
        self.encoders = {rr.VideoCodec.H265: "libx265", rr.VideoCodec.H264: "libx264"}
        print("found laptop camera")

    def pipelineInit(self):
        av.logging.set_level(av.logging.VERBOSE)
        container = av.open("/dev/null", "w", format=self.formats[self.codec])
        self.stream = container.add_stream(self.encoders[self.codec], rate=self.fps)
        assert isinstance(self.stream, av.video.stream.VideoStream)
        self.stream.bit_rate = 500000
        self.stream.width = self.width
        self.stream.height = self.height
        self.stream.pix_fmt = "yuv420p"
        self.stream.max_b_frames = 0
        print("found pipeline")

    def logCameras(self):
        camloc = [[0, 0.25, 0.5], [0.25, 0, 0.5], [0, -0.25, 0.5], [-0.25, 0.0, 0.5]]
        camrot = [0, 270, 180, 90]
        for c in range(4): #4
            quat = rot.from_euler('xz', [-90, camrot[c]], degrees=True).as_quat()
            rr.log(
                f"bot/cam{c+1}",
                rr.Pinhole(
                    width=self.width,
                    height=self.height,
                    focal_length=self.width/2,
                    image_plane_distance=0.2
                ),
                rr.Transform3D(
                    translation=camloc[c],
                    quaternion=quat
                ),
                static=True,
            )
            rr.log(f"bot/cam{c+1}", rr.VideoStream(codec=self.codec), static=True)
        print("logged cams")


    def logEverythingElse(self, panel):
        #  current_state.predicted_plates()[0].position(), current_state.predicted_plates()[1].position(), current_state.predicted_plates()[2].position(), current_state.predicted_plates()[3].position(),
        #             current_state.predicted_plates()[0].velocity(), current_state.predicted_plates()[1].velocity(), current_state.predicted_plates()[2].velocity(), current_state.predicted_plates()[3].velocity(),
        #             current_state.radius(), current_state.radius()))
        #PANEL MARKERS
        # panPos = []
        # pastP = panel[0].position()
        # print(f"Panel {1} x: {panel[0].position()[0]} y: {panel[0].position()[1]} z: {panel[0].position()[2]}")
        # for p in range(1,len(panel)):
        #     panPos.append(p.position())
        #     print(f"Panel {p+1} x: {panel[p].position()[0]} y: {panel[p].position()[1]} z: {panel[p].position()[2]}")
        #     dist = math.sqrt(math.pow(panel[p].position()[0]-panel[0].position()[0], 2) + math.pow(panel[p].position()[0]-panel[0].position()[0], 2) + math.pow(panel[p].position()[0]-panel[0].position()[2], 2))
        rr.log(
            f"markers",
            rr.Points3D(
                panel
            )
        )

        #MAIN ROBOT
        rr.log(
            "bot/main",
            rr.Cylinders3D(lengths=0.5, radii=0.3, colors=[128,128,200], fill_mode=3, centers=(0,0,0.25)),
        )
        rr.log(
            "bot",
            rr.Transform3D(translation=[2,2,0]),
        )
        
        #ESTIMATED OTHER ROBOT LOCATIONS
        '''rr.log(
            "otherbot/main",
            rr.Cylinders3D(lengths=0.5, radii=0.3, colors=[128,128,200], fill_mode=1, centers=(0,0,0))
        )
        panloc = [[0, 0.15, 0.05], [0.15, 0, 0.05], [0, -0.15, 0.05], [-0.15, 0.0, 0.05]]
        panrot = [0, 270, 180, 90]
        for c in range(4):
            quat = rot.from_euler('yz', [-90, panrot[c]], degrees=True).as_quat()
            rr.log(
                f"otherbot/pan{c+1}",
                rr.Boxes3D(
                    centers=panloc[c],
                    half_sizes=[0.1, 0.05, 0.1],
                    quaternions=quat,
                    radii=0.02,
                    colors=[200,128,128],
                    fill_mode=1,
                    labels=[f"pan{c+1}"],
                ),
                rr.Transform3D(translation=panloc[c]),
            )

        rr.log(
            "otherbot",
            rr.Transform3D(translation=[0,0,0.1]),
        )'''


