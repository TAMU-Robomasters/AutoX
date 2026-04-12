import 

class RerunEngine(Engine[AutoAimContext]):
    """Use classical cv to find panels and 3D info to select the best target."""

    def __init__(self):
        self.ctx = AutoAimContext()
        self.detection = ClassicalDetectorModule(self.ctx)
        self.selection = SelectingWith3DModule(self.ctx)
        super().__init__(
            modules=[
                self.detection,
                self.selection,
            ],
            context_type=AutoAimContext,
        )

    def initialize(self):  # noqa: D102

    def execute(self):  # noqa: D102
        # run rerun sending data 

 