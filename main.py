"""Entry point for the AutoX application.

Runs the production auto-aim system. The detection backend is selected by
``vision_backend`` in ``src/info.yaml``:
  * ``python`` (default) -> FullStateAutoAimEngine  (Python detector + shared camera driver)
  * ``cpp``              -> FullStateAutoAimEngineCpp (C++ detector owns the camera)
"""
from src.main import main

if __name__ == "__main__":
    main()
