### How to setup code
- Download and install [git lfs](https://git-lfs.com/). This is used to manage large files for git.
- git clone with the ssh link
- run `git lfs install`
- Install [uv](https://docs.astral.sh/uv/). This is used to manage python packages. Modern version of pip or conda. 
    - Mac/Linux/WSL
    `curl -LsSf https://astral.sh/uv/install.sh | sh`
    - Windows
    `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
> [!WARNING]
> You may have to restart your Windows machine after installing uv
- run `uv sync`
- run `git pull`

### How do I get this code to run?
- Run `. .venv/bin/activate` and `python main.py @CAMERA=WEBCAM`
- Or run `uv run main.py @CAMERA=WEBCAM`

### What is this repo?

This handles all main logic that runs on the Jetson.

### How does the code work?
TODO

# At the competition
TODO

# How to Setup New Xavier 
TODO



```sh
cd repos
git clone git@github.com:TAMU-Robomasters/cv_lite.git
cd cv_lite
sudo ./run/xavier_reset_zerotier
sudo ./run/xavier_setup_boot_script.js
```
