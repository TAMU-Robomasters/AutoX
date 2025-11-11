### How do I get this code to run?

Install python 3.10 and run `pip install -r requirements.txt`

### What is this repo?

This handles all main logic that runs on the Jetson.

### How does the code work?

- `main/main.py` is only ~12 lines of code
     - These are the 3 core functions in the codebase:
     - `model.when_frame_arrives()`
     - `aim.when_bounding_boxes_refresh()`
     - `communicate.when_aiming_refreshes()`
- Everything outside of those functions are just helpers for those functions 
- If a tool/function is generic (used in multiple places) put it in the toolbox folder
- If you need to set a constant (like `our_team_color`) do it in the `./main/info.yaml`
    - To use that value in python do:<br>
    ```py
    from toolbox.globals import path_to, config
    config.our_team_color
    ```

# At the competition

1. After ssh-ing into the Xavier, run `./run/xavier_kill_onboot_cv` to stop the thing from running
2. To change the team color, edit the `run/boot_command.ignore` change WE_RED to WE_BLUE or vice versa

# How to Setup New Xavier 

- Copy an existing xavier
    - Take the xavier SD card out of the jetson, then plug it into your computer
    - Execute `run/save_img_file_from_sd_card`
    - This will give you an img file on yout PC
- Push to a new xavier
    - Either get the img off the google drive or your PC
    - Run `run/push_img_file_to_sd_card`
    - its interactive, so just run the command and follow the instructions

After putting the SD card into the xavier run:

```sh
cd repos
git clone git@github.com:TAMU-Robomasters/cv_lite.git
cd cv_lite
sudo ./run/xavier_reset_zerotier
sudo ./run/xavier_setup_boot_script.js
```
