# Configuration: `info.yaml` and local overrides

`src/info.yaml` is the single source of truth for **all** configurable constants in
AutoX — tunables, file paths, hardware profiles, and mock toggles. The rule of
thumb: prefer adding a field here over hardcoding a constant in Python.

This page covers the file's structure, how profiles get merged, and the generated
local file you use to set your own per-machine defaults.

## Reading config in code

You access the resolved config through a dict-like object with attribute access:

```python
from src.toolbox.globals import config, path_to, absolute_path_to

config.ballistic.gravity        # a tunable
config.mock.enable              # the global mock switch
path_to.robot_constants         # a path, relative to the repo
absolute_path_to.log_file       # the same path, made absolute
```

`config` is the result of loading `info.yaml` **and** merging in whichever profiles
are selected (see below). `path_to` / `absolute_path_to` come from the
`(path_to)` block so file locations live in config too, not scattered as string
literals.

## The shape of `info.yaml`

Names in parentheses are special to the loader (`quik_config`); everything else is
ordinary config data.

- **`(path_to)`** — a central place for file paths (test videos, the model, the
  learned-constants file, the log file…).
- **`(local_data)`** — points at the generated local override file
  (`./local_data.ignore.yaml`); see below.
- **`(profiles)`** — the config itself, organized as a base block plus optional
  overlays:
  - **`(default)`** — always loaded. Holds the base values for everything
    (`classical` detector thresholds, `hardware`, team color, `ballistic`,
    `estimation`, `log`, `mock`, …).
  - **named profile blocks** — merged *on top of* `(default)` when selected.

## Profiles: selecting and merging

A profile is a named block that gets merged over `(default)` when you pick it.
Profiles are grouped by an option name; selecting one on the CLI needs an `@`
prefix:

```sh
uv run main.py @CAMERA=WEBCAM @MCU=SERIAL
```

```{warning}
The `@` is required for **profile selection**. A bare `MCU=SERIAL` (no `@`) is
parsed as a YAML *override* of a key called `MCU` and will error. Use `@MCU=SERIAL`.
```

The option groups that exist today include:

| Option | Values | Picks |
| --- | --- | --- |
| `BOARD` | `LAPTOP` / `XAVIER` | laptop vs. Jetson |
| `CAMERA` | `NONE` / `WEBCAM` / `REALSENSE` | camera backend |
| `MCU` | `MOCK` / `SERIAL` / `ROS` | how the engine reaches the devboard |
| `GPU` | `NONE` / `TENSOR_RT` | inference backend |
| `ROS` | `NONE` / `NAV2` | whether the nav2 stack is running |
| `ESTIMATION` | `PARTICLE_FILTER` / `KALMAN_FILTER` | estimator (archived engine only) |

If you select nothing, the defaults (set in `src/toolbox/globals.py`) are
`GPU=NONE BOARD=LAPTOP CAMERA=NONE MCU=MOCK ROS=NONE ESTIMATION=PARTICLE_FILTER` —
a laptop with no hardware attached.

```{note}
Some selections are validated at startup. For example `MCU=ROS` requires `ROS=NAV2`
(the ROS backend forwards to the nav2-owned serial node); picking `MCU=ROS` without
it raises a clear `ValueError` at load instead of failing later at runtime.
```

## Overriding a single value (without a profile)

To change one value rather than select a whole profile, pass `KEY=VALUE` (no `@`).
This sets/overrides a config key directly:

```sh
uv run main.py our_team_color=blue
```

Use this for one-off tweaks; for anything you'll repeat, put it in the local file.

## The local file: `local_data.ignore.yaml`

`src/local_data.ignore.yaml` is **generated and gitignored** (it matches
`**/*.ignore.*`). It's where your machine remembers its choices so you don't retype
them every run, and where local secrets go. It has two special blocks:

```yaml
(selected_profiles):
- CAMERA=NONE          # profiles applied automatically on every run
(secrets):
   example: key29i5805 # local-only values (tokens, keys) — never committed
```

- **`(selected_profiles)`** — the profiles to apply by default, as if you'd typed
  them on the CLI. Add a line per profile (e.g. `- MCU=SERIAL`) and it's in effect
  for every `uv run` on this machine. Profiles you pass on the CLI still layer on
  top.
- **`(secrets)`** — local-only key/values you don't want in the repo.

### How to add to it

1. Pick the profiles/values you want as your machine's defaults and add them under
   `(selected_profiles)` (one `- OPTION=VALUE` per line).
2. Put any tokens/keys under `(secrets)`.
3. Run normally — the file is read automatically; no CLI flags needed for what's
   listed there.

```{tip}
If config seems "stuck" on a value you didn't pass, `local_data.ignore.yaml` is the
first place to look — its `(selected_profiles)` apply on every run.
```

## Where to put a new constant

Adding a tunable? Put it in the `(default)` block of `info.yaml` and read it via
`config.<group>.<name>`. If it should differ between laptop and Jetson (or any
other axis), add the differing value to the relevant profile block so the base
stays sane and the profile overlays the change.
