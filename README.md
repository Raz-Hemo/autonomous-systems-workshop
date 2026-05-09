# Drone landing simulator
The autonomous systems workshop project, by Raz Hemo.

This project simulates a quadcopter drone that tries to land on a moving target.

## Setup

1. you should place, in this folder, the mujoco version `mujoco-3.6.0-windows-x86_64` from https://github.com/google-deepmind/mujoco/releases/tag/3.6.0
2. install python and uv (from https://github.com/astral-sh/uv). 
3.
```
uv venv
uv pip install -r requirements.txt
```

## Running 
`uv run run_viewer.py`

controls:

- w/a/s/d - move
- q/e - move down/up
- right mouse drag: look around
- mouse wheel: zoom
- escape or space: quit
