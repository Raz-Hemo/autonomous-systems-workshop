# Drone landing simulator
The autonomous systems workshop project, by Raz Hemo.

This project simulates a quadcopter drone that tries to land on a moving target.

## Setup

1. install python and uv (from https://github.com/astral-sh/uv). 
2.
```
uv venv
uv pip install -r requirements.txt
```

## Running 
`uv run run_viewer.py`

Wind can be configured from the CLI:

```
uv run run_viewer.py --wind-strength 0
uv run run_viewer.py --wind-strength 1.2
```

controls:

- w/a/s/d - move
- q/e - move down/up
- right mouse drag: look around
- mouse wheel: zoom
- escape or space: quit
