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
`uv run simulator.py`

There are many CLI arguments available, run `uv run simulator.py --help` for details.

```
uv run simulator.py --wind-strength 0 --policy mpc-fov --car-motion straight --car-speed 2
```
