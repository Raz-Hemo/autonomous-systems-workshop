import mujoco
import numpy as np
import pytest

from sim.config import MODEL_PATH
from sim.motor_mixing import DroneController


@pytest.fixture
def controller() -> DroneController:
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    return DroneController(model, data)


def test_collective_thrust_is_evenly_distributed(controller: DroneController) -> None:
    forces = controller.mix_rotors(8.0, 0.0, 0.0, 0.0)
    np.testing.assert_allclose(forces, np.full(4, 2.0), atol=1e-6)


def test_more_total_thrust_raises_all_rotors(controller: DroneController) -> None:
    low = controller.mix_rotors(4.0, 0.0, 0.0, 0.0)
    high = controller.mix_rotors(8.0, 0.0, 0.0, 0.0)
    assert np.all(high > low)


def test_positive_roll_torque_splits_left_and_right_pairs(
    controller: DroneController,
) -> None:
    forces = controller.mix_rotors(8.0, 0.4, 0.0, 0.0)
    assert forces[0] > forces[1]
    assert forces[2] > forces[3]


def test_positive_pitch_torque_splits_front_and_rear_pairs(
    controller: DroneController,
) -> None:
    forces = controller.mix_rotors(8.0, 0.0, 0.4, 0.0)
    assert forces[2] > forces[0]
    assert forces[3] > forces[1]


def test_positive_yaw_torque_follows_spin_sign_pattern(
    controller: DroneController,
) -> None:
    forces = controller.mix_rotors(8.0, 0.0, 0.0, 0.04)
    assert forces[0] > forces[1]
    assert forces[3] > forces[2]


def test_saturated_commands_stay_within_rotor_limits(
    controller: DroneController,
) -> None:
    forces = controller.mix_rotors(100.0, 10.0, -10.0, 10.0)
    assert np.all(forces >= 0.0)
    assert np.all(forces <= 8.0)
