import astropy.units as u
from astropy.coordinates import Angle

class PIDController:
    def __init__(self):
        self.proportional_gain = 1.0 / u.s
        pass

    def compute_control_rate(
        self,
        desired_position,
        desired_rate,
        desired_time,
        commanded_rate,
        measured_position,
        measured_position_time,
        store_state
    ):
        measured_to_desired_time = desired_time - measured_position_time
        # we assume the telescope is moving at the last commanded rate
        predicted_position_at_desired_time = measured_position + commanded_rate * measured_to_desired_time

        error = desired_position - predicted_position_at_desired_time
        error = Angle(error).wrap_at(180 * u.deg)

        control_rate = desired_rate + error * self.proportional_gain
        return control_rate
