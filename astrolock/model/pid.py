from astrolock.model.util import *

class PIDController:
    def __init__(self):
        self.proportional_gain = 1.0
        pass

    def compute_control_rate(
        self,
        desired_position,
        desired_rate,
        commanded_rate,
        measured_position,
        measurement_seconds_ago
    ):
        # we assume the telescope is moving at the last commanded rate
        predicted_position_at_desired_time = measured_position + commanded_rate * measurement_seconds_ago

        error = desired_position - predicted_position_at_desired_time
        error = wrap_angle_plus_minus_pi_radians(error)

        control_rate = desired_rate + error * self.proportional_gain
        return control_rate
