# KISSTracker
# A really simple thing to drive Celestron mounts with a joystick
# and track objects moving at a constant speed.

import sys
import time
import pygame.joystick
import serial
import math
import time

class TelescopeConnection:
    def __init__(self):
        stream = serial.Serial('COM3', 9600)
        self.istream = stream
        self.ostream = stream

    # axis 0 is azimuth, yaw, or right ascension (the one which needs acceleration when the other is near 90 degrees)
    # axis 1 is altitude, pitch, or declination (which needs no compensation)
    def set_axis_rate(self, axis, arcseconds_per_second):
        if axis > 1 or axis < 0:
            raise "bad axis" 
        rate_int = round(abs(arcseconds_per_second) * 4)
        rate_int_clamped = min(rate_int, 255 * 256 - 1) #sic - rates above this, though representable, make the controller stop
        rate_hi = rate_int_clamped // 256
        rate_lo = rate_int_clamped % 256 
        cmd = bytes([
            ord('P'), # passthru
            3,
            16 + axis,
            6 if arcseconds_per_second > 0 else 7,
            rate_hi,
            rate_lo,
            0,
            0
            ])
      
        self.ostream.write(cmd)
        reply = self.istream.read(1)
        if reply[0] != ord('#'):
            raise "read error setting rate"

        # the rate we actually set, taking clamping and rounding into account
        return math.copysign(rate_int_clamped, arcseconds_per_second) / 4

    def get_axis_position_radians(self, axis):
        cmd = bytes([
            # passthru
            ord('P'),
            # length
            3,
            # destination id
            16 + axis,
            # msgId, MC_GET_POSITION
            1,
            # data 0 to 3
            0,
            0,
            0,
            # response bytes
            3
            ])
        self.ostream.write(cmd)
        angle_bytes = self.ostream.read(4)
        if angle_bytes[3] != ord('#'):
            raise "read error getting axis position"
        angle_int = angle_bytes[0] * 65536 + angle_bytes[1] * 256 + angle_bytes[2]
        angle_radians = angle_int / 16777216 * math.pi * 2
        return angle_radians



telescope = TelescopeConnection()

print("Current altitude:", telescope.get_axis_position_radians(1) * 180 / math.pi, "deg")

pygame.init()
pygame.joystick.init()
print("Number of joysticks: " + str(pygame.joystick.get_count()))
joystick = pygame.joystick.Joystick(0)
joystick.init()
print("Joystick name is " + joystick.get_name())
print("Joystick axis count is " + str(joystick.get_numaxes()))


dead_zone = 0.5

telescope_axis_to_joystick_axis = {
        0: 0,
        1: 1
    }

joystick_axis_to_scale = {
        0: 1.0,
        1: -1.0
    }

# buttons to increment the rate of acceleration
faster_button = 5
slower_button = 4

# emergency stop button
stop_button = 2

# button to double your slew rate in the current direction
accel_button = 1

# button to double your rate of acceleration while you are holding it down
sprint_button = 10

# try to speed up when near gimbals... not sure if this is well formed...
# it basically does the same thing as you get if you follow a fixed compass heading on earth, i.e.,
# a spiral towards the north pole.  But what you want is a great circle, which means your pitch rate must reverse
# so really, would need to reformulate into an angular velocity vector, and read both axes.
# but at least this way, the effect of x and y axis is similar regardless of pitch
do_gimbal_compensation = True

# state
axis_rates = {0: 0.0, 1: 0.0}
speed = 1
speed_buttons_down = False
last_time = time.perf_counter()

try:
    while True:
        # input handling

        if not speed_buttons_down:
            if joystick.get_button(faster_button):
                speed_buttons_down = True
                speed += 1
            elif joystick.get_button(slower_button):
                speed_buttons_down = True
                speed -= 1
        else:
            if not(joystick.get_button(faster_button) or joystick.get_button(slower_button)):
                speed_buttons_down = False

        speed_temp_add = 0
        if joystick.get_button(sprint_button):
            speed_temp_add = 1

        rate_scale = 1
        if joystick.get_button(accel_button):
            rate_scale = 2
        if joystick.get_button(stop_button):
            rate_scale = 0

        if do_gimbal_compensation:
            pitch_radians = telescope.get_axis_position_radians(1)
        
        for telescope_axis in range(0, 2):
            # need to call this somewhere, so doing it in this inner loop because the
            # serial interface is so slow that it would feel bad if we don't get the most up to date joystick value
            pygame.event.poll()
            joystick_axis = telescope_axis_to_joystick_axis[telescope_axis]
            joystick_value = joystick.get_axis(joystick_axis)
            if abs(joystick_value) < dead_zone:
                joystick_value = 0
                #continue would let us run faster, but means we'll accelerate too fast if in only one axis
                #continue
            joystick_value *= joystick_axis_to_scale[joystick_axis]
            joystick_value *= pow(2, speed + speed_temp_add)
            axis_rates[telescope_axis] += joystick_value
            axis_rates[telescope_axis] *= rate_scale

            desired_rate = axis_rates[telescope_axis]

            if do_gimbal_compensation and telescope_axis == 0:
                desired_rate /= math.cos(pitch_radians)
            
            achieved_rate = telescope.set_axis_rate(telescope_axis, desired_rate)

            if do_gimbal_compensation and telescope_axis == 0:
                achieved_rate *= math.cos(pitch_radians)

            axis_rates[telescope_axis] = achieved_rate


        current_time = time.perf_counter()
        dt = current_time - last_time
        last_time = current_time
        fps = 1 / dt if dt > 0 else math.inf
        print("Speed: 2^" + str(speed),
              "Pitch: " + str(round(pitch_radians * 180 / math.pi,5)) if do_gimbal_compensation else "",
              "Az rate:", axis_rates[0], "soa/s",
              "Alt rate:", axis_rates[1], "soa/s",
              "FPS:", round(fps,2)
              )
except KeyboardInterrupt:
    print("Interrupt")
finally:
    print("Stopping telescope")
    telescope.set_axis_rate(0, 0)
    telescope.set_axis_rate(1, 0)
    print("Done\n")














