import math
import tkinter as tk
import tkinter.ttk as ttk
import astrolock.model.tracker as tracker
import astropy.units as u
import threading
import pygame
import pygame.event
import pygame.joystick
import numpy as np
import time

class InputFrame(tk.Frame):
    def __init__(self, *args, tracker, **kwargs):        
        tk.Frame.__init__(self, *args, **kwargs)

        self.tracker = tracker

        # The system tries to send joystick events crazy fast, throttle it a bit so we don't melt the CPU.
        # My PS4 DualShock controller (CUH-ZCT2U) sends hundreds per second, up to a thousand if you're actively moving the sticks.
        # Kinda nice actually, but overkill for us, at least if it soaks up the whole CPU just integrating it.
        max_joystick_update_hz = 100
        self.min_joystick_update_period_ns = int(1e9 / max_joystick_update_hz)

        self.sensitivity_step = 2
        self.sensitivity_max = 2
        self.sensitivity_min = -8
        self.fine_adjust_factor = .2

        self.pygame_thread = threading.Thread(target = self.pygame_thread_func)
        self.pygame_thread.setDaemon(True)
        # after is needed so the thread doesn't start throwing updates at us before we're in the Tkinter main loop, which would raise exceptions.
        self.after(0, self.pygame_thread.start)

        #todo: cool gui for assigning axes...
        
    def pygame_thread_func(self):
        pygame.init()
        pygame.joystick.init()



        last_joystick_update_ns = time.perf_counter_ns()
        skipped_updates = 0

        joystick_motion_events = [
            pygame.JOYAXISMOTION,
            pygame.JOYBALLMOTION,
            pygame.JOYBUTTONDOWN,
            pygame.JOYBUTTONUP,            
        ]

        while True:
            event = pygame.event.wait()

            if event.type in joystick_motion_events:                
                ns = time.perf_counter_ns()
                ns_since_update = ns - last_joystick_update_ns                
                if ns_since_update < self.min_joystick_update_period_ns:
                    skipped_updates += 1
                else:
                    last_joystick_update_ns = ns
                    #print(f'Skipped {skipped_updates} this {ns_since_update * 1e-6} ms period.')
                    dt = ns_since_update * 1e-9

                    skipped_updates = 0

                    left_stick = np.zeros(2)
                    right_stick = np.zeros(2)
                    
                    # just assume a PS4 controller (sigh, PS4 is different...)
                    left_stick[0] = joysticks[0].get_axis(0)
                    left_stick[1] = -joysticks[0].get_axis(1)
                    right_stick[0] = joysticks[0].get_axis(2)
                    right_stick[1] = -joysticks[0].get_axis(3)
                    left_trigger = (joysticks[0].get_axis(4) + 1.0) / 2.0
                    right_trigger = (joysticks[0].get_axis(5) + 1.0) / 2.0
                                        
                    button_cross = joysticks[0].get_button(0)
                    button_circle = joysticks[0].get_button(1)                    
                    button_square = joysticks[0].get_button(2)
                    button_triangle = joysticks[0].get_button(3)
                    button_share = joysticks[0].get_button(4)
                    button_ps = joysticks[0].get_button(5)
                    button_options = joysticks[0].get_button(6)
                    button_l3 = joysticks[0].get_button(7)
                    button_r3 = joysticks[0].get_button(8)
                    button_l1 = joysticks[0].get_button(9)
                    button_r1 = joysticks[0].get_button(10)
                    button_dpad_u = joysticks[0].get_button(11)
                    button_dpad_d = joysticks[0].get_button(12)
                    button_dpad_l = joysticks[0].get_button(13)
                    button_dpad_r = joysticks[0].get_button(14)
                    button_touchpad = joysticks[0].get_button(15)

                    button_l2 = left_trigger > .15
                    button_r2 = right_trigger > .15

                    # We will integrate various things onto tracker_input, and the telescope
                    # loops will calculate averages when they need it.

                    input = self.tracker.tracker_input
                    input.last_input_time_ns = ns
                    input.unconsumed_dt += dt

                    sensitivity_decrease_button = button_l1
                    if sensitivity_decrease_button and not input.sensitivity_decrease_button:
                        input.sensitivity -= 1
                    input.sensitivity_decrease_button = sensitivity_decrease_button

                    sensitivity_increase_button = button_r1
                    if sensitivity_increase_button and not input.sensitivity_increase_button:
                        input.sensitivity += 1
                    input.sensitivity_increase_button = sensitivity_increase_button

                    if input.sensitivity > self.sensitivity_max:
                        input.sensitivity = self.sensitivity_max
                    if input.sensitivity < self.sensitivity_min:
                        input.sensitivity = self.sensitivity_min

                    self.sensitivity_scale = math.pow(self.sensitivity_step, input.sensitivity)

                    input.last_rates = self.sensitivity_scale * (self.apply_deadzone(left_stick) + self.apply_deadzone(right_stick) * self.fine_adjust_factor)
                    input.last_braking = self.sensitivity_scale * self.apply_deadzone(right_trigger)

                    # "Aim down sights" will flip the axes to be correct for visual observation through the telescope
                    # TODO: UI for configuring this behavior
                    # for now, assume an SCT with a diagonal, whose image is natural vertically but flipped horizontally.
                    ads_button = button_l2
                    if ads_button:
                        input.last_rates[0] *= -1

                    input.integrated_rates += input.last_rates * dt
                    input.integrated_braking += input.last_braking * dt

                    emergency_stop_button = button_options
                    if emergency_stop_button:
                        input.last_rates *= 0
                        input.integrated_rates *= 0                        
                        input.emergency_stop_command = True

                    align_button = button_cross
                    if align_button and not input.align_button:
                        input.align_command = True
                    input.align_button = align_button

                    reset_offset_button = button_circle
                    if reset_offset_button and not input.reset_offset_button:
                        input.reset_command = True
                    input.reset_offset_button = reset_offset_button

                    self.tracker.notify_status_changed()

            elif event.type == pygame.JOYDEVICEADDED:
                joysticks = [pygame.joystick.Joystick(x) for x in range(pygame.joystick.get_count())]
                for joystick in joysticks:
                    joystick.init()

    def apply_deadzone(self, vec, deadzone = .15, power = 4):
        mag = np.math.sqrt(np.dot(vec, vec))
        if mag < deadzone:
            return np.zeros_like(vec)
        dir = vec / mag
        if mag > 1.0:
            mag = 1.0
                
        mag = (mag - deadzone) / (1.0 - deadzone)

        # a big curve for easy fine adjustment
        mag = pow(mag, power)

        return dir * mag

