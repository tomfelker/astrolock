import tkinter as tk
import tkinter.ttk as ttk
import astrolock.model.tracker as tracker
import astropy.units as u
import threading
import pygame
import pygame.event
import pygame.joystick
import numpy as np

class InputFrame(tk.Frame):
    def __init__(self, *args, tracker, **kwargs):        
        tk.Frame.__init__(self, *args, **kwargs)

        self.tracker = tracker

        self.pygame_thread = threading.Thread(target = self.pygame_thread_func)
        self.pygame_thread.setDaemon(True)
        # after is needed so the thread doesn't start throwing updates at us before we're in the Tkinter main loop, which would raise exceptions.
        self.after(0, self.pygame_thread.start)

        #todo: cool gui for assigning axes...
        
    def pygame_thread_func(self):
        pygame.init()
        pygame.joystick.init()

        joysticks = [pygame.joystick.Joystick(x) for x in range(pygame.joystick.get_count())]
        for joystick in joysticks:
            joystick.init()

        while True:
            event = pygame.event.wait()
            need_update = False
            if event.type == pygame.JOYAXISMOTION:
                need_update = True

            if need_update:
                #hmm, threading...
                #todo: some cool binding thing
                left_stick = np.zeros(2)
                right_stick = np.zeros(2)
                
                left_stick[0] = joysticks[0].get_axis(0)
                left_stick[1] = -joysticks[0].get_axis(1)
                right_stick[0] = joysticks[0].get_axis(2)
                right_stick[1] = -joysticks[0].get_axis(3)
                right_trigger = (joysticks[0].get_axis(4) + 1.0) / 2.0
                left_trigger = (joysticks[0].get_axis(5) + 1.0) / 2.0
                
                tracker_input = tracker.TrackerInput()
                tracker_input.rate = self.apply_deadzone(left_stick, power=2) + self.apply_deadzone(right_stick,power=4)*.1
                tracker_input.slowdown = left_trigger
                tracker_input.speedup = right_trigger

                self.tracker.set_input(tracker_input)

    def apply_deadzone(self, vec, deadzone = .1, power = 4):
        mag = np.math.sqrt(np.dot(vec, vec))
        if mag < deadzone:
            return np.zeros_like(vec)
        dir = vec / mag
                
        mag = (mag - deadzone) * (1 - deadzone)

        # a big curve for easy fine adjustment
        mag = pow(mag, power)

        return dir * mag

