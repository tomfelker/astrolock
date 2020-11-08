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
        self.pygame_thread.start()

        #todo: cool gui for assigning axes...
        

    def update_gui(self):
        pass

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
                tracker_input = tracker.TrackerInput()
                tracker_input.rate[0] = joysticks[0].get_axis(0)
                tracker_input.rate[1] = -joysticks[0].get_axis(1)
                tracker_input.rate = self.apply_deadzone(tracker_input.rate)
                tracker_input.accel[0] = joysticks[0].get_axis(2)
                tracker_input.accel[1] = -joysticks[0].get_axis(3)
                tracker_input.accel = self.apply_deadzone(tracker_input.accel)
                tracker_input.braking = (joysticks[0].get_axis(4) + 1.0) / 2.0
                tracker_input.speedup = (joysticks[0].get_axis(5) + 1.0) / 2.0
                self.tracker.set_input(tracker_input)

    def apply_deadzone(self, vec, deadzone = .1):
        mag = np.dot(vec, vec)
        scale = 0
        if mag > deadzone:
            new_mag = (mag - deadzone) * (1 - deadzone)
            scale = new_mag / mag
        return vec * scale
