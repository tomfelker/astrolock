import tkinter as tk
import tkinter.ttk as ttk

class TimeFrame(tk.Frame):
    def __init__(self, *args, tracker, **kwargs):        
        tk.Frame.__init__(self, *args, **kwargs)

        self.tracker = tracker
        self.tracker.status_observers.append(self)

        self.label = tk.Label(self, text="Time")
        self.label.pack()

    def on_tracker_status_changed(self):
        self.label.config(text=str(self.tracker.get_time()))