import tkinter as tk
import tkinter.ttk as ttk

from astrolock.model.alignment import *

class AlignmentFrame(tk.Frame):
    def __init__(self, *args, tracker, **kwargs):        
        tk.Frame.__init__(self, *args, **kwargs)

        self.tracker = tracker

        label = tk.Label(self, text="Alignment")
        label.pack()


