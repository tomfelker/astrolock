import tkinter as tk
import tkinter.ttk as ttk

class TimeFrame(tk.Frame):
    def __init__(self, *args, **kwargs):        
        tk.Frame.__init__(self, *args, **kwargs)

        label = tk.Label(self, text="Time")
        label.pack()
