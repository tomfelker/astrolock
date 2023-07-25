import tkinter as tk
import tkinter.ttk as ttk

import astrolock.gui.status
import astrolock.gui.time
import astrolock.gui.alignment
import astrolock.gui.input
import astrolock.gui.targets
import astrolock.model.tracker as tracker

import gc
import os

class MainWindow(tk.Tk):
    def __init__(self, *args, **kwargs):        
        tk.Tk.__init__(self, *args, **kwargs)
           
        self.tracker = tracker.Tracker()

        notebook = ttk.Notebook(self)

        self.status_tab = astrolock.gui.status.StatusFrame(notebook, tracker = self.tracker)
        self.status_tab.pack()

        self.input_tab = astrolock.gui.input.InputFrame(notebook, tracker = self.tracker)
        self.input_tab.pack()

        self.time_tab = astrolock.gui.time.TimeFrame(notebook, tracker = self.tracker)
        self.time_tab.pack()

        self.alignment_tab = astrolock.gui.alignment.AlignmentFrame(notebook, tracker = self.tracker)
        self.alignment_tab.pack()

        self.targets_tab = astrolock.gui.targets.TargetsFrame(notebook, tracker = self.tracker)
        self.targets_tab.pack()

        notebook.add(self.status_tab, text = "Status")
        notebook.add(self.input_tab, text = "Input")
        notebook.add(self.time_tab, text = "Time")
        notebook.add(self.alignment_tab, text = "Alignment")
        notebook.add(self.targets_tab, text = "Targets")

        notebook.pack(fill = tk.BOTH, expand=True)

        self.bind("<Destroy>", self._destroy)

    def _destroy(self, *args, **kwargs):
        self.tracker.disconnect_from_telescope()

def tracker_on_idle(self):
    self.after(5, gc.collect)

def gui_loop():
    window = MainWindow()
    window.mainloop()
    # it'd be nice to let the tracker threads update cleanly, but for some reason that's not always working:
    os._exit(0)

