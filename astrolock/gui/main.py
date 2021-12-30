import tkinter as tk
import tkinter.ttk as ttk

import astrolock.gui.status
import astrolock.gui.time
import astrolock.gui.input
import astrolock.gui.targets
import astrolock.model.tracker as tracker

class MainWindow(tk.Tk):
    def __init__(self, *args, **kwargs):        
        tk.Tk.__init__(self, *args, **kwargs)

        self.tracker = tracker.Tracker()
        self.tracker.update_gui_callback = self.update_gui_from_tracker_threads

        notebook = ttk.Notebook(self)

        self.status_tab = astrolock.gui.status.StatusFrame(notebook, tracker = self.tracker)
        self.status_tab.pack()

        self.input_tab = astrolock.gui.input.InputFrame(notebook, tracker = self.tracker)
        self.input_tab.pack()

        self.time_tab = astrolock.gui.time.TimeFrame(notebook, tracker = self.tracker)
        self.time_tab.pack()

        self.targets_tab = astrolock.gui.targets.TargetsFrame(notebook, tracker = self.tracker)
        self.targets_tab.pack()

        notebook.add(self.status_tab, text = "Status")
        notebook.add(self.input_tab, text = "Input")
        notebook.add(self.time_tab, text = "Time")
        notebook.add(self.targets_tab, text = "Targets")

        notebook.pack(fill = tk.BOTH)

        self.bind("<Destroy>", self._destroy)

        self.ui_update_loop()

    # Trigger UI updates from the main (Tk) thread, for when the tracker is not running
    def ui_update_loop(self):
        self.after(1000, self.ui_update_loop)
        self.update_gui()

    # Called from the telescope connection threads when the status has changed, so we can show the newest info in the UI
    def update_gui_from_tracker_threads(self):
        try:
            self.after(0, self.update_gui)
        except RuntimeError:
            # to catch "main thread is not in main loop" on shutdown
            pass

    def update_gui(self):
        self.status_tab.update_gui()
        self.targets_tab.update_gui()

    def _destroy(self, *args, **kwargs):
        self.tracker.disconnect_from_telescope()

def gui_loop():
    window = MainWindow()
    window.mainloop()

