import tkinter as tk
import tkinter.ttk as ttk

import astrolock.gui.status
import astrolock.gui.time

class MainWindow(tk.Tk):
    def __init__(self, *args, **kwargs):        
        tk.Tk.__init__(self, *args, **kwargs)

        notebook = ttk.Notebook(self)

        status_tab = astrolock.gui.status.StatusFrame(notebook)
        status_tab.pack()

        time_tab = astrolock.gui.time.TimeFrame(notebook)

        notebook.add(status_tab, text = "Status")
        notebook.add(time_tab, text = "Time")

        notebook.pack(fill = tk.BOTH)

def gui_loop():
    window = MainWindow()
    window.mainloop()