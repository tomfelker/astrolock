import tkinter as tk
import tkinter.ttk as ttk

class TimeFrame(tk.Frame):
    def __init__(self, *args, tracker, **kwargs):        
        tk.Frame.__init__(self, *args, **kwargs)

        self.tracker = tracker
        self.tracker.status_observers.append(self)

        request_gps_button = tk.Button(self, text = "Request GPS from Telescope", command = self.request_gps_from_telescope)
        request_gps_button.pack()

        self.label = tk.Label(self, text="Time")
        self.label.pack()

        self.bind("<Destroy>", self._destroy)

    def _destroy(self, *args, **kwargs):
        self.tracker.status_observers.remove(self)

    def on_tracker_status_changed(self):
        self.after(0, self.update_label)

    def update_label(self):
        self.label.config(text=str(self.tracker.get_time()))

    def request_gps_from_telescope(self):
        if self.tracker.primary_telescope_connection is not None:
            self.tracker.primary_telescope_connection.request_gps()
        