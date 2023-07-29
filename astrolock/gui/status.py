import tkinter as tk
import tkinter.ttk as ttk
import astrolock.model.tracker as tracker
import astropy.units as u

class StatusFrame(tk.Frame):
    def __init__(self, *args, tracker, **kwargs):        
        tk.Frame.__init__(self, *args, **kwargs)

        self.tracker = tracker
        self.tracker.status_observers.append(self)

        connection_frame = tk.LabelFrame(self, text="Connection")
        connection_frame.grid(row=0, column=0, sticky='nsew')
        self.columnconfigure(0, weight=1, uniform='fred') # you've gotta be kidding me
        self.columnconfigure(1, weight=1, uniform='fred')

        connections = self.tracker.get_recommended_connection_urls()
        if len(connections) == 0:
            connections = ["No connections found"]
        print('\n'.join(connections))
        self.connections_dropdown_choice = tk.StringVar(self, connections[0])
        self.connections_dropdown = tk.OptionMenu(connection_frame, self.connections_dropdown_choice, *connections)
        #ttk.OptionMenu(self, self.connections_dropdown_choice, *connections)
        self.connections_dropdown.grid()

        start_button = tk.Button(connection_frame, text = "Start!", command = self.start)
        start_button.grid()
        
        stop_button = tk.Button(connection_frame, text = "Stop!", command = self.stop)
        stop_button.grid()

        mode_frame = tk.LabelFrame(self, text="Mode")
        mode_frame.grid(row=0, column=1, sticky='nsew')

        self.mode_var = tk.StringVar(self)
        self.mode_dropdown = tk.OptionMenu(mode_frame, self.mode_var, *self.tracker.modes, command=self.on_mode_selected)
        self.mode_dropdown.grid()

        status_frame = tk.LabelFrame(self, text="Status")
        status_frame.grid(row=1, columnspan=2, sticky='wse')

        #hmm, how to do sizes in cm?
        self.label = tk.Label(status_frame, text="Status", font=("TkFixedFont"), anchor = 'nw', justify = 'left', width = 120, height = 25)
        self.label.pack()

        self.bind("<Destroy>", self._destroy)

    def _destroy(self, *args, **kwargs):
        self.tracker.status_observers.remove(self)

    def start(self):
        self.tracker.connect_to_telescope(self.connections_dropdown_choice.get()) 

    def stop(self):
        self.tracker.disconnect_from_telescope()

    def on_mode_selected(self, *args, **kwargs):
        self.tracker.current_mode = self.mode_var.get()

    def on_tracker_status_changed(self):
        self.after(1, self.update_status_label)

    def update_status_label(self):
        if self.tracker is not None:
            self.label.config(text = self.tracker.get_status())
            self.mode_var.set(self.tracker.current_mode)
