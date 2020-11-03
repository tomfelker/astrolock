import tkinter as tk
import tkinter.ttk as ttk
import astrolock.model.telescope_connection as telescope_connection
import astropy.units as u

class StatusFrame(tk.Frame):
    def __init__(self, *args, **kwargs):        
        tk.Frame.__init__(self, *args, **kwargs)

        self.current_connection = None

        connections = telescope_connection.TelescopeConnection.get_urls()
        if len(connections) is 0:
            connections = ["No connections found"]
        print('\n'.join(connections))
        self.connections_dropdown_choice = tk.StringVar(self, connections[0])
        self.connections_dropdown = tk.OptionMenu(self, self.connections_dropdown_choice, *connections)
        #ttk.OptionMenu(self, self.connections_dropdown_choice, *connections)
        self.connections_dropdown.pack()

        start_button = tk.Button(self, text = "Start!", command = self.start)
        start_button.pack()
        
        stop_button = tk.Button(self, text = "Stop!", command = self.stop)
        stop_button.pack()

        self.alt_rate_slider = tk.Scale(self, from_ = -10, to = 10, resolution = .1, length = 130, sliderlength = 30, orient = tk.VERTICAL)    
        self.alt_rate_slider.pack() 
        self.az_rate_slider = tk.Scale(self, from_ = -10, to = 10, resolution = .1, length = 130, sliderlength = 30, orient = tk.HORIZONTAL)
        self.az_rate_slider.pack()

        self.label = tk.Label(self, text="Status", font=("TkFixedFont"))
        self.label.pack()

    def start(self):
        if self.current_connection is None:
            self.current_connection = telescope_connection.TelescopeConnection.get_connection(self.connections_dropdown_choice.get())
            self.current_connection.set_update_callback(self.connection_thread_update)
            self.current_connection = self.current_connection.__enter__()       

        print('start')
        pass

    def stop(self):
        print('stop')
        if self.current_connection is not None:
            self.current_connection.__exit__(None, None, None)
            self.current_connection = None
        pass

    def connection_thread_update(self):
        self.after(0, self.main_thread_update)

    def main_thread_update(self):
        if self.current_connection is None:
            self.label.config(text = "Not connected")
        else:
            self.label.config(text = self.current_connection.get_status_string())
            self.current_connection.desired_axis_rates[0] = float(self.alt_rate_slider.get()) * (u.deg / u.s)
            self.current_connection.desired_axis_rates[1] = float(self.az_rate_slider.get()) * (u.deg / u.s)
        