import tkinter as tk
import tkinter.ttk as ttk
import astrolock.model.tracker as tracker
import astropy.units as u

class StatusFrame(tk.Frame):
    def __init__(self, *args, tracker, **kwargs):        
        tk.Frame.__init__(self, *args, **kwargs)

        self.tracker = tracker

        connections = self.tracker.get_recommended_connection_urls()
        if len(connections) == 0:
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

        self.alt_rate_slider = tk.Scale(self, from_ = 5, to = -5, resolution = .1, length = 130, sliderlength = 30, orient = tk.VERTICAL, command = self.sliders_changed)    
        self.alt_rate_slider.pack() 
        self.az_rate_slider = tk.Scale(self, from_ = -5, to = 5, resolution = .1, length = 130, sliderlength = 30, orient = tk.HORIZONTAL, command = self.sliders_changed)
        self.az_rate_slider.pack()

        #hmm, how to do sizes in cm?
        self.label = tk.Label(self, text="Status", font=("TkFixedFont"), anchor = 'nw', justify = 'left', width = 80, height = 25)
        self.label.pack()

    def start(self):
        self.tracker.connect_to_telescope(self.connections_dropdown_choice.get()) 

    def stop(self):
        self.tracker.disconnect_from_telescope()

    def update_gui(self):
        if self.tracker is not None:
            self.label.config(text = self.tracker.get_status())

            self.az_rate_slider.set(self.tracker.tracker_input.rate[0])
            self.alt_rate_slider.set(self.tracker.tracker_input.rate[1])
            
    def sliders_changed(self, val):
        if False:
            tracker_input = tracker.TrackerInput()
            tracker_input.rate[0] = self.az_rate_slider.get()
            tracker_input.rate[1] = self.alt_rate_slider.get()
            self.tracker.set_input(tracker_input)
    