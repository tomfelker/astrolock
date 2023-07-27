import tkinter as tk
import tkinter.ttk as ttk
import astropy
import astropy.coordinates
import astropy.units as u
import requests
import urllib.parse
import time

def time_string_with_delta(time, current_time):
    if time is None:
        return 'None'
    delta = (time - current_time).to_value(u.s)
    return f'{time.iso} (current {delta:+.3f} s)'

def location_string(location):
    # note weird order
    if location is None:
        return 'None'
    lon, lat, height = location.to_geodetic()
    return f'lat: {lat.to_value(u.deg): 3.8f} deg, lon: {lon.to_value(u.deg): 3.8f} deg, h: {height.to_value(u.m): 3.3f} m'

def parse_angle(angle_str):
    # astropy seems pretty nice here
    return astropy.coordinates.Angle(angle_str, unit=u.deg)


class TimeFrame(tk.Frame):
    def __init__(self, *args, tracker, **kwargs):        
        tk.Frame.__init__(self, *args, **kwargs)

        self.tracker = tracker
        self.tracker.status_observers.append(self)

        self.grid_columnconfigure(0, weight=1)

        commands_frame = ttk.LabelFrame(self, text="Commands")
        commands_frame.grid(columnspan=2, sticky='nwe')

        request_gps_button = tk.Button(commands_frame, text = "Request GPS from Telescope", command = self.request_gps_from_telescope)
        request_gps_button.grid()

        time_frame = ttk.LabelFrame(self, text="Time")
        time_frame.grid(row=1, column=0, sticky='nwe')

        self.use_telescope_time_var = tk.BooleanVar()
        self.use_telescope_time_var.set(self.tracker.use_telescope_time)
        self.use_telescope_time_var.trace_add('write', self.use_telescope_time_var_written)
        self.use_telescope_time_checkbutton = tk.Checkbutton(time_frame, text="Use telescope time if available", variable=self.use_telescope_time_var)
        self.use_telescope_time_checkbutton.grid(sticky='w')

        self.offset_frame = tk.Frame(time_frame)
        self.offset_frame.grid(sticky='w')
        self.offset_frame.columnconfigure(9, weight=1)

        tk.Label(self.offset_frame, text='Offset by:').grid(row=0, column=0)

        self.offset_d_var = tk.IntVar()
        self.offset_d = tk.Spinbox(self.offset_frame, from_=-3650, to=3650, textvariable=self.offset_d_var, width=5, command=self.time_offset_changed)
        self.offset_d.grid(row=0, column=1)
        tk.Label(self.offset_frame, text='days').grid(row=0, column=2)

        self.offset_h_var = tk.IntVar()
        self.offset_h = tk.Spinbox(self.offset_frame, from_=-24, to=24, textvariable=self.offset_h_var, width=5, command=self.time_offset_changed)
        self.offset_h.grid(row=0, column=3)
        tk.Label(self.offset_frame, text='hours').grid(row=0, column=4)

        self.offset_m_var = tk.IntVar()
        self.offset_m = tk.Spinbox(self.offset_frame, from_=-60, to=60, textvariable=self.offset_m_var, width=5, command=self.time_offset_changed)
        self.offset_m.grid(row=0, column=5)
        tk.Label(self.offset_frame, text='minutes').grid(row=0, column=6)

        self.offset_s_var = tk.IntVar()
        self.offset_s = tk.Spinbox(self.offset_frame, from_=-60, to=60, textvariable=self.offset_s_var, width=5, command=self.time_offset_changed)
        self.offset_s.grid(row=0, column=7)
        tk.Label(self.offset_frame, text='seconds').grid(row=0, column=8)

        self.current_time_label = tk.Label(time_frame, text="<current time>", font=("TkFixedFont"), anchor = 'nw', justify = 'left', width = 120, height = 5)
        self.current_time_label.grid()

        location_frame = ttk.LabelFrame(self, text="Location")
        location_frame.grid(row=2, column=0, sticky='nwe')

        self.use_telescope_location_var = tk.BooleanVar()
        self.use_telescope_location_var.set(self.tracker.use_telescope_location)
        self.use_telescope_location_var.trace_add('write', self.use_telescope_location_var_written)
        self.use_telescope_location_checkbutton = tk.Checkbutton(location_frame, text="Use telescope location if available", variable=self.use_telescope_location_var)
        self.use_telescope_location_checkbutton.grid(sticky='w')

        location_entry_frame = tk.Frame(location_frame)
        location_entry_frame.grid(row=3, column=0, sticky='nwe')

        tk.Label(location_entry_frame, text='lat (deg or dms):').grid(row=0, column=0)
        self.user_location_lat_text = tk.Entry(location_entry_frame)
        self.user_location_lat_text.grid(row=0, column=1)
        tk.Label(location_entry_frame, text='lon (deg or dms):').grid(row=0, column=2)
        self.user_location_lon_text = tk.Entry(location_entry_frame)
        self.user_location_lon_text.grid(row=0, column=3)
        tk.Label(location_entry_frame, text='height (m):').grid(row=0, column=4)
        self.user_location_height_text = tk.Entry(location_entry_frame)
        self.user_location_height_text.grid(row=0, column=5)
        self.user_location_button = tk.Button(location_entry_frame, text='Save to User Location', command=self.on_user_location_button_click)
        self.user_location_button.grid(row=0, column=6)

        address_entry_frame = tk.Frame(location_frame)
        address_entry_frame.grid(row=4, column=0, sticky='nwe')

        tk.Label(address_entry_frame, text='Address:').grid(row=0, column=0)
        self.user_address_text = tk.Entry(address_entry_frame, width=100)
        self.user_address_text.grid(row=0, column=1, sticky='we')
        self.user_address_button = tk.Button(address_entry_frame, text='Lookup and Save to User Location', command=self.on_user_address_button_click)
        self.user_address_button.grid(row=0, column=2)

        self.current_location_label = tk.Label(location_frame, text="<current location>", font=("TkFixedFont"), anchor = 'nw', justify = 'left', width = 120, height = 5)
        self.current_location_label.grid()

        self.bind("<Destroy>", self._destroy)

    def _destroy(self, *args, **kwargs):
        self.tracker.status_observers.remove(self)

    def use_telescope_time_var_written(self, *args, **kwargs):
        self.tracker.use_telescope_time = self.use_telescope_time_var.get()

    def use_telescope_location_var_written(self, *args, **kwargs):
        self.tracker.use_telescope_location = self.use_telescope_location_var.get()
        self.tracker.update_location()

    def time_offset_changed(self, *args, **kwarg):
        offset_s = (
            self.offset_d_var.get() * 24 * 60 * 60 +
            self.offset_h_var.get() * 60 * 60 +
            self.offset_m_var.get() * 60 +
            self.offset_s_var.get()
        )
        self.tracker.user_time_offset = offset_s * u.s

    def on_user_location_button_click(self, *args, **kwargs):
        try:
            lat = parse_angle(self.user_location_lat_text.get())
            lon = parse_angle(self.user_location_lon_text.get())
            height = u.Quantity(float(self.user_location_height_text.get()), unit=u.m)

            self.tracker.user_location = astropy.coordinates.EarthLocation.from_geodetic(lat=lat, lon=lon, height=height)
            self.tracker.update_location()
        except:
            pass

    def on_user_address_button_click(self, *args, **kwargs):
        address = self.user_address_text.get()        
        try:
            url = 'https://nominatim.openstreetmap.org/search/' + urllib.parse.quote(address) +'?format=json'
            # TODO: should be async...
            response = requests.get(url).json()
            lat = u.Quantity(response[0]["lat"], unit=u.deg)
            lon = u.Quantity(response[0]["lon"], unit=u.deg)
            #alas, no height
            self.tracker.user_location = astropy.coordinates.EarthLocation.from_geodetic(lat=lat, lon=lon)
            self.tracker.update_location()
        except:
            # TODO: don't do this
            pass

    def on_tracker_status_changed(self):
        self.after(0, self.update_times)
        self.after(0, self.update_locations)

    def update_times(self):
        if not self.winfo_viewable():
            return

        # TODO: impressive that these two consecutive calls, both checking the system time, have ~2 ms delay between them - is python / astropy really _that_ slow?? 
        current_time = self.tracker.get_time()
        system_time = astropy.time.Time.now()
        telescope_time = self.tracker.primary_telescope_connection.get_time() if self.tracker.primary_telescope_connection is not None else None

        if telescope_time is not None:            
            sync_age = (time.perf_counter_ns() - self.tracker.primary_telescope_connection.gps_measurement_time_ns) * 1e-9
            telescope_time_str = f'{time_string_with_delta(telescope_time, current_time)}, synced {sync_age:3.3f} s ago'
        else:
            telescope_time_str = 'None'

        text = (
            '\n'
            f'  Current time: {current_time.iso}\n'
            '\n'
            f'   System time: {time_string_with_delta(system_time, current_time)}\n'
            f'Telescope time: {telescope_time_str}\n'
        )
        self.current_time_label.config(text=text)

    def update_locations(self):
        if not self.winfo_viewable():
            return

        text = (
            '\n'
            f'  Current location: {location_string(self.tracker.location_ap)}\n'
            '\n'
            f'     User location: {location_string(self.tracker.user_location)}\n'
            f'Telescope location: {location_string(self.tracker.primary_telescope_connection.gps_location if self.tracker.primary_telescope_connection is not None else None)}\n'
        )
        self.current_location_label.config(text=text)

    def request_gps_from_telescope(self):
        if self.tracker.primary_telescope_connection is not None:
            self.tracker.primary_telescope_connection.request_gps()
        