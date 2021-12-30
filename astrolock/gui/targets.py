import tkinter as tk
import tkinter.ttk as ttk
import tkinter.scrolledtext

import astropy.coordinates
import astropy.units as u

import astrolock.model.target_sources.opensky
import astrolock.model.astropy_util

import time

class TargetsFrame(tk.Frame):
    def __init__(self, *args, tracker, **kwargs):        
        tk.Frame.__init__(self, *args, **kwargs)

        self.tracker = tracker

        # hax:
        self.tracker.location = astropy.coordinates.EarthLocation.from_geodetic(lat = "37d30'39.02\"", lon = "-122d16'19.33\"", height = 64)

        self.targets_treeview = ttk.Treeview(self, show = 'headings')
        self.targets_treeview['columns'] = ('callsign', 'url', 'latitude', 'longitude', 'altitude', 'azimuth', 'distance')
        self.targets_treeview.heading('callsign', text = 'Callsign')
        self.targets_treeview.heading('url', text = 'URL')
        self.targets_treeview.heading('latitude', text = 'Latitude')
        self.targets_treeview.heading('longitude', text = 'Longitude')
        self.targets_treeview.heading('altitude', text = 'Altitude')
        self.targets_treeview.heading('azimuth', text = 'Azimuth')
        self.targets_treeview.heading('distance', text = 'Distance')
        self.targets_treeview.pack()

        self.connection_source = astrolock.model.target_sources.opensky.OpenSkyTargetSource(self.tracker)
        self.connection_source.start()

    
    def update_gui(self):
        targets = self.connection_source.get_targets()

        #hax:
        #targets = targets[0:5]

        old_selection = self.targets_treeview.selection()

        # it's insane that this is the best way... seems O(n^2)
        self.targets_treeview.delete(*self.targets_treeview.get_children())

        tracker_altaz = astropy.coordinates.AltAz(location = self.tracker.location, obstime = 'J2000')

        for target in targets:
            values = (target.display_name, target.url)
            self.targets_treeview.insert(parent = '', index = 'end', iid = target.url, values = values)

            # the value we got from OpenSky
            self.targets_treeview.set(item = target.url, column = 'latitude', value = target.latitude_deg)
            self.targets_treeview.set(item = target.url, column = 'longitude', value = target.longitude_deg)
            
            # the cooked value (should be the same), but this is way too slow for some reason:
            #self.targets_treeview.set(item = target.url, column = 'latitude', value = target.location.lat.to_string(decimal = True))
            #self.targets_treeview.set(item = target.url, column = 'longitude', value = target.location.lon.to_string(decimal = True))

            start_time_ns = time.perf_counter_ns()

            #25 ms, wtf?
            #and our fast version is still 6 ms
            #target_altaz = target.location.itrs.transform_to(tracker_altaz)
            # maybe that was just due to pathing to find the appropriate transform?
            # even that's still 5ish ms
            target_altaz = astrolock.model.astropy_util.itrs_to_altaz(target.location.itrs, tracker_altaz)
            
            
            #print(f"transform took {(time.perf_counter_ns() - start_time_ns)*1e-6} ms")
            
            self.targets_treeview.set(item = target.url, column = 'altitude', value = target_altaz.alt.to_string(decimal = True))
            self.targets_treeview.set(item = target.url, column = 'azimuth', value = target_altaz.az.to_string(decimal = True))
            self.targets_treeview.set(item = target.url, column = 'distance', value = target_altaz.distance.to(u.km))
            
        try:
            self.targets_treeview.selection_set(old_selection)
        except:
            # the old target may have gone away... this is lame, but not worth fixing since eventually we'll have a map and remember targets
            pass

    def set_text(self, widget, text):
        # omfg why is this hard
        scroll_first, scroll_last = widget.yview()
        widget.delete('1.0', tk.END)
        widget.insert('1.0', text)
        widget.yview_moveto(scroll_first)

