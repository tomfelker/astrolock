import tkinter as tk
import tkinter.ttk as ttk
import tkinter.scrolledtext

import astropy.coordinates
import astropy.units as u

import skyfield
import skyfield.api

import astrolock.model.target_sources.opensky
import astrolock.model.astropy_util

import time

class TargetsFrame(tk.Frame):
    def __init__(self, *args, tracker, **kwargs):        
        tk.Frame.__init__(self, *args, **kwargs)

        self.tracker = tracker

        # hax:
        self.tracker.location_ap = astropy.coordinates.EarthLocation.from_geodetic(lat = 37.510839 * u.deg, lon = -122.272036 * u.deg, height = 64 * u.m)
        self.tracker.location_sf = skyfield.api.wgs84.latlon(latitude_degrees = 37.510839, longitude_degrees = -122.272036, elevation_m = 64)
        self.tracker.location_sfa = skyfield.api.wgs84.latlon(latitude_degrees = [37.510839], longitude_degrees = [-122.272036], elevation_m = [64])


        self.targets_treeview = ttk.Treeview(self, show = 'headings')
        self.targets_treeview['columns'] = ('callsign', 'url', 'latitude', 'longitude', 'altitude', 'azimuth', 'distance')
        self.targets_treeview.heading('callsign', text = 'Callsign')
        self.targets_treeview.heading('url', text = 'URL')
        self.targets_treeview.heading('latitude', text = 'Latitude')
        self.targets_treeview.heading('longitude', text = 'Longitude')
        self.targets_treeview.heading('altitude', text = 'Altitude')
        self.targets_treeview.heading('azimuth', text = 'Azimuth')
        self.targets_treeview.heading('distance', text = 'Distance')
        self.targets_treeview.pack(side = 'left', fill = 'both', expand = True)
        self.targets_treeview_scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.targets_treeview.yview)
        self.targets_treeview_scrollbar.pack(side='right', fill='y')
        self.targets_treeview.configure(yscrollcommand=self.targets_treeview_scrollbar.set)

        self.connection_source = astrolock.model.target_sources.opensky.OpenSkyTargetSource(self.tracker)
        self.connection_source.start()

    
    def update_gui(self):
        targets = self.connection_source.get_targets()

        targets.sort(key = (lambda target : target.score), reverse = True)

        #hax:
        #targets = targets[0:5]

        old_selection = self.targets_treeview.selection()

        # it's insane that this is the best way... seems O(n^2)
        self.targets_treeview.delete(*self.targets_treeview.get_children())

        for target in targets:
            values = (target.display_name, target.url)
            self.targets_treeview.insert(parent = '', index = 'end', iid = target.url, values = values)

            for column in self.targets_treeview['columns']:
                if column in target.display_columns:
                    self.targets_treeview.set(item = target.url, column = column, value = target.display_columns[column])

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

