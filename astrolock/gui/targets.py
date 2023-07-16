import tkinter as tk
import tkinter.ttk as ttk
import tkinter.scrolledtext

import astropy.coordinates
import astropy.units as u

import skyfield
import skyfield.api


import astrolock.model.astropy_util

import time

class TargetsFrame(tk.Frame):
    def __init__(self, *args, tracker, **kwargs):        
        tk.Frame.__init__(self, *args, **kwargs)

        self.tracker = tracker
 
        self.target_source = None

        self.selected_target_source_name = tk.StringVar()
        self.selected_target_source_name.trace_add('write', self.selected_target_source_name_changed)
        self.selected_target_source_name.set(next(iter(self.tracker.target_source_map)))

        self.target_source_menu = tk.OptionMenu(self, self.selected_target_source_name, *list(self.tracker.target_source_map.keys()))
        self.target_source_menu.pack()

        track_selected_button = tk.Button(self, text = "Track Selected Target", command = self.track_selected)
        track_selected_button.pack()

        stop_tracking_button = tk.Button(self, text = "Stop Tracking", command = self.stop_tracking)
        stop_tracking_button.pack()

        self.filter_string = tk.StringVar()
        self.filter_string.trace_add("write", self.filter_string_changed)
        filter_entry = tk.Entry(self, textvariable=self.filter_string)
        filter_entry.pack()

        self.targets_treeview = ttk.Treeview(self, show = 'headings')
        self.targets_treeview.bind("<Double-1>", self.track_selected)
        self.targets_treeview['columns'] = ('callsign', 'url', 'latitude', 'longitude', 'altitude', 'azimuth', 'distance', 'age')
        self.targets_treeview.heading('callsign', text = 'Callsign')
        self.targets_treeview.heading('url', text = 'URL')
        self.targets_treeview.heading('latitude', text = 'Latitude')
        self.targets_treeview.heading('longitude', text = 'Longitude')
        self.targets_treeview.heading('altitude', text = 'Altitude')
        self.targets_treeview.heading('azimuth', text = 'Azimuth')
        self.targets_treeview.heading('distance', text = 'Distance')
        self.targets_treeview.heading('age', text = 'Age')
        self.targets_treeview.pack(side = 'left', fill = 'both', expand = True)
        self.targets_treeview_scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.targets_treeview.yview)
        self.targets_treeview_scrollbar.pack(side='right', fill='y')
        self.targets_treeview.configure(yscrollcommand=self.targets_treeview_scrollbar.set)


    def selected_target_source_name_changed(self, *args):
        print(f'selected {self.selected_target_source_name}')
        
        old_target_source = self.target_source
        new_target_source = self.tracker.target_source_map[self.selected_target_source_name.get()]
        if old_target_source != new_target_source:
            self.target_source = new_target_source

            if old_target_source:
                old_target_source.stop()
                old_target_source.observers.remove(self)
            
            if new_target_source:
                new_target_source.observers.append(self)
                new_target_source.start()


    def filter_string_changed(self, *args):
        self.update_targets_treeview()


    def on_target_source_updated(self, target_source):
        """
        Called from the target source's thread when it has new targets for us.
        """
        # this tells the tracker to look up potential new target info so it can do extrapolation for planes etc.
        # todo: should this be its own observer? need to get from target to source then, which can add
        self.tracker.update_targets(target_source.target_map)
        
        # must schedule this instead of calling directly, since we're on a different thread
        self.after(0, self.update_targets_treeview)

    def get_filtered_targets(self):
        if self.target_source is not None:
            targets = list(self.target_source.target_map.values())
            filter_string = self.filter_string.get().strip().casefold()
            if len(filter_string) > 0:
                targets = list(filter(lambda target: filter_string in target.display_name.casefold() or filter_string in target.url.casefold(), targets))
            targets.sort(key = (lambda target : target.score), reverse = True)
            return targets
        return []

    def update_targets_treeview(self):
        self.targets_dirty = False

        targets = self.get_filtered_targets()

        old_selection = self.targets_treeview.selection()

        # it's insane that this is the best way... seems O(n^2)
        self.targets_treeview.delete(*self.targets_treeview.get_children())

        ap_now = astropy.time.Time.now()

        for target in targets:
            values = (target.display_name, target.url)
            self.targets_treeview.insert(parent = '', index = 'end', iid = target.url, values = values)

            for column in self.targets_treeview['columns']:
                if column in target.display_columns:
                    self.targets_treeview.set(item = target.url, column = column, value = target.display_columns[column])

            if target.last_known_location_time is not None:
                age = ap_now - target.last_known_location_time
                self.targets_treeview.set(item = target.url, column = 'age', value = age.to_value(u.s))

        try:
            self.targets_treeview.selection_set(old_selection)
        except:
            # the old target may have gone away... this is lame, but not worth fixing since eventually we'll have a map and remember targets
            pass

    def track_selected(self, *args):
        if self.target_source is not None:
            selected_urls = self.targets_treeview.selection()
            if len(selected_urls) == 1:
                self.tracker.set_target(self.target_source.target_map[selected_urls[0]])

    def stop_tracking(self):
        for selected_item in self.targets_treeview.selection():
            self.targets_treeview.selection_remove(selected_item)
        self.tracker.set_target(None)
