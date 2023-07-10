import math
import random
import tkinter as tk
import tkinter.ttk as ttk
import tkinter.filedialog
import json
import os.path

import astrolock.model.alignment
from astrolock.model.alignment import AlignmentDatum

import astropy
from astropy import units as u

class AlignmentDatumTreeviewItem:
    def __init__(self, datum):
        self.datum = datum
        self.iid = None
        self.enabled = True

class AlignmentFrame(tk.Frame):
    def __init__(self, *args, tracker, **kwargs):        
        tk.Frame.__init__(self, *args, **kwargs)

        # actual data
        self.tracker = tracker
        self.observation_items = []

        # our whole tab should expand with the window (duh)
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # frames for the window layout
        tab_frame = ttk.Frame(self)
        tab_frame.grid(sticky='nsew')
        tab_frame.grid_rowconfigure(1, weight=1)
        tab_frame.grid_columnconfigure(2, weight=1)
        
        command_frame = ttk.LabelFrame(tab_frame, text="Commands")
        command_frame.grid(row=0, column=0, sticky='nw')        

        dev_command_frame = ttk.LabelFrame(tab_frame, text="Dev")
        dev_command_frame.grid(row=0, column=1, sticky='nw')        
        
        status_frame = ttk.LabelFrame(tab_frame, text="Current Alignment")
        status_frame.grid(row=0, column=2, sticky='we')        
        
        observations_frame = ttk.LabelFrame(tab_frame, text="Observations")        
        observations_frame.grid(row=1, column=0, columnspan=3, sticky='nsew')
        observations_frame.grid_rowconfigure(1, weight=1)
        observations_frame.grid_columnconfigure(0, weight=1)

        # command frame

        button = ttk.Button(command_frame, text = "Add Current Observation", command = self.add_observation)   
        button.grid(sticky='w')

        button = ttk.Button(command_frame, text = "Perform Alignment", command = self.align)
        button.grid(sticky='w')

        # dev command frame

        button = ttk.Button(dev_command_frame, text = "Add Test Observations", command = self.add_test_observations)   
        button.grid(sticky='w')

        button = ttk.Button(dev_command_frame, text = "Add Random Stepper Offsets", command = self.add_random_stepper_offsets)   
        button.grid(sticky='w')

        # status frame

        self.current_alignment_label = ttk.Label(status_frame, font=("TkFixedFont"), anchor = 'nw', justify = 'left')
        self.current_alignment_label.grid(sticky='nsew')

        # observations_frame

        observations_buttons_frame = ttk.Frame(observations_frame)
        observations_buttons_frame.grid(sticky='w')

        button = ttk.Button(observations_buttons_frame, text = "Load", command = self.load_observations)
        button.grid(row=0, column=0)

        button = ttk.Button(observations_buttons_frame, text = "Save As", command = self.save_observations)
        button.grid(row=0, column=1)

        button = ttk.Button(observations_buttons_frame, text = "Enable Selected", command = self.enable_selected_observations)
        button.grid(row=0, column=2)

        button = ttk.Button(observations_buttons_frame, text = "Disable Selected", command = self.disable_selected_observations)
        button.grid(row=0, column=3)

        button = ttk.Button(observations_buttons_frame, text = "Delete Selected", command = self.delete_selected_observations)
        button.grid(row=0, column=4)

        button = ttk.Button(observations_buttons_frame, text = "Delete All", command = self.delete_all_observations)
        button.grid(row=0, column=5)

        observations_treeview_frame = ttk.Frame(observations_frame)
        observations_treeview_frame.grid(sticky='nsew')
        observations_treeview_frame.grid_rowconfigure(0, weight=1)
        observations_treeview_frame.grid_columnconfigure(0, weight=1)        

        self.alignment_data_treeview = ttk.Treeview(observations_treeview_frame, show = 'headings')
        self.alignment_data_treeview['columns'] = ('target_name', 'target_url', 'time', 'raw_axis_0', 'raw_axis_1')
        self.alignment_data_treeview.heading('target_name', text = 'Target')
        self.alignment_data_treeview.heading('target_url', text = 'URL')
        self.alignment_data_treeview.heading('time', text = 'Time')
        self.alignment_data_treeview.heading('raw_axis_0', text = 'Raw Az / RA')
        self.alignment_data_treeview.heading('raw_axis_1', text = 'Raw Alt / Dec')
        self.alignment_data_treeview.tag_configure('disabled', foreground='grey')
        self.alignment_data_treeview.grid(row=0, column=0, sticky='nsew')

        self.alignment_data_treeview_scrollbar = ttk.Scrollbar(observations_treeview_frame, orient="vertical", command=self.alignment_data_treeview.yview)
        self.alignment_data_treeview_scrollbar.grid(row=0, column=1, sticky='ns')

        self.alignment_data_treeview.configure(yscrollcommand=self.alignment_data_treeview_scrollbar.set)

        self.update_gui()

    def add_observation(self):
        if self.tracker.primary_telescope_connection is None:
            print("You need to connect to a telescope so we can tell where it's pointing.")
            return
        
        estimated_current_axis_angles, current_time = self.tracker.primary_telescope_connection.get_estimated_axis_angles_and_time()
        new_datum = AlignmentDatum(None, current_time, estimated_current_axis_angles)
        print(repr(new_datum))

        self.observation_items.append(AlignmentDatumTreeviewItem(new_datum))

        self.autosave_observations()
        self.update_gui()

    def add_test_observations(self):
        # these were captured while connected to Stellarium, but they differ in time similarly to how
        # they would be if from a real telescope.
        test_alignments = []
        
        # Spica (HIP 65474)
        # astrolock.model.alignment.AlignmentDatum(None, <Time object: scale='utc' format='datetime' value=2023-03-06 09:01:44.365060>, <Quantity [2.61243707, 0.62959202] rad>)
        test_alignments.append(AlignmentDatum(
            None,
            astropy.time.Time("2023-03-06 09:01:44.365060"),
            [2.61243707, 0.62959202] * u.rad
        ))

        # Algorab (HIP 60965)
        #astrolock.model.alignment.AlignmentDatum(None, <Time object: scale='utc' format='datetime' value=2023-03-06 09:03:04.564831>, <Quantity [2.92833387, 0.61100651] rad>)
        test_alignments.append(AlignmentDatum(
            None,
            astropy.time.Time("2023-03-06 09:03:04.564831"),
            [2.92833387, 0.61100651] * u.rad
        ))

        if True:
            # Arcturus (HIP 69673)
            # probably was az. 109d15m34.6s alt 52d46m55.9
            #astrolock.model.alignment.AlignmentDatum(None, <Time object: scale='utc' format='datetime' value=2023-03-06 09:04:56.716076>, <Quantity [1.92051186, 0.93189984] rad>)
            test_alignments.append(AlignmentDatum(
                None,
                astropy.time.Time("2023-03-06 09:04:56.716076"),
                [1.92051186, 0.93189984] * u.rad
            ))

        #  the moon
        # astrolock.model.alignment.AlignmentDatum(None, <Time object: scale='utc' format='datetime' value=2023-07-08 06:51:50.532457>, <Quantity [-2.26993803,  1.01434353] rad>)
        # but really
        # astrolock.model.alignment.AlignmentDatum(None, <Time object: scale='utc' format='datetime' value=2023-03-06 09:16:56.716076>, <Quantity [-2.26993803,  1.01434353] rad>)
        test_alignments.append(AlignmentDatum(
                None,
                astropy.time.Time("2023-03-06 09:16:56.716076"),
                [-2.26993803,  1.01434353] * u.rad
            ))
        
        # mars, 3 hours before:
        # astrolock.model.alignment.AlignmentDatum(None, <Time object: scale='utc' format='datetime' value=2023-07-08 06:55:51.437480>, <Quantity [-1.53162858,  0.73345739] rad>)
        # but really:
        # astrolock.model.alignment.AlignmentDatum(None, <Time object: scale='utc' format='datetime' value=2023-03-06 06:16:56.716076>, <Quantity [-1.53162858,  0.73345739] rad>)
        test_alignments.append(AlignmentDatum(
                None,
                astropy.time.Time("2023-03-06 06:16:56.716076"),
                [-1.53162858,  0.73345739] * u.rad
            ))

        for alignment_datum in test_alignments:
            self.observation_items.append(AlignmentDatumTreeviewItem(alignment_datum))
        self.update_gui()        

    def add_random_stepper_offsets(self):
        test_stepper_offsets = [random.random() * 2.0 * math.pi, random.random() * 2.0 * math.pi] * u.rad
        print(f"Fuzzing offsets by {test_stepper_offsets}")
        for item in self.observation_items:
            item.datum.raw_axis_values += test_stepper_offsets
        self.update_gui()

    def update_gui(self):
        self.current_alignment_label.config(text = str(self.tracker.primary_telescope_alignment))

        # it's insane that this is the best way... seems O(n^2)
        self.alignment_data_treeview.delete(*self.alignment_data_treeview.get_children())

        for item in self.observation_items:
            alignment_datum = item.datum

            if alignment_datum.target is not None:
                target_name = alignment_datum.target.display_name
                target_url = alignment_datum.target.url
            else:
                target_name = '<unknown>'
                target_url = ''

            #('target_name', 'target_url', 'time', 'raw_axis_0', 'raw_axis_1')
            values = (target_name, target_url, str(alignment_datum.time), alignment_datum.raw_axis_values[0], alignment_datum.raw_axis_values[1])

            tags = []
            if not item.enabled:
                tags.append('disabled')

            item.iid = self.alignment_data_treeview.insert(parent = '', index = 'end', iid = item.iid, values = values, tags=tags)

    def get_alignment_data_from_gui(self):
        alignment_data = []
        for item in self.observation_items:
            if item.enabled:
                alignment_data.append(item.datum)
        return alignment_data

    def align(self):
        targets = []
        for target_source in self.tracker.target_source_map.values():
            if target_source.use_for_alignment:
                target_source.start()
                target_map = target_source.get_target_map()
                for target in target_map.values():
                    targets.append(target)
        
        if len(targets) == 0:
            print("Couldn't find any targets to align to.")
            return
        
        print(f"Aligning with { len(targets) } targets.")
        if False:
            for target in targets:
                print(f'\n\t{target.display_name}')            

        alignment_data = self.get_alignment_data_from_gui()

        if len(alignment_data) == 0:
            print("Can't align without any alignment data.")
            return

        alignment = astrolock.model.alignment.align(self.tracker, alignment_data, targets)

        self.tracker.primary_telescope_alignment = alignment

        self.update_gui()

    
    def load_observations_from_filename(self, filename):
        with open(filename, 'r') as f:
            self.observation_items.clear()
            alignmen_data_arr = json.load(f)
            for alignment_datum_dict in alignmen_data_arr:
                datum = astrolock.model.alignment.AlignmentDatum.from_json(alignment_datum_dict)
                item = AlignmentDatumTreeviewItem(datum)
                self.observation_items.append(item)
            self.update_gui()


    def save_observations_to_filename(self, filename):
        alignmen_data_arr = list(map(lambda d: d.to_json(), self.get_alignment_data_from_gui()))
        with open(filename, 'w') as f:
            json.dump(alignmen_data_arr, f)
            

    def autosave_observations(self):
        self.save_observations_to_filename(os.path.join('data', 'alignments', 'autosave.json'))

    def load_observations(self):
        filename = tkinter.filedialog.askopenfilename(defaultextension='json', filetypes=[('JSON','*.json')], initialdir=os.path.join('data', 'alignments'))
        if filename is not None:
            self.load_observations_from_filename(filename)

    def save_observations(self):
        filename = tkinter.filedialog.asksaveasfilename(defaultextension='json', filetypes=[('JSON','*.json')], initialdir=os.path.join('data', 'alignments'))
        if filename is not None:
            self.save_observations_to_filename(filename)

    def set_selection_enabled(self, enabled):
        for iid in self.alignment_data_treeview.selection():
            for item in self.observation_items:
                if item.iid == iid:
                    item.enabled = enabled                    
        self.update_gui()


    def enable_selected_observations(self):
        self.set_selection_enabled(True)


    def disable_selected_observations(self):
        self.set_selection_enabled(False)


    def delete_selected_observations(self):        
        for iid in self.alignment_data_treeview.selection():
            self.observation_items = list(filter(lambda i: i.iid != iid, self.observation_items))
                    
        self.update_gui()


    def delete_all_observations(self):
        self.observation_items.clear()
        self.update_gui()