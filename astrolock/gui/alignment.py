import math
import random
import tkinter as tk
import tkinter.ttk as ttk

import astrolock.model.alignment
from astrolock.model.alignment import AlignmentDatum

import astropy
from astropy import units as u

class AlignmentFrame(tk.Frame):
    def __init__(self, *args, tracker, **kwargs):        
        tk.Frame.__init__(self, *args, **kwargs)

        self.tracker = tracker

        label = tk.Label(self, text="Alignment")
        label.pack()

        add_alignment_star_button = tk.Button(self, text = "Add Star", command = self.add_alignment_star)   
        add_alignment_star_button.pack()

        add_test_alignment_button = tk.Button(self, text = "Add Test Stars", command = self.add_test_alignments)   
        add_test_alignment_button.pack()

        align_button =  tk.Button(self, text = "align", command = self.align)
        align_button.pack()

        self.alignment_data = []

        self.alignment_data_treeview = ttk.Treeview(self, show = 'headings')
        self.alignment_data_treeview['columns'] = ('target_name', 'target_url', 'time', 'raw_axis_0', 'raw_axis_1')
        self.alignment_data_treeview.heading('target_name', text = 'Target')
        self.alignment_data_treeview.heading('target_url', text = 'URL')
        self.alignment_data_treeview.heading('time', text = 'Time')
        self.alignment_data_treeview.heading('raw_axis_0', text = 'Axis 0')
        self.alignment_data_treeview.heading('raw_axis_1', text = 'Axis 1')        
        self.alignment_data_treeview.pack(side = 'left', fill = 'both', expand = True)
        self.alignment_data_treeview_scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.alignment_data_treeview.yview)
        self.alignment_data_treeview_scrollbar.pack(side='right', fill='y')
        # TODO: hmm, what was this for?  it errors...
        #self.alignment_data_treeview.configure(yscrollcommand=self.alignment_data_treeview.set)


    def add_alignment_star(self):
        if self.tracker.primary_telescope_connection is None:
            print("You need to connect to a telescope so we can tell where it's pointing.")
            return
        
        estimated_current_axis_angles, current_time = self.tracker.primary_telescope_connection.get_estimated_axis_angles_and_time()
        new_datum = AlignmentDatum(None, current_time, estimated_current_axis_angles)
        print(repr(new_datum))
        self.update_gui()

    def add_test_alignments(self):
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

        fuzz_offsets = True
        if fuzz_offsets:
            test_stepper_offsets = [random.random() * 2.0 * math.pi, random.random() * 2.0 * math.pi] * u.rad
            print(f"Fuzzing offsets by {test_stepper_offsets}")
            for alignment in test_alignments:
                alignment.raw_axis_values += test_stepper_offsets
                alignment.ground_truth_stepper_offset = test_stepper_offsets

        self.alignment_data = test_alignments
        self.update_gui()
        

    def update_gui(self):
        # it's insane that this is the best way... seems O(n^2)
        self.alignment_data_treeview.delete(*self.alignment_data_treeview.get_children())

    
        for alignment_datum in self.alignment_data:
            
            if alignment_datum.target is not None:
                target_name = alignment_datum.target.display_name
                target_url = alignment_datum.target.url
            else:
                target_name = '<unknown>'
                target_url = ''

            #('target_name', 'target_url', 'time', 'raw_axis_0', 'raw_axis_1')
            values = (target_name, target_url, str(alignment_datum.time), alignment_datum.raw_axis_values[0], alignment_datum.raw_axis_values[1])
            self.alignment_data_treeview.insert(parent = '', index = 'end', iid = None, values = values)

    def align(self):
        if len(self.alignment_data) == 0:
            print("Can't align without any alignment data.")
            return

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

        astrolock.model.alignment.align(self.tracker, self.alignment_data, targets)

        self.update_gui()
