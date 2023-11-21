import unittest
import glob
import json
from .alignment import *
from .tracker import *
from astropy import units as u

class TestAlignment(unittest.TestCase):

    def test_alignment(self):
        tested = False

        tracker = Tracker()
        targets = []
        for target_source in tracker.target_source_map.values():
            if target_source.use_for_alignment:
                target_source.start()
                for target in target_source.target_map.values():
                    if target.use_for_alignment:
                        targets.append(target)

        self.assertGreater(len(targets), 0)

        filenames =  glob.glob("data/alignments/unit_tests/*.json")
        self.assertGreater(len(filenames), 0)
        for filename in filenames:
            with self.subTest(filename):
                expected_stars = filename.lower().split(".")[-2].split("__")[1].split("_")
                self.assertGreater(len(expected_stars), 0)

                alignment_data = self.load_alignment(filename)
                self.assertGreater(len(alignment_data), 0)

                alignment = align(tracker, alignment_data, targets)
                self.assertTrue(alignment.valid)

                for observation_index, alignment_datum in enumerate(alignment_data):                    

                    reconstructed_star_name = alignment_datum.reconstructed_target.display_name.split(" ")[0].lower()
                    expected_star_name = expected_stars[observation_index]
                    self.assertEqual(reconstructed_star_name, expected_star_name)

                    error_threshold = .5 * u.deg
                    self.assertLess(alignment_datum.angular_error.to(u.deg), error_threshold)

    def load_alignment(self, filename):
        data = []
        with open(filename, 'r') as f:
            alignment_data_arr = json.load(f)
            for alignment_datum_dict in alignment_data_arr:
                datum = AlignmentDatum.from_json(alignment_datum_dict)
                data.append(datum)
        return data


if __name__ == '__main__':
    unittest.main()