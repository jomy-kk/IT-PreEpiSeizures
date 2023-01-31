import unittest
from datetime import datetime

from ltbio.biosignals.modalities import ACC, ECG
from ltbio.biosignals.modalities import RESP
from ltbio.biosignals.sources import Sense
from ltbio.biosignals.timeseries.Timeseries import Timeseries


class SenseTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data_path = 'resources/Sense_CSV_tests/'  # This is a test directory with CSV files in the Sense structure,
        cls.defaults_path = 'resources/Sense_CSV_tests/sense_defaults.json'  # Path to default mappings
        cls.device_id = 'run_chest'  # Device id corresponding to the mapping to be used
        cls.Sense = Sense(cls.device_id, cls.defaults_path)
        cls.initial = datetime(2022, 6, 20, 19, 18, 57, 426000)

    def verify_data(self, x, label, sf, n_samples, unit, first_sample):
        self.assertTrue(isinstance(x, tuple))
        self.assertEqual(x[1], 'Chest')
        x = x[0]
        self.assertEqual(len(x), len(label))
        self.assertTrue(isinstance(list(x.keys())[0], str))
        self.assertEqual(tuple(x.keys()), label)
        for i, l in enumerate(label):
            self.assertTrue(isinstance(x[l], Timeseries))
            # And all these properties should match:
            self.assertEqual(x[l].sampling_frequency, sf)
            self.assertEqual(len(x[l]), n_samples)
            self.assertEqual(x[l].units, unit)
            # Also, checking the second sample
            self.assertEqual(float((x[l])[self.initial]), float(first_sample[i]))

    def test_read_ECG(self):
        x = self.Sense._timeseries(self.data_path, ECG)
        self.verify_data(x, ('Gel', 'Band' ), 1000.0, 899000, None, (1904.0, 1708.0))

    def test_read_RESP(self):
        x = self.Sense._timeseries(self.data_path, RESP)
        self.verify_data(x, ('Resp Band', ), 1000.0, 899000, None, (2214.0, ))

    def test_read_ACC(self):
        x = self.Sense._timeseries(self.data_path, ACC)
        self.verify_data(x, ('x', 'y', 'z'), 1000.0, 899000, None, (1392., 2322., 1821.))


if __name__ == '__main__':
    unittest.main()
