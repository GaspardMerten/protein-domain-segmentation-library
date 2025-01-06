import unittest

from src import MerizoCluster


class TestProcessInput(unittest.TestCase):
    def test_merizo(self):
        expected_output = "1-71,72-143"

        actual_output = MerizoCluster().predict_from_pdb('prot.pdb')

        self.assertEqual(actual_output, expected_output, "Merizo output does not match expected")
