import unittest

from src import ChainsawCluster


class TestChainsaw(unittest.TestCase):

    def test_chainsaw(self):
        expected_output = "3-71,77-143"
        actual_output = ChainsawCluster().predict_from_pdb('prot.pdb')
        self.assertEqual(actual_output, expected_output, "Chainsaw output does not match expected")
