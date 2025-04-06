import unittest
from pathlib import Path

from protein_domain_segmentation import ChainsawCluster

TEST_FOLDER = Path(__file__).parent


class TestChainsaw(unittest.TestCase):

    def test_chainsaw(self):
        expected_output = "3-71,77-143"
        actual_output = ChainsawCluster().predict_from_pdb(
            str((TEST_FOLDER / "prot.pdb"))
        )
        self.assertEqual(
            actual_output, expected_output, "Chainsaw output does not match expected"
        )
