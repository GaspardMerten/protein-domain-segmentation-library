import unittest
from pathlib import Path

from src.protein_domain_segmentation import MerizoCluster

TEST_FOLDER = Path(__file__).parent


class TestProcessInput(unittest.TestCase):
    def test_merizo(self):
        expected_output = "1-71,72-143"

        actual_output = MerizoCluster().predict_from_pdb(
            str((TEST_FOLDER / "prot.pdb"))
        )

        self.assertEqual(
            actual_output, expected_output, "Merizo output does not match expected"
        )
