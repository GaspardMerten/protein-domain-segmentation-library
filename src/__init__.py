import abc
import tempfile
from typing import Dict

import MDAnalysis as mda

from src.chainsaw.run_chainsaw import predict_chopping_chainsaw_from_pdb
from src.merizo.run_merizo import predict_chopping_merizo_from_pdb


class Cluster(abc.ABC):
    @abc.abstractmethod
    def predict_from_pdb(self, pdb_path: str, model_params: Dict = None):
        """
        Predicts the chopping of a protein from a PDB file
        """
        raise NotImplementedError

    def predict_from_universe(self, universe: mda.Universe, model_params: Dict = None):
        """
        Predicts the chopping of a protein from a MDAnalysis Universe
        """
        with tempfile.TemporaryFile() as temp_pdb:
            universe.atoms.write(temp_pdb, format='pdb')
            return self.predict_from_pdb(temp_pdb.name, model_params)


class MerizoCluster(Cluster):
    def predict_from_pdb(self, pdb_path: str, model_params: Dict = None):
        """
        Predicts the chopping of a protein from a PDB file
        """
        return predict_chopping_merizo_from_pdb(pdb_path, **(model_params or {}))


class ChainsawCluster(Cluster):
    def predict_from_pdb(self, pdb_path: str, model_params: Dict = None):
        """
        Predicts the chopping of a protein from a PDB file
        """
        return predict_chopping_chainsaw_from_pdb(pdb_path, **(model_params or {}))
