import os

import Bio.PDB
import Bio.PDB.Structure


def get_model_structure(structure_path) -> Bio.PDB.Structure:
    """
    Returns the Bio.PDB.Structure object for a given PDB or MMCIF file
    """
    structure_id = os.path.split(structure_path)[-1].split('.')[0]
    if structure_path.endswith('.pdb'):
        structure = Bio.PDB.PDBParser().get_structure(structure_id, structure_path)
    elif structure_path.endswith('.cif'):
        structure = Bio.PDB.MMCIFParser().get_structure(structure_id, structure_path)
    else:
        raise ValueError(f'Unrecognized file extension: {structure_path}')
    model = structure[0]
    return model
