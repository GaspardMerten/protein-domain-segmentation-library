# Protein Domain Segmentation Library

This library provides a convenient interface for protein domain segmentation using the powerful tools [Merizo](https://github.com/psipred/Merizo) and [Chainsaw](https://github.com/JudeWells/chainsaw). The library abstracts the complexity of using these tools, allowing users to predict protein domain boundaries directly from PDB files or MDAnalysis universes.

## Features

- **Unified Interface**: Use a consistent API to access both Merizo and Chainsaw functionalities.
- **Flexible Input Support**: Predict from PDB files or MDAnalysis universes.
- **Lightweight Usage**: Avoid integrating full tool codebases directly into your project.

## Installation

Install the library and its dependencies using pip:

```bash
pip install protein-domain-segmentation
```

## Usage

Below is an example demonstrating how to use the `MerizoCluster` and `ChainsawCluster` classes:

```python
from protein_domain_segmentation import MerizoCluster, ChainsawCluster
import MDAnalysis as mda

# Initialize clusters
merizo_cluster = MerizoCluster()
chainsaw_cluster = ChainsawCluster()

# Example PDB file path
pdb_path = "example.pdb"

# Predict chopping using Merizo
merizo_result = merizo_cluster.predict_from_pdb(pdb_path)
print("Merizo Results:", merizo_result)

# Predict chopping using Chainsaw
chainsaw_result = chainsaw_cluster.predict_from_pdb(pdb_path)
print("Chainsaw Results:", chainsaw_result)

# Predict chopping from an MDAnalysis Universe
u = mda.Universe(pdb_path)
merizo_from_universe = merizo_cluster.predict_from_universe(u)
print("Merizo Results from Universe:", merizo_from_universe)
```

## API Overview

### Classes

- **`Cluster`**: Abstract base class for clustering tools.
  - **`predict_from_pdb(pdb_path: str, model_params: Dict = None)`**:
    Predicts the chopping of a protein from a PDB file.
  - **`predict_from_universe(universe: mda.Universe, model_params: Dict = None)`**:
    Predicts the chopping of a protein from an MDAnalysis Universe.

- **`MerizoCluster`**:
  Implementation of `Cluster` for Merizo.

- **`ChainsawCluster`**:
  Implementation of `Cluster` for Chainsaw.

### Methods

- **`MerizoCluster.predict_from_pdb(pdb_path: str, model_params: Dict = None)`**:
  Uses Merizo to predict chopping from a PDB file.
  
- **`ChainsawCluster.predict_from_pdb(pdb_path: str, model_params: Dict = None)`**:
  Uses Chainsaw to predict chopping from a PDB file.

## Credits

This package wraps the functionality of the following tools:

- **Merizo**: Developed by the UCL Bioinformatics Group. Visit the [Merizo GitHub repository](https://github.com/psipred/Merizo).
- **Chainsaw**: Developed by Jude Wells and collaborators. Visit the [Chainsaw GitHub repository](https://github.com/JudeWells/chainsaw).

### Citation

If you use this package in your research, please cite the respective papers for Merizo and Chainsaw:

- **Merizo**: Lau et al., 2023. Merizo: a rapid and accurate protein domain segmentation method using invariant point attention. *Nature Communications*.
- **Chainsaw**: Wells et al., Chainsaw: protein domain segmentation with fully convolutional neural networks. *Bioinformatics*.

## License

This package is distributed under the MIT License. See the LICENSE file for details.

## Acknowledgments

We extend our gratitude to the developers of Merizo and Chainsaw for their invaluable contributions to the field of protein domain segmentation.