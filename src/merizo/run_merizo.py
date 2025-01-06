from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from torch.types import Device

from src.merizo.model.network import Merizo
from src.merizo.model.utils.features import generate_features_domain
from src.merizo.model.utils.utils import (
    instance_matrix,
    clean_domains,
    clean_singletons,
    get_ids,
    remap_ids,
    shuffle_ids,
    separate_components, get_device,
)
from src.shared.structure import get_model_structure

MIN_DOMAIN_SIZE = 50
MIN_FRAGMENT_SIZE = 10
DOM_AVE = 200
CONF_THRESHOLD = 0.75


def iterative_segmentation(
        network: torch.nn.Module, features: dict, max_iterations: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Iteratively segment protein domains with a trained network.
    """
    n_iterations, ignore_index, iterate = 0, [], True
    domain_ids, conf_res = features['domain_ids'], features['conf_res']

    while iterate and n_iterations < max_iterations:
        ids, _ = get_ids(domain_ids)
        # Identify domains that are large enough to be re-segmented
        unique_ids = {
            d.item(): conf_res[domain_ids == d].mean()
            for d in ids
            if d.item() not in ignore_index and (domain_ids == d).sum() > DOM_AVE
        }

        if not unique_ids:
            break

        # Re-segment each domain that meets criteria
        counter = 1
        for old_did, _ in unique_ids.items():
            mask = domain_ids == old_did
            new_ids, new_conf = network(features, mask=mask)
            _, ndoms_ = get_ids(new_ids)

            if ndoms_ == 1:  # If it stayed one domain, ignore future re-segmentation
                ignore_index.append(old_did)
            else:
                # Offset new IDs to avoid collisions
                offset_ids = new_ids + (counter * network.no_classes)
                offset_ids[new_ids == 0] = 0
                domain_ids[mask] = offset_ids
                conf_res[mask] = new_conf
                counter += 1

        n_iterations += 1

    return domain_ids, conf_res


def read_split_weight_files(directory: str) -> dict:
    """
    Read .pt weight files in a directory into a single dictionary.
    """
    weights = {}
    for f in os.listdir(directory):
        if f.endswith('.pt'):
            weights.update(torch.load(os.path.join(directory, f), weights_only=True))

    return weights


def segment(
        pdb_path: str,
        network: torch.nn.Module,
        device: Device,
        length_conditional_iterate: bool,
        iterate: bool,
        max_iterations: int,
        shuffle_indices: bool,
        pdb_chain: str = "A",
) -> dict:
    """
    Segment domains in a protein structure.
    """
    features = generate_features_domain(pdb_path, device, pdb_chain)
    if length_conditional_iterate and features['nres'] > 512:
        iterate = True

    features['domain_ids'], features['conf_res'] = network(features)

    # Optionally iterate multiple times if large
    if iterate and features['nres'] > DOM_AVE * 2:
        features['domain_ids'], features['conf_res'] = iterative_segmentation(
            network, features, max_iterations
        )

    features['domain_map'] = instance_matrix(features['domain_ids'])[0]
    features['domain_ids'] = separate_components(features)

    # Clean up small domains or singletons
    if torch.unique(features['domain_ids']).numel() > 1:
        features['domain_ids'] = clean_domains(features['domain_ids'], MIN_DOMAIN_SIZE)
        features['domain_ids'] = clean_singletons(features['domain_ids'], MIN_FRAGMENT_SIZE)

    features['domain_map'] = instance_matrix(features['domain_ids'])[0]
    features['conf_global'] = features['conf_res'].mean()
    features['ndom'] = get_ids(features['domain_ids'])[1]
    features['domain_ids'] = (
        shuffle_ids(features['domain_ids']) if shuffle_indices else remap_ids(features['domain_ids'])
    )

    return features


def features_to_chopping_string(features) -> str:
    # Convert domain IDs into consecutive-residue group strings
    domain_ids = np.array(features["domain_ids"])
    outputs = []
    for d in np.unique(domain_ids):
        matching = np.where(domain_ids == d)[0]
        # Find consecutive stretches
        groups, curr = [], []
        for idx in matching:
            if not curr or idx == curr[-1] + 1:
                curr.append(idx)
            else:
                groups.append(curr)
                curr = [idx]
        if curr:
            groups.append(curr)
        # Format each group into "start-end" or single residue
        chunked = [f"{g[0] + 1}-{g[-1] + 1}" if len(g) > 1 else f"{g[0] + 1}" for g in groups]
        outputs.append("_".join(chunked))
    output = ",".join(outputs)
    return output


def predict_chopping_merizo_from_pdb(
        input_path: str,
        device: str = 'cpu',
        max_iterations: int = 3,
        length_conditional_iterate: bool = False,
        iterate: bool = False,
        shuffle_indices: bool = False,
        pdb_chain: str = None,
) -> str:
    """
    Run Merizo on a PDB file and return the chopping string.
    """
    device = get_device(device)
    network = Merizo().to(device)

    weights_dir = Path(__file__).parent.parent / "models" / "merizo"
    network.load_state_dict(read_split_weight_files(str(weights_dir)), strict=True)
    network.eval()

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"{input_path} does not exist.")

    with torch.no_grad():
        _, _ = os.path.splitext(input_path)

        # If no chain is provided, pick the first chain from the structure
        if not pdb_chain:
            structure = get_model_structure(input_path)
            chains = [c.id for c in structure.get_chains()]
            if not chains:
                raise ValueError(f"No chains found in {input_path}")
            pdb_chain = chains[0]

        features = segment(
            pdb_path=input_path,
            network=network,
            device=device,
            length_conditional_iterate=length_conditional_iterate,
            iterate=iterate,
            max_iterations=max_iterations,
            shuffle_indices=shuffle_indices,
            pdb_chain=pdb_chain,
        )

        return features_to_chopping_string(features)
