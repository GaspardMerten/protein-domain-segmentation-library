import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict

from src.chainsaw import featurisers
from src.chainsaw.domain_assignment.util import convert_domain_dict_strings
from src.chainsaw.factories import pairwise_predictor
from src.chainsaw.utils import common as common_utils
from src.shared.structure import get_model_structure

MODEL_DIR = Path(__file__).parent.parent / 'models' / 'chainsaw'
LOG = logging.getLogger(__name__)


def load_model(
        remove_disordered_domain_threshold: float = 0.35,
        min_ss_components: int = 2,
        min_domain_length: int = 30,
        post_process_domains: bool = True, ):
    config = common_utils.load_json(os.path.join(MODEL_DIR, "config.json"))
    feature_config = common_utils.load_json(os.path.join(MODEL_DIR, "feature_config.json"))
    config["learner"]["remove_disordered_domain_threshold"] = remove_disordered_domain_threshold
    config["learner"]["post_process_domains"] = post_process_domains
    config["learner"]["min_ss_components"] = min_ss_components
    config["learner"]["min_domain_length"] = min_domain_length
    config["learner"]["dist_transform_type"] = config["data"].get("dist_transform", 'min_replace_inverse')
    config["learner"]["distance_denominator"] = config["data"].get("distance_denominator", None)
    learner = pairwise_predictor(config["learner"], output_dir=MODEL_DIR)
    learner.feature_config = feature_config
    learner.load_checkpoints()
    learner.eval()

    return learner


@dataclass
class PredictionResult:
    """
    Holds the result of a single domain prediction.
    """
    domain_id: str
    chopping: str
    confidence: float


class Seg:
    """
    Represents a contiguous segment of residues (by index).
    """

    def __init__(self, domain_id: str, start_index: int, end_index: int, index_to_label_map: Dict[int, str]):
        self.domain_id = domain_id
        self.start_index = start_index
        self.end_index = end_index
        self._index_to_label_map = index_to_label_map

    def _res_label_of_index(self, index: int) -> str:
        try:
            return self._index_to_label_map[index]
        except KeyError:
            raise ValueError(
                f"Index {index} not found in residue label map "
                f"({sorted(self._index_to_label_map.keys())})."
            )

    @property
    def start_label(self) -> str:
        return self._res_label_of_index(self.start_index)

    @property
    def end_label(self) -> str:
        return self._res_label_of_index(self.end_index)


class Dom:
    """
    Represents a collection of one or more Segs (which may be discontinuous).
    """

    def __init__(self, domain_id: str):
        self.domain_id = domain_id
        self.segs: List[Seg] = []

    def add_seg(self, seg: Seg) -> None:
        self.segs.append(seg)


# -----------------------------------------------------


def _predict(model, pdb_path, renumber_pdbs=True, pdbchain=None) -> str | None:
    """
    Makes the prediction and returns a list of PredictionResult objects
    """

    # get model structure metadata
    model_structure = get_model_structure(pdb_path)

    if pdbchain is None:
        # get all the chain ids from the model structure
        all_chain_ids = [c.id for c in model_structure.get_chains()]
        # take the first chain id
        pdbchain = all_chain_ids[0]

    model_residues = featurisers.get_model_structure_residues(model_structure, chain=pdbchain)
    model_res_label_by_index = {int(r.index): str(r.res_label) for r in model_residues}

    x = featurisers.inference_time_create_features(pdb_path,
                                                   feature_config=model.feature_config,
                                                   chain=pdbchain,
                                                   renumber_pdbs=renumber_pdbs,
                                                   model_structure=model_structure,
                                                   )

    _, domain_dict, _ = model.predict(x)
    # Convert 0-indexed to 1-indexed to match AlphaFold indexing:
    domain_dict = [{k: [r + 1 for r in v] for k, v in d.items()} for d in domain_dict]
    names_str, bounds_str = convert_domain_dict_strings(domain_dict[0])

    if names_str == "":
        names = bounds = ()
    else:
        names = names_str.split('|')
        bounds = bounds_str.split('|')

    assert len(names) == len(bounds)

    class Seg:
        def __init__(self, domain_id: str, start_index: int, end_index: int):
            self.domain_id = domain_id
            self.start_index = int(start_index)
            self.end_index = int(end_index)

        def res_label_of_index(self, index: int):
            if index not in model_res_label_by_index:
                raise ValueError(f"Index {index} not in model_res_label_by_index ({model_res_label_by_index})")
            return model_res_label_by_index[int(index)]

        @property
        def start_label(self):
            return self.res_label_of_index(self.start_index)

        @property
        def end_label(self):
            return self.res_label_of_index(self.end_index)

    class Dom:
        def __init__(self, domain_id, segs=None):
            self.domain_id = domain_id
            if segs is None:
                segs = []
            self.segs = segs

        def add_seg(self, seg: Seg):
            self.segs.append(seg)

    # gather choppings into segments in domains
    domains_by_domain_id = {}
    for domain_id, chopping_by_index in zip(names, bounds):
        if domain_id not in domains_by_domain_id:
            domains_by_domain_id[domain_id] = Dom(domain_id)
        start_index, end_index = chopping_by_index.split('-')
        seg = Seg(domain_id, start_index, end_index)
        domains_by_domain_id[domain_id].add_seg(seg)

    # sort domain choppings by the start residue in first segment
    domains = sorted(domains_by_domain_id.values(), key=lambda dom: dom.segs[0].start_index)

    # collect domain choppings as strings
    domain_choppings = []
    for dom in domains:
        # convert segments to strings
        segs_str = [f"{seg.start_label}-{seg.end_label}" for seg in dom.segs]
        segs_index_str = [f"{seg.start_index}-{seg.end_index}" for seg in dom.segs]
        LOG.info(f"Segments (index to label): {segs_index_str} -> {segs_str}")
        # join discontinuous segs with '_'
        domain_choppings.append('_'.join(segs_str))

    # join domains with ','
    chopping_str = ','.join(domain_choppings)

    num_domains = len(domain_choppings)
    if num_domains == 0:
        chopping_str = None

    return chopping_str


def predict_chopping_chainsaw_from_pdb(
        structure_file,
        post_process_domains=True,
        remove_disordered_domain_threshold=0.35,
        min_domain_length=30,
        min_ss_components=2,
        use_first_chain=True,
):
    pdb_chain_id = None if use_first_chain else 'A'

    model = load_model(
        remove_disordered_domain_threshold=remove_disordered_domain_threshold,
        min_ss_components=min_ss_components,
        min_domain_length=min_domain_length,
        post_process_domains=post_process_domains,
    )

    return _predict(model, structure_file, pdbchain=pdb_chain_id)
