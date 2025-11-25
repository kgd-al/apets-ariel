from typing import Any

from abrain import Genome as CPPNGenome
from mujoco import MjModel, MjData
from networkx import DiGraph

from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import HighProbabilityDecoder
from .config import EvoConfig
from .misc.debug import kgd_debug
from .controllers.cpg import RevolveCPG
from ..miel.genotype import Genotype


def decode_body(genotype: Genotype.Body, config: EvoConfig) -> DiGraph:
    # System parameters
    num_modules = config.max_modules

    kgd_debug(f"nde decoder: {config.nde_decoder}")
    (type_probability_space,
     conn_probability_space,
     rotation_probability_space) = config.nde_decoder.forward(genotype.data)
    kgd_debug("ping")

    # Decode the high-probability graph
    hpd = HighProbabilityDecoder(num_modules)
    kgd_debug("ping")
    return hpd.probability_matrices_to_graph(
        type_probability_space,
        conn_probability_space,
        rotation_probability_space,
    )


def decode_brain(genotype: CPPNGenome,
                 model: MjModel, data: MjData, config: EvoConfig) -> Any:
    return RevolveCPG.from_cppn(genotype, model, data)

