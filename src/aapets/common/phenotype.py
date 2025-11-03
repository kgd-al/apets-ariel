from typing import Any

from abrain import Genome as CPPNGenome
from mujoco import MjModel, MjData
from networkx import DiGraph

from aapets.common.config import CommonConfig
from aapets.common.misc.debug import kgd_debug
from aapets.common.phenotypes.cpg import RevolveCPG
from aapets.miel.genotype import Genotype
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import HighProbabilityDecoder


def decode_body(genotype: Genotype.Body, config: CommonConfig) -> DiGraph:
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
                 model: MjModel, data: MjData, config: CommonConfig) -> Any:
    return RevolveCPG.from_cppn(genotype, model, data)

