from typing import Union, Any

from abrain import Genome as CPPNGenome
from networkx import DiGraph

from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import HighProbabilityDecoder
from config import CommonConfig
from genotype import Genotype


def decode(genotype: Genotype, config: CommonConfig) -> Union[DiGraph, Any]:
    body = decode_body(genotype.body, config)
    return body, decode_brain(genotype.brain, body, config)


def decode_body(genotype: Genotype.Body, config: CommonConfig) -> DiGraph:
    # System parameters
    num_modules = config.max_modules

    (type_probability_space,
     conn_probability_space,
     rotation_probability_space) = config.nde_decoder.forward(genotype.data)

    # Decode the high-probability graph
    hpd = HighProbabilityDecoder(num_modules)
    return hpd.probability_matrices_to_graph(
        type_probability_space,
        conn_probability_space,
        rotation_probability_space,
    )


def decode_brain(genotype: CPPNGenome, body: DiGraph, config: CommonConfig) -> Any:
    print("[kgd-debug] decode_brain")
    print(body)
    print(body.nodes(data=True))

    class Placeholder:
        pass
    return Placeholder()
