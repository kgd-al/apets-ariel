from dataclasses import dataclass
import functools

import cma
import matplotlib
import numpy as np

from aapets.common.controllers.ABCpg import SymmetricalABCPG
from aapets.common.metrics_storage import EvaluationMetrics
from aapets.common.world_builder import compile_world
from aapets.g_cpg import evaluation
from aapets.g_cpg.config import Config
from aapets.g_cpg.types import Individual, fixed_morphology


class CMAWrap:
    def __init__(self, config: Config):
        self.config = config

        self.evaluator = evaluation.evaluator(config.task)

        robot = fixed_morphology(config.fixed_morphology)()
        state, *_ = compile_world(robot)
        self.body = robot.spec.to_xml()
        self.params = SymmetricalABCPG.num_parameters(state=state, name="")

        initial_mean = self.params * [0.5]
        initial_std = .5
        options = cma.CMAOptions()
        options.set("verb_filenameprefix", str(config.data_folder) + "/")
        options.set("seed", config.seed)
        options.set("tolfun", 0)
        options.set("tolflatfitness", 10)
        self.es = cma.CMAEvolutionStrategy(initial_mean, initial_std, options)

        self.evaluate_weights = functools.partial(self._evaluate,
            robot=self.body, evaluator=self.evaluator, config=self.config, return_metrics=False)

    def run(self, budget: int):
        self.es.optimize(self.evaluate_weights, maxfun=budget, n_jobs=self.config.threads, verb_disp=1)
        with open(self.config.data_folder.joinpath("cma-es.pkl"), "wb") as f:
            f.write(self.es.pickle_dumps())
        result = self.es.result_pretty()

        @dataclass
        class FakeDEAPFitness:
            values: list[float]

        return self._individual(weights=result.xbest, robot=self.body,
                                fitness=FakeDEAPFitness(values=[result.fbest]))

    def evaluate(self, ind: Individual, *args, **kwargs):
        return self.evaluate_weights(ind.weights, *args, **kwargs)

    @staticmethod
    def _individual(weights, robot: str, **kwargs):
        ind = Individual(
            genome=None,
            body=robot,
            weights=weights,
            brain_type=SymmetricalABCPG,
        )
        for k, v in kwargs.items():
            setattr(ind, k, v)
        return ind

    @staticmethod
    def _evaluate(weights: np.ndarray, robot: str, evaluator: evaluation.Evaluator, config: Config,
                  return_metrics):
        ind = CMAWrap._individual(weights, robot)
        state = evaluator.prepare(ind, config)
        evaluator.reset(state)
        result = evaluator.evaluate(state, ind.weights, config, return_metrics)

        if return_metrics:
            return result
        else:
            return -result.fitness

    def plot(self, return_metrics):
        folder = self.config.data_folder
        matplotlib.use("agg")
        cma.plot(str(folder) + "/", abscissa=1)
        # plt.tight_layout()
        cma.s.figsave(folder.joinpath('plot.png'), bbox_inches='tight')  # save current figure
        cma.s.figsave(folder.joinpath('plot.pdf'), bbox_inches='tight')  # save current figure

    def save(self, champion: Individual, metrics: EvaluationMetrics):
        return self.evaluator.save_robot(champion, metrics, self.config, data=None)
