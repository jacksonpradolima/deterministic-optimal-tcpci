import os
import pickle
import time
from pathlib import Path

from deterministic_optimal_tcpci.utils.monitor import MonitorCollector

Path("backup").mkdir(parents=True, exist_ok=True)


class Environment(object):
    """
    The environment class run the simulation.
    """

    def __init__(self, scenario_provider, evaluation_metric):
        self.scenario_provider = scenario_provider
        self.evaluation_metric = evaluation_metric
        self.is_industrial_dataset = self.scenario_provider.is_industrial_dataset
        self.reset()

    def reset(self):
        """
        Reset all variables (for a new simulation)
        :return:
        """
        # Monitor saves the feedback during the process
        self.monitor = MonitorCollector()

    def run_single(self, experiment, trials=100, restore=True):
        """
        Execute a simulation
        :param experiment: Current Experiment
        :param trials: The max number of scenarios that will be analyzed
        :param restore: restore the experiment if fail (i.e., energy down)
        :return:
        """
        # create a bandit (initially is None to know that is the first bandit
        bandit = None

        # restore to step
        r_t = 1

        if restore:
            r_t, self.monitor, bandit = self.load_experiment(experiment)
            self.scenario_provider.last_build(r_t)
            r_t += 1  # start 1 step after the last build

        sort_columns = ['Verdict', 'Duration'] if self.is_industrial_dataset else ['NumErrors', 'Duration']

        # For each "virtual scenario (vsc)" I must analyse it and evaluate it
        for (t, vsc) in enumerate(self.scenario_provider, start=r_t):
            # The max number of scenarios that will be analyzed
            if t > trials:
                break

            self.evaluation_metric.update_available_time(vsc.get_available_time())

            # Compute time
            start = time.time()

            df = vsc.get_testcases()
            df.sort_values(sort_columns, ascending=[False, True], inplace=True)

            end = time.time()

            self.evaluation_metric.evaluate(df.to_dict(orient='record'))

            print(f"Exp {experiment} - Ep {t} - Deterministic ", end="")

            print(f" - NAPFD/APFDc: {self.evaluation_metric.fitness:.4f}/{self.evaluation_metric.cost:.4f}")

            self.monitor.collect(self.scenario_provider, vsc.get_available_time(), experiment, t, "Deterministic",
                                 "-", self.evaluation_metric, (end - start), -1)

            # Save experiment each 50000 builds
            if restore and t % 50000 == 0:
                self.save_experiment(experiment, t, bandit)

    def run(self, experiments=1, trials=100, restore=True):
        """
        Execute a simulation
        :param experiments: Number of experiments
        :param trials: The max number of scenarios that will be analyzed
        :param print_log:
        :param bandit_type:
        :param restore: restore the experiment if fail (i.e., energy down)
        :return:
        """
        self.reset()

        for exp in range(experiments):
            self.run_single(exp, trials, restore)

    def store_experiment(self, name):
        self.monitor.save(name)

    def load_experiment(self, experiment):
        filename = f'backup/{str(self.scenario_provider)}_ex_{experiment}.p'

        if os.path.exists(filename):
            return pickle.load(open(filename, "rb"))

        return 0, self.monitor, None

    def save_experiment(self, experiment, t, bandit):
        filename = f'backup/{str(self.scenario_provider)}_ex_{experiment}.p'
        pickle.dump([t, self.monitor, bandit], open(filename, "wb"))
