import os
import pickle
import time
from pathlib import Path

from deterministic_optimal_tcpci.scenarios import IndustrialDatasetHCSScenarioProvider, VirtualHCSScenario
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

        self.variant_montitors = {}

        if isinstance(self.scenario_provider, IndustrialDatasetHCSScenarioProvider):
            if self.scenario_provider.get_total_variants() > 0:
                for variant in self.scenario_provider.get_all_variants():
                    self.variant_montitors[variant] = MonitorCollector()

    def run_single(self, experiment, trials=100, restore=True):
        """
        Execute a simulation
        :param experiment: Current Experiment
        :param trials: The max number of scenarios that will be analyzed
        :param restore: restore the experiment if fail (i.e., energy down)
        :return:
        """
        # restore to step
        r_t = 1

        if restore:
            r_t, self.monitor = self.load_experiment(experiment)
            self.scenario_provider.last_build(r_t)
            r_t += 1  # start 1 step after the last build

        sort_columns = ['Verdict', 'Duration'] if self.is_industrial_dataset else [
            'NumErrors', 'Duration']

        avail_time_ratio = self.scenario_provider.get_avail_time_ratio()

        # For each "virtual scenario (vsc)" I must analyse it and evaluate it
        for (t, vsc) in enumerate(self.scenario_provider, start=r_t):
            # The max number of scenarios that will be analyzed
            if t > trials:
                break

            self.evaluation_metric.update_available_time(
                vsc.get_available_time())

            # Compute time
            start = time.time()

            df_main = vsc.get_testcases()
            df_main.sort_values(sort_columns, ascending=[
                                False, True], inplace=True)
            action = df_main['Name'].tolist()  # current test cases

            end = time.time()

            self.evaluation_metric.evaluate(df_main.to_dict(orient='record'))

            print(f"Exp {experiment} - Ep {t} - Deterministic " +
                  f" - NAPFD/APFDc: {self.evaluation_metric.fitness:.4f}/{self.evaluation_metric.cost:.4f}")

            self.monitor.collect(self.scenario_provider,
                                 vsc.get_available_time(),
                                 experiment,
                                 t,
                                 "Deterministic",
                                 "-",
                                 self.evaluation_metric,
                                 self.scenario_provider.total_build_duration,
                                 (end - start),
                                 -1,
                                 action)

            # If we are working with HCS scenario and there are variants?
            if type(vsc) == VirtualHCSScenario and len(vsc.get_variants()) > 0:
                # Get the variants that exist in the current commit
                variants = vsc.get_variants()

                # For each variant I will evaluate the impact of the main
                # prioritization
                for variant in variants['Variant'].unique():
                    # Get the data from current variant
                    df_variant = variants[variants.Variant == variant]

                    # Order by the test cases according to the main
                    # prioritization
                    df_variant['CalcPrio'] = df_variant[
                        'Name'].apply(lambda x: action.index(x) + 1)
                    df_variant.sort_values(by=['CalcPrio'], inplace=True)

                    total_build_duration = df_variant['Duration'].sum()
                    total_time = total_build_duration * avail_time_ratio

                    # Update the available time concerning the variant build
                    # duration
                    self.evaluation_metric.update_available_time(total_time)

                    # Submit prioritized test cases for evaluation step and get
                    # new measurements
                    self.evaluation_metric.evaluate(
                        df_variant.to_dict(orient='record'))

                    # Save the information
                    self.variant_montitors[variant].collect(self.scenario_provider,
                                                            total_time,
                                                            experiment,
                                                            t,
                                                            "Deterministic",
                                                            "-",
                                                            self.evaluation_metric,
                                                            total_build_duration,
                                                            (end - start),
                                                            -1,
                                                            df_variant['Name'].tolist())

            # Save experiment each 50000 builds
            if restore and t % 50000 == 0:
                self.save_experiment(experiment, t)

    def run(self, experiments=1, trials=100, restore=True):
        """
        Execute a simulation
        :param experiments: Number of experiments
        :param trials: The max number of scenarios that will be analyzed
        :param print_log:
        :param restore: restore the experiment if fail (i.e., energy down)
        :return:
        """
        self.reset()

        for exp in range(experiments):
            self.run_single(exp, trials, restore)

    def create_file(self, name):
        self.monitor.create_file(name)

        if isinstance(self.scenario_provider, IndustrialDatasetHCSScenarioProvider):
            if self.scenario_provider.get_total_variants() > 0:
                # Ignore the extension
                name = name.split(".csv")[0]
                name = f"{name}_variants"

                Path(name).mkdir(parents=True, exist_ok=True)

                for variant in self.scenario_provider.get_all_variants():
                    self.variant_montitors[variant].create_file(
                        f"{name}/{name.split('/')[-1].split('@')[0]}@{variant.replace('/', '-')}.csv")

    def store_experiment(self, name):
        self.monitor.save(name)

        if isinstance(self.scenario_provider, IndustrialDatasetHCSScenarioProvider):
            if self.scenario_provider.get_total_variants() > 0:
                # Ignore the extension
                name2 = name.split(".csv")[0]
                name2 = f"{name2}_variants"

                Path(name2).mkdir(parents=True, exist_ok=True)

                for variant in self.scenario_provider.get_all_variants():
                    self.variant_montitors[variant].save(
                        f"{name2}/{name.split('/')[-1].split('@')[0]}@{variant.replace('/', '-')}.csv")

    def load_experiment(self, experiment):
        filename = f'backup/{str(self.scenario_provider)}_ex_{experiment}.p'

        if os.path.exists(filename):
            return pickle.load(open(filename, "rb"))

        return 0, self.monitor

    def save_experiment(self, experiment, t):
        filename = f'backup/{str(self.scenario_provider)}_ex_{experiment}.p'
        pickle.dump([t, self.monitor], open(filename, "wb"))
