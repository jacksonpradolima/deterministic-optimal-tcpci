import argparse
import logging
import multiprocessing
import os
import time
from pathlib import Path

from deterministic_optimal_tcpci.environment import Environment
from deterministic_optimal_tcpci.evaluation import NAPFDMetric, NAPFDVerdictMetric
from deterministic_optimal_tcpci.scenarios import IndustrialDatasetScenarioProvider, \
    IndustrialDatasetHCSScenarioProvider

DEFAULT_EXPERIMENT_DIR = 'results/optimal_deterministic'
DEFAULT_SCHED_TIME_RATIO = [0.1, 0.5, 0.8]
INDUSTRIAL_DATASETS = ['iofrol', 'paintcontrol',
                       'gsdtsr', 'lexisnexis',  'libssh@libssh-mirror']


def get_pool_size():
    return max(1, multiprocessing.cpu_count() - 1)


def run_experiments_with_threads(repo_path, dataset, output_dir, sched_time_ratio, evaluation_metric, considers_variants):
    # Get scenario
    if considers_variants:
        scenario = IndustrialDatasetHCSScenarioProvider(
            f"{repo_path}/{dataset}/features-engineered.csv",
            f"{repo_path}/{dataset}/data-variants.csv",
            sched_time_ratio)
    else:
        scenario = IndustrialDatasetScenarioProvider(
            f"{repo_path}/{dataset}/features-engineered.csv",
            sched_time_ratio)

    # Stop conditional
    trials = scenario.max_builds

    # Prepare the experiment
    env = Environment(scenario, evaluation_metric)

    # create a file with a unique header for the scenario (workaround)
    env.create_file(f"{output_dir}/{str(env.scenario_provider)}.csv")

    # Compute time
    start = time.time()

    env.run_single(1, trials)
    env.store_experiment(f"{output_dir}/{str(env.scenario_provider)}.csv")

    end = time.time()

    print(f"Time expend to run the experiments: {end - start}\n\n")


def get_metric(dataset, project_dir):
    return NAPFDVerdictMetric() if dataset in INDUSTRIAL_DATASETS or project_dir.split(
        os.sep)[-1] in INDUSTRIAL_DATASETS else NAPFDMetric()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    ap = argparse.ArgumentParser(description='Optimal')

    ap.add_argument('--project_dir', required=True)
    ap.add_argument('--considers_variants', default=False,
                    type=lambda x: (str(x).lower() == 'true'))

    ap.add_argument('--datasets', nargs='+', default=[], required=True,
                    help='Datasets to analyse. Ex: \'deeplearning4j@deeplearning4j\'')

    ap.add_argument('--sched_time_ratio', nargs='+',
                    default=DEFAULT_SCHED_TIME_RATIO, help='Schedule Time Ratio')
    ap.add_argument('-o', '--output_dir', default=DEFAULT_EXPERIMENT_DIR)

    args = ap.parse_args()

    time_ratio = [float(t) for t in args.sched_time_ratio]
    considers_variants = args.considers_variants

    parameters = []

    for tr in time_ratio:
        output_dir = os.path.join(args.output_dir, f"time_ratio_{int(tr * 100)}")
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        for dataset in args.datasets:
            parameters.append((args.project_dir, dataset,
                               output_dir, tr,
                               get_metric(dataset, args.project_dir),
                               considers_variants))

    with multiprocessing.Pool(get_pool_size()) as p:
        p.starmap(run_experiments_with_threads, parameters)
