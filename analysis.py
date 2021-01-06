import argparse
import copy
import os
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from roses.effect_size import vargha_delaney
from roses.statistical_test.kruskal_wallis import kruskal_wallis
from pathlib import Path

MAX_XTICK_WIDTH = 13

# For a beautiful plots
plt.style.use('ggplot')
sns.set_style("whitegrid")
sns.set(palette="pastel")


class Analisys(object):

    def __init__(self, project_dir, results_dir, font_size_plots=25, sched_time_ratio=[0.1, 0.5, 0.8]):
        self.project_dir = project_dir
        self.project = project_dir.split('/')[-1]
        self.results_dir = results_dir

        self.update_figure_dir(f"{self.results_dir}/{self.project}")

        self.sched_time_ratio = sched_time_ratio
        self.sched_time_ratio_names = [
            str(int(tr * 100)) for tr in sched_time_ratio]

        self.reward_names = {
            'Time-ranked Reward': 'TimeRank',
            'Reward Based on Failures': 'RNFail'
        }

        # Load the information about the system
        self.df_system = self._get_df_system()

        # Load the results from system
        self.datasets = {}
        self._load_datasets()

        self.font_size_plots = font_size_plots
        self._update_rc_params()

    def update_figure_dir(self, path):
        self.figure_dir = path
        Path(self.figure_dir).mkdir(parents=True, exist_ok=True)

    def update_project(self, project_dir):
        self.project_dir = project_dir
        self.project = project_dir.split('/')[-1]

        self.figure_dir = f"{self.results_dir}/{self.project}"

        Path(self.figure_dir).mkdir(parents=True, exist_ok=True)

        # Load the information about the system
        self.df_system = self._get_df_system()

        # Load the results from system
        self.datasets = {}
        self._load_datasets()

    def update_font_size(self, font_size_plots):
        self.font_size_plots = font_size_plots
        self._update_rc_params()

    def _update_rc_params(self):
        plt.rcParams.update({
            'font.size': self.font_size_plots,
            'xtick.labelsize': self.font_size_plots,
            'ytick.labelsize': self.font_size_plots,
            'legend.fontsize': self.font_size_plots,
            'axes.titlesize': self.font_size_plots,
            'axes.labelsize': self.font_size_plots,
            'figure.max_open_warning': 0,
            'pdf.fonttype': 42
        })

    def _get_df_system(self):
        # Dataset Info
        df = pd.read_csv(f'{self.project_dir}/features-engineered.csv', sep=';', thousands=',')
        df = df.groupby(['BuildId'], as_index=False).agg({'Duration': np.sum})
        df.rename(columns={'BuildId': 'step',
                           'Duration': 'duration'}, inplace=True)

        return df

    def _load_datasets(self):
        for tr in self.sched_time_ratio_names:
            df_path = f"{self.results_dir}/time_ratio_{tr}"

            df = pd.read_csv(f'{df_path}/{self.project}.csv', sep=';', thousands=',', low_memory=False)

            df = df[['experiment', 'step', 'policy', 'reward_function', 'prioritization_time', 'time_reduction', 'ttf',
                     'fitness', 'avg_precision', 'cost', 'rewards']]

            df['reward_function'] = df['reward_function'].apply(
                lambda x: x if x == '-' else x.replace(x, self.reward_names[x]))

            df['name'] = df.apply(lambda row: f"{row['policy']} ({row['reward_function']})" if 'Deterministic' not in row['policy'] else 'Deterministic', axis=1)

            builds = df['step'].max()
            # Find the deterministic
            dt = df[df['name'] == 'Deterministic']

            # As we have only one experiment run (deterministic), we increase to have 30 independent runs
            # This allow us to calculate the values without problems :D
            dt = dt.append([dt] * 29, ignore_index=True)
            dt['experiment'] = np.repeat(list(range(1, 31)), builds)

            # Clean
            df = df[df['name'] != 'Deterministic']

            df = df.append(dt)

            df.sort_values(by=['name'], inplace=True)

            df.drop_duplicates(inplace=True)

            self.datasets[tr] = df

    def _get_metric_ylabel(self, column, rw=None):
        metric = 'NAPFD'
        ylabel = metric
        if column == 'cost':
            metric = 'APFDc'
            ylabel = metric
        elif column == 'ttf':
            metric = 'RFTC'
            ylabel = 'Rank of the Failing Test Cases'
        elif column == 'prioritization_time':
            metric = 'PrioritizationTime'
            ylabel = 'Prioritization Time (sec.)'
        elif column == "rewards":
            metric = rw
            ylabel = rw

        return metric, ylabel

    def _get_rewards(self):
        if len(self.datasets.keys()) > 0:
            return self.datasets[list(self.datasets.keys())[0]]['reward_function'].unique()
        else:
            return []

    def _get_policies(self):
        if len(self.datasets.keys()) > 0:
            return self.datasets[list(self.datasets.keys())[0]]['name'].unique()
        else:
            return []

    def print_mean(self, df, column, direction='max'):
        mean = df.groupby(['name'], as_index=False).agg(
            {column: ['mean', 'std', 'max', 'min']})
        mean.columns = ['name', 'mean', 'std', 'max', 'min']

        # sort_df(mean)

        # Round values (to be used in the article)
        mean = mean.round({'mean': 4, 'std': 3, 'max': 4, 'min': 4})
        mean = mean.infer_objects()

        bestp = mean.loc[mean['mean'].idxmax() if direction ==
                         'max' else mean['mean'].idxmin()]

        val = 'Highest' if direction == 'max' else 'Lowest'

        print(f"\n{val} Value found by {bestp['name']}: {bestp['mean']:.4f}")
        print("\nMeans:")
        print(mean)

        return mean, bestp['name']

    def print_mean_latex(self, x, column):
        policies = self._get_policies()

        print(*policies, sep="\t")
        cont = len(policies)

        for policy in policies:
            df_temp = x[x.name == policy]
            print(f"{df_temp[column].mean():.4f} $\pm$ {df_temp[column].std():.3f} ", end="")

            cont -= 1
            if cont != 0:
                print("& ", end="")
        print()

    def _define_axies(self, ax, tr, column, ylabel=None):
        metric, ylabel_temp = self._get_metric_ylabel(column)

        ax.set_xlabel('CI Cycle', fontsize=self.font_size_plots)
        ax.set_ylabel(ylabel + " " + metric if ylabel is not None else metric,
                      fontsize=self.font_size_plots)
        ax.set_title(f"Time Budget: {tr}%", fontsize=self.font_size_plots)

    def _plot_accumulative(self, df, ax, tr, column='fitness'):
        df = df[['step', 'name', column]]
        df.groupby(['step', 'name']).mean()[
            column].unstack().cumsum().plot(ax=ax, linewidth=3)

        self._define_axies(ax, tr, column, ylabel='Accumulative')

    def plot_accumulative(self, figname, column='fitness'):
        policies = self._get_policies()

        fig, axes = plt.subplots(
            ncols=len(self.datasets.keys()), sharex=True, sharey=True, figsize=(25, 8))
        # Todo try a generic way
        (ax1, ax2, ax3) = axes

        for df_k, tr, ax in zip(self.datasets.keys(), self.sched_time_ratio_names, [ax1, ax2, ax3]):
            self._plot_accumulative(self.datasets[df_k], ax, tr, column)

        handles, labels = ax1.get_legend_handles_labels()
        lgd = fig.legend(handles, labels, bbox_to_anchor=(
            0, -0.03, 1, 0.2), loc='lower center', ncol=len(policies))
        # lgd = ax1.legend(handles, labels, bbox_to_anchor=(-0.02,
        # 1.05, 1, 0.2), loc='lower left', ncol=len(policies))

        ax1.get_legend().remove()
        ax2.get_legend().remove()
        ax3.get_legend().remove()

        plt.tight_layout()
        plt.savefig(f"{self.figure_dir}/{figname}.pdf", bbox_inches='tight')
        plt.cla()
        plt.close(fig)

    def _plot_lines(self, df, ax, tr, column='fitness'):
        df = df[['step', 'name', column]]
        df.groupby(['step', 'name']).mean()[
            column].unstack().plot(ax=ax, linewidth=3)

        self._define_axies(ax, tr, column)

    def plot_lines(self, figname, column='fitness'):
        policies = self._get_policies()

        fig, axes = plt.subplots(
            nrows=len(self.datasets.keys()), sharex=True, sharey=True, figsize=(int(10 * 3 * (len(policies) / 3)), 20))
        # Todo try a generic way
        (ax1, ax2, ax3) = axes

        for df_k, tr, ax in zip(self.datasets.keys(), self.sched_time_ratio_names, [ax1, ax2, ax3]):
            self._plot_lines(self.datasets[df_k], ax, tr, column)

        handles, labels = ax1.get_legend_handles_labels()
        lgd = ax1.legend(handles, labels, bbox_to_anchor=(-0.005, 1.05, 1, 0.2), loc='lower left',
                         ncol=len(policies))

        ax2.get_legend().remove()
        ax3.get_legend().remove()

        plt.tight_layout()
        plt.savefig(f"{self.figure_dir}/{figname}.pdf", bbox_inches='tight')
        plt.cla()
        plt.close(fig)

    def _visualize_ntr(self, df, tr, ax, total_time_spent):
        policies = df['name'].unique()

        # Only the commits which failed
        x = df[['experiment', 'name', 'time_reduction']
               ][(df.avg_precision == 123)]

        df_ntr = pd.DataFrame(columns=['experiment', 'name', 'n_reduction'])

        print(*policies, sep="\t")
        cont = len(policies)

        for policy in policies:
            df_ntr_temp = x[x.name == policy]

            # sum all differences (time_reduction column) in all cycles for
            # each experiment
            df_ntr_temp = df_ntr_temp.groupby(['experiment'], as_index=False).agg({
                'time_reduction': np.sum})

            # Evaluate for each experiment
            df_ntr_temp['n_reduction'] = df_ntr_temp['time_reduction'].apply(
                lambda x: x / (total_time_spent))

            df_ntr_temp['name'] = policy

            df_ntr_temp = df_ntr_temp[['experiment', 'name', 'n_reduction']]

            df_ntr = df_ntr.append(df_ntr_temp)

            print(f"{df_ntr_temp['n_reduction'].mean():.4f} $\pm$ {df_ntr_temp['n_reduction'].std():.3f} ", end="")

            cont -= 1
            if cont != 0:
                print("& ", end="")
        print()

        if len(df_ntr) > 0:
            df_ntr.sort_values(by=['name', 'experiment'], inplace=True)
            sns.boxplot(x='name', y='n_reduction', data=df_ntr, ax=ax)
            ax.set_xlabel('')
            ax.set_ylabel('Normalized Time Reduction' if tr ==
                          '10' else '')  # Workaround
            ax.set_title(f"Required Time: {tr}%")
            ax.set_xticklabels(textwrap.fill(x.get_text(), MAX_XTICK_WIDTH)
                               for x in ax.get_xticklabels())

    def visualize_ntr(self):
        # Total time spent in each Cycle
        total_time_spent = self.df_system['duration'].sum()
        policies = self._get_policies()

        fig, axes = plt.subplots(
            ncols=len(self.datasets.keys()), sharex=True, sharey=True, figsize=(int(8.3 * 3 * (len(policies) / 3)), 8))
        (ax1, ax2, ax3) = axes

        for df_k, tr, ax in zip(self.datasets.keys(), self.sched_time_ratio_names, [ax1, ax2, ax3]):
            print(f"\nRequired Time {tr}%")
            df = self.datasets[df_k]
            self._visualize_ntr(df, tr, ax, total_time_spent)

        plt.tight_layout()
        plt.savefig(f"{self.figure_dir}/NTR.pdf", bbox_inches='tight')
        plt.cla()
        plt.close(fig)

    def _visualize_duration(self, df):
        dd = df[['name', 'prioritization_time']]
        # sort_df(dd)
        self.print_mean(dd, 'prioritization_time', direction='min')
        self.print_mean_latex(dd, 'prioritization_time')

    def visualize_duration(self):
        # print(f"\n\n||||||||||||||||||||||||||||||| PRIORITIZATION DURATION
        # |||||||||||||||||||||||||||||||\n")
        for df_k, tr in zip(self.datasets.keys(), self.sched_time_ratio_names):
            print(f"\nRequired Time {tr}%")
            df = self.datasets[df_k]
            self._visualize_duration(df)

    def _statistical_test_kruskal(self, df, ax, column):
        if column == 'ttf':
            df = df[df.ttf > 0]

        if(len(df)) > 0:
            # Get the mean of fitness in each experiment
            x = df[['experiment', 'name', column]].groupby(['experiment', 'name'], as_index=False).agg(
                {column: np.mean})

            # Remove unnecessary columns
            x = x[['name', column]]

            mean, best = self.print_mean(x, column, 'min' if column in [
                                         'ttf', 'prioritization_time'] else 'max')
            mean['eff_symbol'] = " "

            posthoc = None

            try:
                k = kruskal_wallis(x, column, 'name')
                kruskal, posthoc = k.apply(ax)
                print(f"\n{kruskal}")  # Kruskal results

                if posthoc is not None:
                    print("\n--- POST-HOC TESTS ---")
                    print("\np-values:")
                    print(posthoc[0])

                    # Get the posthoc
                    df_eff = vargha_delaney.reduce(posthoc[1], best)

                    print(df_eff)

                    mean['eff_symbol'] = mean.apply(lambda x: "$\\bigstar$" if x['name'] == best
                                                    else df_eff.loc[df_eff.compared_with == x['name'], 'effect_size_symbol'].values[0]
                                                    if len(df_eff.loc[df_eff.compared_with == x['name'], 'effect_size_symbol'].values) > 0
                                                    else df_eff.loc[df_eff.base == x['name'], 'effect_size_symbol'].values[0], axis=1)

            except:
                print("Kruskal-Wallis not applied")

            # Concat the values to a unique columns
            mean['avg_std_effect'] = mean['mean'].map(str) + ' $\\pm$ ' + mean['std'].map(str) + " " + mean[
                'eff_symbol'].map(str)

            # Select the main information
            mean = mean[['name', 'avg_std_effect']]

            mean_trans = mean.copy()
            mean_trans.index = mean['name']
            mean_trans = mean_trans.transpose()

            print("\nAVG +- STD EFFECT_SIZE:")
            temp_x = mean_trans.to_string(
                index=False, index_names=False).split("\n")[1:]
            print(temp_x[0])  # Column names

            # Just a beautiful print to use with LaTeX table :}
            temp_split = temp_x[1].split()
            cont = 1
            divn = 4 if posthoc is not None else 3

            for x in temp_split:
                print(f"{x} ", end="")
                if (cont % divn == 0 and cont != len(temp_split)):
                    print("& ", end="")
                cont += 1
            print("\n\n")

    def statistical_test_kruskal(self, column='fitness'):
        metric, ylabel = self._get_metric_ylabel(column)

        print(
            f"\n\n\n\n||||||||||||||||||||||||||||||| STATISTICAL TEST - KRUSKAL WALLIS - {metric} |||||||||||||||||||||||||||||||\n")

        rewards = self._get_rewards()
        policies = self._get_policies()

        fig, axes = plt.subplots(
            ncols=len(self.datasets.keys()), sharex=True, sharey=True, figsize=(int(8.3 * 3 * (len(policies) / 3)), 8))
        (ax1, ax2, ax3) = axes

        for df_k, tr, ax in zip(self.datasets.keys(), self.sched_time_ratio_names, [ax1, ax2, ax3]):
            print(f"~~~~ Time Budget {tr}% ~~~~")
            df = self.datasets[df_k]
            self._statistical_test_kruskal(df, ax, column)
            ax.set_title(f"Time Budget: {tr}%")
            ax.set_ylabel(ylabel if tr == '10' else '')  # Workaround
            ax.set_xticklabels(textwrap.fill(x.get_text(), MAX_XTICK_WIDTH)
                               for x in ax.get_xticklabels())

        plt.tight_layout()
        plt.savefig(f"{self.figure_dir}/{metric}_Kruskal.pdf", bbox_inches='tight')
        plt.cla()
        plt.close(fig)

    def _rmse_calculation(self, df, column='fitness'):
        policies = self._get_policies()
        columns = [column, 'experiment', 'step']

        builds = df['step'].max()

        # Get only the required columns
        df = df[['experiment', 'step', 'name', column]]

        # Orderby to guarantee the right va
        df.sort_values(by=['experiment', 'step'], inplace=True)

        df_rmse = pd.DataFrame(
            columns=['experiment', 'step', 'Deterministic'])

        dt = df.loc[df['name'] == 'Deterministic', columns]

        df_rmse['Deterministic'] = dt[column]
        df_rmse['experiment'] = dt['experiment']
        df_rmse['step'] = dt['step']

        dt = df.loc[df['name'] == 'FRRMAB (RNFail)', columns]
        df_rmse['FRRMAB (RNFail)'] = dt[column].tolist()

        dt = df.loc[df['name'] == 'FRRMAB (TimeRank)', columns]
        df_rmse['FRRMAB (TimeRank)'] = dt[column].tolist()

        dt = df.loc[df['name'] == 'ANN (RNFail)', columns]
        df_rmse['ANN (RNFail)'] = dt[column].tolist()

        dt = df.loc[df['name'] == 'ANN (TimeRank)', columns]

        df_rmse['ANN (TimeRank)'] = dt[column].tolist()

        df_rmse = df_rmse.reset_index()

        df_rmse['RMSE - FRRMAB (RNFail)'] = df_rmse.apply(
            lambda x: (x['FRRMAB (RNFail)'] - x['Deterministic'])**2, axis=1)
        df_rmse['RMSE - FRRMAB (TimeRank)'] = df_rmse.apply(
            lambda x: (x['FRRMAB (TimeRank)'] - x['Deterministic'])**2, axis=1)

        df_rmse['RMSE - ANN (RNFail)'] = df_rmse.apply(
            lambda x: (x['ANN (RNFail)'] - x['Deterministic'])**2, axis=1)
        df_rmse['RMSE - ANN (TimeRank)'] = df_rmse.apply(
            lambda x: (x['ANN (TimeRank)'] - x['Deterministic'])**2, axis=1)

        df_f = df_rmse.groupby(['experiment'], as_index=False).agg(
            {'RMSE - ANN (RNFail)': lambda x: np.sqrt(sum(x) / builds)})
        print(f"ANN (RNFail): {round(df_f['RMSE - ANN (RNFail)'].mean(), 4):.4f} $\\pm$ {round(df_f['RMSE - ANN (RNFail)'].std(), 4):.4f}")

        df_f = df_rmse.groupby(['experiment'], as_index=False).agg(
            {'RMSE - ANN (TimeRank)': lambda x: np.sqrt(sum(x) / builds)})
        print(f"ANN (TimeRank): {round(df_f['RMSE - ANN (TimeRank)'].mean(), 4):.4f} $\\pm$ {round(df_f['RMSE - ANN (TimeRank)'].std(), 4):.4f}")

        df_f = df_rmse.groupby(['experiment'], as_index=False).agg(
            {'RMSE - FRRMAB (RNFail)': lambda x: np.sqrt(sum(x) / builds)})
        print(f"FRRMAB (RNFail): {round(df_f['RMSE - FRRMAB (RNFail)'].mean(), 4):.4f} $\\pm$ {round(df_f['RMSE - FRRMAB (RNFail)'].std(), 4):.4f}")

        df_f = df_rmse.groupby(['experiment'], as_index=False).agg(
            {'RMSE - FRRMAB (TimeRank)': lambda x: np.sqrt(sum(x) / builds)})
        print(f"FRRMAB (TimeRank): {round(df_f['RMSE - FRRMAB (TimeRank)'].mean(), 4):.4f} $\\pm$ {round(df_f['RMSE - FRRMAB (TimeRank)'].std(), 4):.4f}")

    def rmse_calculation(self, column='fitness'):
        for df_k, tr in zip(self.datasets.keys(), self.sched_time_ratio_names):
            print(f"\nRequired Time {tr}%")
            df = self.datasets[df_k]
            self._rmse_calculation(df, column)


def run_complete_analysis(ana):
    # Accumulative
    ana.plot_accumulative("ACC_NAPFD")
    ana.plot_accumulative("ACC_APFDc", 'cost')  # APFDc
    # ana.plot_accumulative("ACC_Reward", 'rewards')  # Rewards

    # Variation Visualization along the CI Cycles
    ana.plot_lines("NAPFD_Variation")
    ana.plot_lines("APFDc_Variation", 'cost')  # APFDc

    # Normalized Time Reduction
    print(
        f"\n\n\n\n||||||||||||||||||||||||||||||| NORMALIZED TIME REDUCTION (NTR) |||||||||||||||||||||||||||||||\n")
    ana.visualize_ntr()

    print(
        f"\n\n\n\n||||||||||||||||||||||||||||||| RMSE |||||||||||||||||||||||||||||||\n")
    print("\n======= NAPFD =======")
    ana.rmse_calculation()
    print("\n\n======= APFDc =======")
    ana.rmse_calculation('cost')

    # Apply the Kruskal-Wallis Test in the Data
    ana.statistical_test_kruskal()  # NAPFD
    ana.statistical_test_kruskal('cost')  # APFDc
    ana.statistical_test_kruskal('ttf')  # RFTC
    ana.statistical_test_kruskal('prioritization_time')  # Prioritization
    # Time

if __name__ == '__main__':
    # DATASETS = ['alibaba@druid']
    DATASETS = ['alibaba@druid', 'alibaba@fastjson', 'deeplearning4j@deeplearning4j', 'DSpace@DSpace',
                'gsdtsr', 'google@guava', 'iofrol', 'lexisnexis', 'paintcontrol',
                'square@okhttp', 'square@retrofit', 'zxing@zxing']
    project_dir_temp = '/mnt/sda4/mab-datasets'
    #project_dir_temp = 'data'
    results_dir = 'results/experiments'

    # for i, dataset in enumerate(args.datasets):
    for i, dataset in enumerate(DATASETS):
        print(f"====================================================\n\t\t{dataset}\n"
              f"====================================================")

        project_dir = f"{project_dir_temp}/{dataset}"

        if i == 0:
            ana = Analisys(project_dir, results_dir)
        else:
            ana.update_project(project_dir)

        run_complete_analysis(ana)
