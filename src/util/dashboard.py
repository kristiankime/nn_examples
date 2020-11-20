import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import preprocessing

skills_cols = ["very_easy", "easy", "medium", "hard", "very_hard", "Graphing", "Numerical", "Verbal", "Algebraic", "Precalc", "Trig", "Logs", "Exp", "Alt.Var.Names", "Abstract.Constants", "Limits...Continuity", "Continuity..Definition", "Derivative..Definition..Concept", "Derivative..Shortcuts", "Product.Rule", "Quotient.Rule", "Chain.Rule", "Implicit.Differentiation", "Function.Analysis", "Applications", "Antiderivatives"]


def normalize_dashboard(dashboard):
    min_max_scaler = preprocessing.MinMaxScaler()
    scaled = min_max_scaler.fit_transform(dashboard.values)
    dashboard_normalized = pd.DataFrame(scaled)
    dashboard_normalized.columns = skills_cols
    return dashboard_normalized


def compare_dashboard_df(df1: pd.DataFrame, name1: str, df2: pd.DataFrame, name2: str, col: str) -> pd.DataFrame:
    return pd.concat([df1[col], df2[col]], axis=1, keys=[name1 + "_" + col, name2 + "_" + col])


def plot_dashboard_compare(run_dir, comp, skill, name):
    fig1, ax1 = plt.subplots()
    row_nums = range(0, len(comp.index))
    ax1.plot(row_nums, comp[f'nn_{skill}'], '-o', label=f'nn_{skill}')
    ax1.plot(row_nums, comp[f'pfa_{skill}'], '-o', label=f'pfa_{skill}')
    skill_fn = skill.replace(".", "_")
    fig1.savefig(os.path.join(run_dir, skill_fn, f'pfa_dnn_dash_compare_{skill_fn}_{name}.pdf'), bbox_inches='tight')
    plt.close(fig1)

def evaluate_dashboards(run_dir, nn_dashboard_normalized, pfa_dashboard_normalized):
    corr_list = []
    # for skill in ['very_easy']:
    for skill in skills_cols:

        comp = compare_dashboard_df(nn_dashboard_normalized, "nn", pfa_dashboard_normalized, "pfa", skill)
        comp_1000 = comp.sample(n=1000)
        comp_sorted_nn = comp.sort_values(by=f'nn_{skill}')
        comp_sorted_pfa = comp.sort_values(by=f'pfa_{skill}')
        comp_1000_sorted_nn = comp_1000.sort_values(by=f'nn_{skill}')
        comp_1000_sorted_pfa = comp_1000.sort_values(by=f'pfa_{skill}')

        skill_fn = skill.replace(".", "_")
        if not os.path.exists(os.path.join(run_dir, skill_fn)):
            os.makedirs(os.path.join(run_dir, skill_fn))

        # build a correlation CSV
        corr = comp.corr().iloc[0][1]
        corr_list.append((skill, corr))

        # write out the correlation
        with open(os.path.join(run_dir, skill_fn, f'corr_{skill_fn}.txt'), "w") as text_file:
            text_file.write(str(comp.corr()))

        plot_dashboard_compare(run_dir, comp, skill, "default")
        plot_dashboard_compare(run_dir, comp_sorted_nn, skill, "sorted_nn")
        plot_dashboard_compare(run_dir, comp_sorted_pfa, skill, "sorted_pfa")

        plot_dashboard_compare(run_dir, comp_1000, skill, "default_1k")
        plot_dashboard_compare(run_dir, comp_1000_sorted_nn, skill, "sorted_nn_1k")
        plot_dashboard_compare(run_dir, comp_1000_sorted_pfa, skill, "sorted_pfa_1k")

    # create DataFrame using data
    corr_df = pd.DataFrame(corr_list, columns =['skill', 'correlation'])
    corr_df.to_csv(os.path.join(run_dir, "correlations.csv"), header=True, index=False)