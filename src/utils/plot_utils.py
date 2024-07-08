import os
import pandas as pd
import seaborn as sns

from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from utils.mol_utils import mols_from_smiles
from utils.config import get_data_info, DATA_DIR
from data.preprocess import add_atom_counts, add_bond_counts, add_ring_counts, add_property


sns.set_theme('paper')
sns.set_style('whitegrid', {'axes.grid': False})
params = {
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica'],
    'legend.fontsize': 'x-small',
    'legend.handlelength': 1,
    'legend.handletextpad': 0.2,
    'legend.columnspacing': 0.8,
    'xtick.labelsize': 'x-small',
    'ytick.labelsize': 'x-small'}
plt.rcParams.update(params)

ratio = 0.4

props = ['qed', 'SAS', 'logP']

feats = {
    'atoms': ['C', 'F', 'N', 'O', 'Other'],
    'bonds': ['SINGLE', 'DOUBLE', 'TRIPLE'],
    'rings': ['Tri', 'Quad', 'Pent', 'Hex']
}

MODEL = 'OURS'

def plot_property(df, name, prop, ax=None):
    new_names = dict([(p, p.upper()) for p in props])
    df.rename(columns=new_names, inplace=True)
    sns.distplot(df[prop.upper()][df.who==name], hist=False, label=name, ax=ax)
    ax = sns.distplot(df[prop.upper()][df.who==MODEL], hist=False, label=MODEL, ax=ax)
    ax.set_aspect(1.0/ax.get_data_ratio()*ratio)


def plot_count(df, name, feat, ax=None):
    s1 = df[feats[feat]][df.who==name].mean(axis=0)
    s2 = df[feats[feat]][df.who==MODEL].mean(axis=0)
    data = pd.DataFrame([s1, s2], index=[name, MODEL])
    ax = data.plot(kind='bar', stacked=True, ax=ax, rot=0)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.9),
          ncol=len(feats[feat]), framealpha=0, borderpad=1, title=feat.upper())


def plot_counts(df, dataset_name, sample_name, result_dir):
    fig, axs = plt.subplots(1, 3)
    for i, f in enumerate(feats):
        plot_count(df, dataset_name, f, ax=axs.flat[i])
        axs.flat[i].set_aspect(1.0/axs.flat[i].get_data_ratio()*ratio)
    fig.savefig(result_dir / f'{sample_name}_counts_{dataset_name}.svg')


def plot_props(df, dataset_name, sample_name, result_dir):
    fig, axs = plt.subplots(1, 3)
    for i, p in enumerate(props):
        plot_property(df, dataset_name, p, ax=axs.flat[i])
        axs.flat[i].set_aspect(1.0/axs.flat[i].get_data_ratio()*ratio)
    fig.savefig(result_dir / f'{sample_name}_props_{dataset_name}.svg')


def plot_paper_figures(run_dir, run_name, sample_name):
    data_name = "ZINC" if "ZINC" in run_dir.name else "PCBA"
    result_dir = run_dir / 'results'
    df = pd.read_csv(result_dir / f'{sample_name}.smi')
    data_info = get_data_info(data_name)

    # Check if the df has all the columns detailed in data_info['column_order']
    processed = True
    for col in data_info['column_order']:
        if col not in df.columns:
            processed = False
    if not processed:
        smiles = df.smiles.tolist()
        mols = mols_from_smiles(smiles)
        df = add_atom_counts(df, mols, data_info)
        df = add_bond_counts(df, mols, data_info)
        df = add_ring_counts(df, mols, data_info)
        print("Adding properties ...")
        for prop in tqdm(data_info['properties']):
            if prop not in df.columns:
                df = add_property(df, mols, prop)
        df.to_csv(run_dir / 'results' / f'{sample_name}.smi', index=False)
    df['who'] = 'OURS'
    df_train = pd.read_csv(DATA_DIR / data_name / 'processed' / 'train.smi')
    df_train['who'] = data_name
    dataset = pd.concat([df, df_train], ignore_index=True)
    plot_counts(dataset, data_name, sample_name, result_dir)
    plot_props(dataset, data_name, sample_name, result_dir)