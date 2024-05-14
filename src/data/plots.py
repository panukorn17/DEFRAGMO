import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from tqdm import tqdm
from rdkit.Chem import BRICS
from utils.mol_utils import mol_from_smiles
from pathlib import Path

plt.style.use('ggplot')

# Enable progress_apply
tqdm.pandas()

TITLE_FONT_SIZE = 16
TICK_FONT_SIZE = 12
IMAGE_DIR = Path('./images/ZINC')

def plot_number_of_fragments(df, method, min_len):
    # plot the counts of the number of fragments in a bar chart using matplotlib with n_fragments on the x-axis and the count on the y-axis
    # initialise the plot
    # sort plot by number of fragments
    plt.figure(figsize=(8, 6))
    df['n_fragments'].value_counts().sort_index().plot(kind='bar')
    plt.xlabel('Number of fragments')
    plt.ylabel('Number of Molecules with n fragments')
    #plt.title(f'Number of fragments in {method} dataset', fontsize=TITLE_FONT_SIZE)
    # Tick font size
    plt.xticks(fontsize=TICK_FONT_SIZE)
    plt.yticks(fontsize=TICK_FONT_SIZE)
    # font size for title
    # save the plot
    plt.savefig(IMAGE_DIR / f"number_of_fragments_{method}_min_len_{min_len}.png")

def plot_most_frequent_fragments(df, method, min_len):
    # plot the counts of the top 10 frequently occurring fragments in a bar chart using matplotlib with fragments on the x-axis and the count on the y-axis
    fragments = df['fragments'].str.split(' ')
    # flatten the list of lists
    fragments = [frag for frag_list in fragments for frag in frag_list]
    # count the frequency of each fragment
    fragment_counts = pd.Series(fragments).value_counts()
    # initialise the plot
    plt.figure(figsize=(10, 12))
    fragment_counts.head(15).plot(kind='bar')
    plt.xlabel('Fragment')
    plt.ylabel('Count')
    #plt.title(f'Most frequently occurring fragments in {method} dataset', fontsize=TITLE_FONT_SIZE)
    # Tick font size
    plt.xticks(fontsize=TICK_FONT_SIZE)
    plt.yticks(fontsize=TICK_FONT_SIZE)
    # save the plot
    plt.savefig(IMAGE_DIR / f"most_frequent_fragments_{method}_min_len_{min_len}.png")

def plot_fragment_efficiency(df, method, min_len):
    #create a histogram and box plot of the fragment efficiency
    plt.figure(figsize=(10, 6))
    df['fragmentation_efficiency'].plot(kind='hist', bins=10)
    plt.xlabel('Fragmentation Efficiency')
    plt.ylabel('Frequency')
    #plt.title(f'Fragmentation Efficiency in {method} dataset', fontsize=TITLE_FONT_SIZE)
    # Tick font size
    plt.xticks(fontsize=TICK_FONT_SIZE)
    plt.yticks(fontsize=TICK_FONT_SIZE)
    # save the plot
    plt.savefig(IMAGE_DIR / f"fragment_efficiency_hist_{method}_min_len_{min_len}.png")
    
    ticks = np.arange(0, 1.1, 0.2)
    plt.figure(figsize=(5, 5))
    df['fragmentation_efficiency'].plot(kind='box')
    #plt.ylabel('Fragmentation Efficiency')
    #plt.title(f'Fragmentation Efficiency in {method} dataset', fontsize=TITLE_FONT_SIZE)
    # Tick font size
    plt.yticks(ticks, fontsize=TICK_FONT_SIZE)
    # save the plot
    plt.savefig(f"{IMAGE_DIR}/fragment_efficiency_box_{method}_min_len_{min_len}.png")

def get_number_of_BRICS_bonds(df):
    """try:
        df['n_BRICS_bonds']
        return df
    except:"""
    # get the number of BRIC bonds in each molecule
    df['n_BRICS_bonds'] = df['smiles'].progress_apply(lambda x: len(list(BRICS.FindBRICSBonds(mol_from_smiles(x)))))
    return df

def get_fragmentation_efficiency(df):
    """try:
        df['fragmentation_efficiency'] = df['n_fragments'] / (df['n_BRICS_bonds']+1)
        return df
    except:"""
    df = get_number_of_BRICS_bonds(df)
    df['fragmentation_efficiency'] = df['n_fragments'] / (df['n_BRICS_bonds']+1)
    return df
