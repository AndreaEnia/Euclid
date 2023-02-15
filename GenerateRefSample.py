import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import pandas as pd, seaborn as sns
from astropy.table import Table
from tqdm import tqdm

def decode_column(df, colname):
    df[colname] = df[colname].str.decode('utf-8')
    return

pos_path = '/scratch/astro/andrea.enia/MAMBO_ref_sample/posteriors/'
ref_path = '/scratch/astro/andrea.enia/MAMBO_ref_sample/Ref_sample/'

def generate_pp_file(pos_file, pp, rp):
    import os
    idx = int(pos_file.split('_')[-1].split('.')[0])
    if os.path.exists(rp+'pp_data_{}.npy'.format(idx)) == True:
        print('Idx. {} already processed'.format(idx))
        return
    print('Processing file {}'.format(pos_file))
    sample_posterior = Table.read(pp+pos_file).to_pandas()
    decode_column(sample_posterior, 'OBJECT_ID')
    pp_columns = ['REDSHIFT', 'RED_CURVE_INDEX', 'EB_V', 'LUMINOSITY', 'AGE', 'METALLICITY', 'SFR', 'STELLARMASS', 'TAU']
    temp = [Table.from_pandas(sample_posterior[sample_posterior['OBJECT_ID'] == idx][pp_columns]) for idx in tqdm(sample_posterior['OBJECT_ID'].unique())]
    np.save(rp+'pp_data_{}.npy'.format(idx), np.array(temp))
    print('Saved idx {}'.format(idx))
    return

from functools import partial
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

posterior_files = [p for p in os.listdir(pos_path) if p.startswith('Sample')]
with ProcessPoolExecutor() as executor: executor.map(partial(generate_pp_file, pos_path=pos_path, ref_path=ref_path), posterior_files)