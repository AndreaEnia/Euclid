import argparse
import os, multiprocessing
max_cores = multiprocessing.cpu_count()
import numpy as np, pandas as pd
pd.set_option('mode.chained_assignment', None)
from astropy.io import fits
import corner
import itertools
from functools import partial
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

def extract_quantiles(i_target, quantiles, params):
    galaxy_row = phzCat[1].data[i_target]
    # In order to extract the multivariate PDFs, some work should be done with the OBJECT_ID,
    # NEIGHBOR_IDS, NEIGHBOR_WEIGHTS and NEIGHBOR_SCALING columns for each target source.
    kNN_IDs, kNN_wgt, kNN_scaling = galaxy_row['NEIGHBOR_IDS'], galaxy_row['NEIGHBOR_WEIGHTS'], galaxy_row['NEIGHBOR_SCALING']
    kNN_IDs, kNN_wgt, kNN_scaling = zip(*sorted(zip(kNN_IDs, kNN_wgt, kNN_scaling))) # Sort per index
    temp_kNN_df = index_df[index_df['id'].isin(kNN_IDs)]
    # .isin does not keep the original list order.
    # This horrible sequence of passages is fundamental
    # to assure that each ID has the proper associated weight
    temp_kNN_df['sort_cat'] = pd.Categorical(temp_kNN_df['id'], categories=kNN_IDs, ordered=True)
    temp_kNN_df.sort_values('sort_cat', inplace=True)
    temp_kNN_df.reset_index(inplace=True, drop = True)
    temp_kNN_df = temp_kNN_df.drop(columns={'sort_cat'})
    temp_kNN_df['WGT'] = kNN_wgt
    # Ok, now actually measure the quantiles
    posterior_df = pd.DataFrame(columns = params + ['WGT'])
    for _, row in temp_kNN_df.iterrows():
        pp_file = row['pp_file']
        pp_offset = row['pp_offset']
        try:
            sub = pp[int(pp_file)-1][int(pp_offset)]
            temp_pdf = pd.DataFrame(sub)
        except: continue
        temp_pdf['WGT'] = row['WGT']
        posterior_df = pd.concat([posterior_df, temp_pdf[params + ['WGT']]])
        ## Log at least the masses
        #for param in posterior_df.columns:
        #	if posterior_df[param].values.ptp() > 1E4: posterior_df[param] = np.log10(posterior_df[param])
        #	elif posterior_df[param].values.min() > 1E4: posterior_df[param] = np.log10(posterior_df[param])
        # posterior_df is a dataframe with all the posteriors points from the 30 NN, with an associated weight column
    # After that, the quantiles can be estimated with the corner integrated function corner.quantiles
    # (or directly using the whatever scipy function is calling).
    return np.insert(np.array([corner.quantile(posterior_df[param], q = quantiles, weights = posterior_df['WGT']) for param in params]).flatten(), 0, int(galaxy_row['ID']))

parser = argparse.ArgumentParser(description='Retrieve the desired quantiles (default 16, 50, 84) from NNPZ-PP output.')
parser.add_argument('--nnpz_output', type=str, help='Output file produced by NNPZ.', required = True)
parser.add_argument('--ppref_path', type=str, help='Path to the reference sample files.', required = True)
parser.add_argument('--params', type=str, nargs = '+', help='Parameters names for which the quantiles will be computed.', required = True)
parser.add_argument('--quantiles', type=float, nargs = '+', default = [.16, .5, .84], help='Desired quantiles. Default is 16, 50 and 84.')
parser.add_argument('--cores', type=int, default = 0, choices = range(1, max_cores+1), help='The maximum number of cores to use. Default is the maximum number of available cores minus 1.')
parser.add_argument('--out_file', type=str, help='Output filename.', required = True)
args = parser.parse_args()

if __name__ == '__main__':
    # NNPZ's output is the phzCat.fits file, which contains lots of information on the
    # (configuration file requested) parameters PDFs (see column names for more info).
    print('Reading NNPZ output from {0}'.format(args.nnpz_output))
    phzCat = fits.open(args.nnpz_output)
    tot_rows = len(phzCat[1].data)

    # Read the reference sample (both index and pp_{} are necessary)
    print('Reading the reference sample files from {0}'.format(args.ppref_path))
    index_df = pd.DataFrame(np.load(args.ppref_path+'index.npy'))
    pp_paths = sorted([args.ppref_path+p for p in os.listdir(args.ppref_path) if p.startswith('pp_')])
    pp = [np.load(p, allow_pickle = True) for p in tqdm(pp_paths)]

    # Number of cores to use to run everything in parallel.
    if args.cores == 0: max_workers = max_cores - 1
    else: max_workers = args.cores

    # Process the rows in ||.
    print('Processing the results')
    with ProcessPoolExecutor(max_workers = max_workers) as executor:
        results = list(tqdm(executor.map(partial(extract_quantiles, quantiles=args.quantiles, params=args.params), range(tot_rows)), \
                        total = tot_rows, desc='Evaluating quantiles', bar_format='{l_bar}{bar}| Elapsed: {elapsed} ETA: {remaining}'))

    # Save the results to a DataFrame.
    columns = ['ID'] + [r[0]+'_'+str(r[1]) for r in itertools.product(args.params, args.quantiles)]
    pd.DataFrame(results, columns = columns).to_csv(args.out_file, index = False)
    print('Quantiles stored in {0}'.format(args.out_file))
