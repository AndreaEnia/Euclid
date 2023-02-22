import argparse
import os
from functools import partial
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
from astropy.table import Table
from tqdm import tqdm

def decode_column(df, colname):
    df[colname] = df[colname].str.decode('utf-8')
    return

def generate_pp_file(pos_file, pp, rp):
    import os
    pf_idx = int(pos_file.split('_')[-1].split('.')[0])
    if os.path.exists(rp+'pp_data_{}.npy'.format(pf_idx)) == True:
        print('Posterior idx. {} already processed'.format(pf_idx))
        return
    print('Processing file {}'.format(pos_file))
    sample_posterior = Table.read(pp+pos_file).to_pandas()
    try: decode_column(sample_posterior, 'OBJECT_ID')
    except: pass
    pp_columns = ['REDSHIFT', 'RED_CURVE_INDEX', 'EB_V', 'LUMINOSITY', 'AGE', 'METALLICITY', 'SFR', 'STELLARMASS', 'TAU']
    temp = [Table.from_pandas(sample_posterior[sample_posterior['OBJECT_ID'] == idx][pp_columns]) for idx in tqdm(sample_posterior['OBJECT_ID'].unique())]
    np.save(rp+'pp_data_{}.npy'.format(pf_idx), np.array(temp))
    print('Saved pf_idx {}'.format(pf_idx))
    return

parser = argparse.ArgumentParser(description='Generate the pp_data_{}.npy reference files for NNPZ.')
parser.add_argument('--pos_path', type=str, help='Path to the posteriors folder.', required = True)
parser.add_argument('--ref_path', type=str, help='Path to the reference sample folder.', required = True)
parser.add_argument('--cores', type=int, default = 0, choices = range(1, max_cores+1), help='The maximum number of cores to use. Default is the maximum number of available cores minus 1.')
args = parser.parse_args()

if __name__ == '__main__':
    # Number of cores to use to run everything in parallel.
    if args.cores == 0: max_workers = max_cores - 1
    else: max_workers = args.cores

    print('Fixing index file...')
    pp_index = Table.read(args.pos_path+'Index_File_posterior.fits').to_pandas()
    decode_column(pp_index, 'FILE_NAME')
    pp_index['pp_file'] = [int(s.split('_')[-1].split('.')[0]) for s in pp_index['FILE_NAME']]
    pp_index['pp_offset'] = np.array([list(np.arange(len(pp_index[pp_index['pp_file'] == idx]))) for idx in np.unique(pp_index['pp_file'])]).flatten()
    pp_index = pp_index.rename(columns = {'OBJECT_ID': 'id'})
    pp_index = pp_index[['id', 'pp_file', 'pp_offset']]
    pp_index['id'] = pp_index['id'].astype('int')
    index_df = pd.DataFrame(np.load(args.ref_path+'index.npy'))
    index_df = pd.merge(index_df, pp_index, on='id', how='outer')
    subprocess.call('mv '+args.ref_path+'index.npy '+args.ref_path+'original_index.npy', shell = True)
    print('Moved old index.npy file to {0}/original_index.npy'.format(args.ref_path))
    np.save(args.ref_path+'index.npy', index_df.to_records(index=False))
    print('Index file stored in {0}/index.npy'.format(args.ref_path))
    print('...done!')

    # Read the posterior files path
    posterior_files = [p for p in os.listdir(args.pos_path) if p.startswith('Sample')]

    # Process the files in ||.
    print('Generating the reference sample...')
    with ProcessPoolExecutor(max_workers = max_workers) as executor: executor.map(partial(generate_pp_file, pp=args.pos_path, rp=args.ref_path), posterior_files)
    print('Reference sample stored in {0}'.format(args.ref_path))
