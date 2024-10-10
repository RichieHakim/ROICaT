# Demo script for running the tracking pipeline

from pathlib import Path
import datetime
import argparse

import roicat


## Ingest command line arguments
parser = argparse.ArgumentParser(description='Run the tracking pipeline on a dataset')
### path_params
parser.add_argument('--path_params',      type=str, default=None, help='Path to the parameters .yaml file to use')
### dir_outer
parser.add_argument('--dir_outer',        type=str, default=None, help='Path to the directory containing the dataset')
### dir_save
parser.add_argument('--dir_save',         type=str, default=None, help='Path to the directory to save the results')
### prefix_name_save. Default is the current date and time
parser.add_argument('--prefix_name_save', type=str, default=None, help='Prefix to append to the saved files')

args = parser.parse_args()

def clean_path(path, must_exist=False):
    if path is not None:
        path = str(Path(path).resolve())
        if not Path(path).exists() and must_exist:
            raise FileNotFoundError(f'Path not found: {path}')
    return path

path_params = clean_path(args.path_params, must_exist=True)
dir_outer =   clean_path(args.dir_outer, must_exist=True)
dir_save =    clean_path(args.dir_save, must_exist=False)
prefix_name_save = args.prefix_name_save if args.prefix_name_save is not None else str(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))


## Load in parameters to use
params = roicat.helpers.yaml_load(path_params)

## These lines are for safety, to make sure that all params are present and valid
params_defaults = roicat.util.get_default_parameters(pipeline='tracking')
params = roicat.helpers.prepare_params(params=params, defaults=params_defaults)

## User specified directories
params['data_loading']['dir_outer'] = dir_outer
params['results_saving']['dir_save'] = dir_save
params['results_saving']['prefix_name_save'] = prefix_name_save  ## Prefix to append to the saved files

## Run pipeline
results, run_data, params = roicat.pipelines.pipeline_tracking(params=params)