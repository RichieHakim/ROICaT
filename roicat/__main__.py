from typing import Optional

import argparse
import datetime
from pathlib import Path

from roicat import pipelines, util, helpers

PIPELINES = {
    'tracking': pipelines.pipeline_tracking,
}

def add_args(parser: argparse.ArgumentParser):
    """
    Adds arguments to the parser.
    """
    # Running pipelines
    ## pipeline type
    parser.add_argument('--pipeline', type=str, required=False, help=f"Type of pipeline to run. Options: {', '.join(PIPELINES.keys())}")

    ## path_params
    parser.add_argument('--path_params', type=str, required=False, help='Path to the parameters .yaml file to use')

    ## dir_outer
    parser.add_argument('--dir_data', type=str, required=False, default=None, help='Path to the outer directory containing the dataset')

    ## dir_save
    parser.add_argument('--dir_save', type=str, required=False, default=None, help='Path to the directory to save the results')

    ## prefix_name_save. Default is the current date and time
    parser.add_argument('--prefix_name_save', type=str, required=False, default=None, help='Prefix to append to the saved files. Default is the current date and time')

    ## verbose
    parser.add_argument('--verbose', action='store_true', help='Print verbose output')

    ## use_onnx
    parser.add_argument('--use_onnx', action='store_true', default=False,
        help='Use ONNX runtime for ROInet inference (faster on CPU)')

    ## export_csv
    parser.add_argument('--export_csv', type=str, default=None,
        help='Path to export tracking results as CSV')

    # Other arguments
    ## version
    parser.add_argument('--version', action='store_true', help='Print the version number.')

    return parser


def run_pipeline(
    pipeline_name: str = 'tracking',
    path_params: Optional[str] = None,
    dir_data: Optional[str] = None,
    dir_save: Optional[str] = None,
    prefix_name_save: Optional[str] = None,
    verbose: bool = False,
    use_onnx: bool = False,
    export_csv: Optional[str] = None,
):
    """
    Call a pipeline with the specified parameters.
    """

    # Load in parameters to use
    if path_params is not None:
        params = helpers.yaml_load(path_params)
    else:
        print(f"WARNING: No parameters file specified. Using default parameters for pipeline '{pipeline_name}'")
        params = {}

    # These lines are for safety, to make sure that all params are present and valid
    params_defaults = util.get_default_parameters(pipeline=pipeline_name)
    params = helpers.prepare_params(params=params, defaults=params_defaults)

    # User specified directories
    def inplace_update_if_not_none(d, key, value):
        if value is not None:
            d[key] = value
    inplace_update_if_not_none(params['data_loading'], 'dir_outer', dir_data)
    inplace_update_if_not_none(params['results_saving'], 'dir_save', dir_save)
    inplace_update_if_not_none(params['results_saving'], 'prefix_name_save', prefix_name_save)
    inplace_update_if_not_none(params['general'], 'verbose', verbose)

    # Enable ONNX runtime for ROInet inference if requested
    if use_onnx:
        params['ROInet']['network']['use_onnx'] = True

    # Run pipeline
    results, run_data, params = PIPELINES[pipeline_name](params=params)

    # Export tracking results to CSV if requested
    if export_csv:
        if hasattr(util, 'export_tracking_results_to_csv'):
            util.export_tracking_results_to_csv(results, export_csv)
        else:
            print("Warning: export_tracking_results_to_csv not available in this version of ROICaT")


def _prepare_path(path, must_exist=False):
    """
    Resolve path and check if it exists.
    """
    if path is not None:
        path = str(Path(path).resolve())
        if not Path(path).exists() and must_exist:
            raise FileNotFoundError(f'Path not found: {path}')
    return path


def main():
    """
    Main function.
    """
    parser = argparse.ArgumentParser(description='Run a ROICaT pipeline')
    parser = add_args(parser)
    args = parser.parse_args()

    if args.version:
        import importlib.metadata
        print(f"ROICaT version: {importlib.metadata.version('roicat')}")
    
    ## If nothing is specified, print a message
    if not any(vars(args).values()):
        print("Welcome to ROICaT!\n    Use the --help flag to see the available options.")

    ## If the pipeline argument is specified, call the pipeline function

    path_params = _prepare_path(args.path_params, must_exist=True)
    dir_data =    _prepare_path(args.dir_data,    must_exist=True)
    dir_save =    _prepare_path(args.dir_save,    must_exist=False)
    prefix_name_save = args.prefix_name_save if args.prefix_name_save is not None else str(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))

    if args.pipeline is not None:
        run_pipeline(
            pipeline_name=args.pipeline,
            path_params=path_params,
            dir_data=dir_data,
            dir_save=dir_save,
            prefix_name_save=prefix_name_save,
            verbose=args.verbose,
            use_onnx=args.use_onnx,
            export_csv=args.export_csv,
        )


if __name__ == '__main__':
    main()