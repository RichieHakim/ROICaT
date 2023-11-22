from argparse import ArgumentParser
from vrAnalysis import database
from vrAnalysis import fileManagement
from roicat.pipelines import pipeline_tracking
from roicat.util import get_default_parameters

WORKFLOWS = ['tracking']

# === type definitions for argument parser ===
def workflow(string):
    """duck-type string to check if it is a valid ROICaT pipeline"""
    try:
        get_default_parameters(pipeline=string)
    except:
        raise ValueError(f"workflow {string} not supported")
    else:
        return string

# === argument parser for run ===    
def handle_arguments():
    parser = ArgumentParser(description='Arguments for running an ROICaT pipeline.')
    parser.add_argument('--mouse', type=str, required=True, help='the name of the mouse to process')
    parser.add_argument('--workflow', type=workflow, required=True, help='which workflow to perform')
    parser.add_argument('--no-database-update', default=False, action='store_true', help='if used, will not do a database update')
    parser.add_argument('--nosave', action='store_true', default=False, help='whether to prevent saving')
    return parser.parse_args()

# === custom method for getting the appropriate dir_outer ===
def define_dirs(args):
    # first define save path and save name
    dir_save = fileManagement.localDataPath() / args.mouse
    name_save = lambda planeName: args.mouse + '.' + planeName
    
    # identify sessions for requested mouse (that match relevant criteria in database)
    vrdb = database.vrDatabase('vrSessions')
    ises = vrdb.iterSessions(imaging=True, mouseName=args.mouse)
    assert len(ises)>0, f"no sessions found for mouse={args.mouse}"
    session_paths = [ses.sessionPath() for ses in ises]

    # for identified sessions, determine which planes are present
    plane_in_ses = [set([pn.stem for pn in ses.suite2pPath().rglob('plane*')]) for ses in ises]
    plane_names = set.union(*plane_in_ses)
    assert all([planes == plane_names for planes in plane_in_ses]), "requested sessions have different sets of planes"
    from natsort import natsorted
    plane_names = natsorted(list(plane_names), key=lambda x: x.lower()) 

    # inform the user what was found for this run
    print('')
    print(f"Running ROICaT:{args.workflow} on the following sessions:")
    for idx, ises in enumerate(ises):
        print('  ', idx, ises.sessionPrint())
    
    print('')
    print(f"Using plane_names: {', '.join(plane_names)}")

    # return to main
    return session_paths, plane_names, dir_save, name_save

# === main program that runs the requested pipeline ===
if __name__ == "__main__":
    args = handle_arguments()
    params = get_default_parameters(args.workflow) # get default params
    session_paths, plane_names, dir_save, name_save = define_dirs(args) # get session paths and planes to track
    params['data_loading']['dir_outer'] = session_paths # load paths into the params dictionary
        
    if args.nosave: 
        params['results_saving']['dir_save'] = None
    else:
        params['results_saving']['dir_save'] = dir_save # indicate where to save results

    # Go through each plane and run the pipeline
    for idx_plane, plane_name in enumerate(plane_names):
        params['data_loading']['reMatch_in_path'] = plane_name # update planeName to filter paths by
        params['results_saving']['prefix_name_save'] = name_save(plane_name) # define what the name is (combination of mouse name and plane name)
        results, run_data, _ = pipeline_tracking(params) # do pipeline!





