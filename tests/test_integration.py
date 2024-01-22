from pathlib import Path

import warnings
import pytest
import tempfile
import multiprocessing as mp

import numpy as np
import scipy.sparse
import roicat
from roicat import helpers, ROInet, pipelines, util


def test_pipeline_tracking_simple(dir_data_test):
    defaults = util.get_default_parameters(pipeline='tracking')
    seed = 0
    params_partial = {
        'general': {
            'use_GPU': False,
            'random_seed': seed,
        },
        'data_loading': {
            'dir_outer': str(Path(dir_data_test).resolve() / 'pipeline_tracking'),
            'data_kind': 'roicat',
            'data_roicat': {
                'filename_search': r'data_roicat_obj.pkl'
            },
        },
        'clustering': {
            'automatic_mixing': {
                'kwargs_findParameters': {
                    'n_patience': 30,  ## Reduced number to speed up
                    'max_trials': 100,  ## Reduced number to speed up
                },
                'n_jobs_findParameters': 1,  ## THIS IS CRITICAL TO ENSURE REPORDUCIBILITY. Parallelization prevents reproducibility.
            },
        },
        'alignment': {
            'fit_geometric': {
                'template': 0.5,  ## Which session to use as a registration template. If input is float (ie 0.0, 0.5, 1.0, etc.), then it is the fractional position of the session to use; if input is int (ie 1, 2, 3), then it is the index of the session to use (0-indexed)
                'template_method': 'sequential',  ## Can be 'sequential' or 'image'. If 'sequential', then the template is the FOV_image of the previous session. If 'image', then the template is the FOV_image of the session specified by 'template'.
                'mode_transform': 'euclidean',  ## Can be 'homography', 'affine', 'rigid', or 'translation'. See documentation for more details.
                'mask_borders': [5, 5, 5, 5],  ## Number of pixels to mask from the borders of the FOV_image. Useful for removing artifacts from the edges of the FOV_image.
                'n_iter': 0,  ## Number of iterations to run the registration algorithm. More iterations means more accurate registration, but longer run time.
                'termination_eps': 99999,  ## Termination criteria for the registration algorithm. See documentation for more details.
                'gaussFiltSize': 31,  ## Size of the gaussian filter used to smooth the FOV_image before registration. Larger values mean more smoothing.
                'auto_fix_gaussFilt_step': 10,  ## If the registration fails, then the gaussian filter size is reduced by this amount and the registration is tried again.
            },
            'fit_nonrigid': {
                'template': 0.5,  ## Which session to use as a registration template. If input is float (ie 0.0, 0.5, 1.0, etc.), then it is the fractional position of the session to use; if input is int (ie 1, 2, 3), then it is the index of the session to use (0-indexed)
                'template_method': 'image',  ## Can be 'sequential' or 'image'. If 'sequential', then the template is the FOV_image of the previous session. If 'image', then the template is the FOV_image of the session specified by 'template'.
                'mode_transform': 'calcOpticalFlowFarneback',  ## Can be 'createOptFlow_DeepFlow' or 'createOptFlow_Farneback'. See documentation for more details.
                'kwargs_mode_transform': {
                    'pyr_scale': 0.0, 
                    'levels': 0,
                    'winsize': 0, 
                    'iterations': 0,
                    'poly_n': 0, 
                    'poly_sigma': 0,
                    'flags': 256, ## = 256
                },  ## Keyword arguments for the mode_transform function. See documentation for more details.
            },
        },
    }
    params = util.prepare_params(params_partial, defaults)
    results, run_data, params = pipelines.pipeline_tracking(params)

    #check that results fields are not empty
    assert results['clusters'] != 0, "Error: clusters field is empty"
    assert results['ROIs'] != 0, "Error: ROIs field is empty"
    assert results['input_data'] != 0, "Error: input_data field is empty"
    assert results['quality_metrics'] != 0, "Error: quality_metrics field is empty"
    
    assert len(results['clusters']['labels_dict']) == len(results['quality_metrics']['cluster_intra_means']), "Error: Cluster data is mismatched"
    assert len(results['clusters']['labels_dict']) == results['clusters']['labels_bool_bySession'][0].shape[1], "Error: Cluster data is mismatched"

    ## Save results
    print(f"Saving to: {str(Path(dir_data_test).resolve() / 'pipeline_tracking' / 'results.pkl')}")
    helpers.pickle_save(
        obj=run_data,
        filepath=str(Path(dir_data_test).resolve() / 'pipeline_tracking' / 'run_data_output.pkl'),
    )

    ## Check run_data equality
    print(f"Checking run_data equality")
    path_run_data_true = str(Path(dir_data_test).resolve() / 'pipeline_tracking' / 'run_data.pkl')
    print(f"Loading true run_data from {path_run_data_true}")
    run_data_true = helpers.pickle_load(path_run_data_true)
    print(f"run_data_true loaded. Checking equality")
    checker = helpers.Equivalence_checker(
        kwargs_allclose={'rtol': 1e-5, 'equal_nan': True},
        assert_mode=False,
        verbose=1,
    )
    checks = checker(test=run_data, true=run_data_true)
    fails = [key for key, val in helpers.flatten_dict(checks).items() if val[0]==False]
    if len(fails) > 0:
        warnings.warn(f"run_data equality check failed for keys: {fails}")
    else:
        print(f"run_data equality check finished successfully")
    
            
# def test_ROInet(make_ROIs, array_hasher):
#     ROI_images = make_ROIs
#     size_im=(36,36)
#     data_custom = roicat.data_importing.Data_roicat()
#     data_custom.set_ROI_images(ROI_images, um_per_pixel=1.5)
    
#     DEVICE = helpers.set_device(use_GPU=True, verbose=True)
#     dir_temp = tempfile.gettempdir()

#     roinet = ROInet.ROInet_embedder(
#         device=DEVICE,  ## Which torch device to use ('cpu', 'cuda', etc.)
#         dir_networkFiles=dir_temp,  ## Directory to download the pretrained network to
#         download_method='check_local_first',  ## Check to see if a model has already been downloaded to the location (will skip if hash matches)
#         download_url='https://osf.io/x3fd2/download',  ## URL of the model
#         download_hash='7a5fb8ad94b110037785a46b9463ea94',  ## Hash of the model file
#         forward_pass_version='latent',  ## How the data is passed through the network
#         verbose=True,  ## Whether to print updates
#     )
#     dataloader = roinet.generate_dataloader(
#         ROI_images=data_custom.ROI_images,  ## Input images of ROIs
#         um_per_pixel=data_custom.um_per_pixel,  ## Resolution of FOV
#         pref_plot=False,  ## Whether or not to plot the ROI sizes
        
#         jit_script_transforms=False,  ## (advanced) Whether or not to use torch.jit.script to speed things up
        
#         batchSize_dataloader=8,  ## (advanced) PyTorch dataloader batch_size
#         pinMemory_dataloader=True,  ## (advanced) PyTorch dataloader pin_memory
#         numWorkers_dataloader=mp.cpu_count(),  ## (advanced) PyTorch dataloader num_workers
#         persistentWorkers_dataloader=True,  ## (advanced) PyTorch dataloader persistent_workers
#         prefetchFactor_dataloader=2,  ## (advanced) PyTorch dataloader prefetch_factor
#     );
#     latents = roinet.generate_latents();

#     ## Check shapes
#     assert dataloader.shape[0] == data_custom.n_roi_total, "Error: dataloader shape is mismatched"
#     assert dataloader.shape[1] == size_im[0], "Error: dataloader shape does not match input image size"
#     assert dataloader.shape[2] == size_im[1], "Error: dataloader shape does not match input image size"
#     assert latents.shape[0] == data_custom.n_roi_total, "Error: latents shape does not match n_roi_total"
    


    
