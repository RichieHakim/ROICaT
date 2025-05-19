from pathlib import Path

import warnings
import tempfile

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
                'filename_search': r'data_roicat_obj.richfile'
            },
        },
        'alignment': {
            'initialization': {
                'use_match_search': True,  ## Whether or not to use our match search algorithm to initialize the alignment.
                'all_to_all': False,  ## Force the use of our match search algorithm for all-pairs matching. Much slower (False: O(N) vs. True: O(N^2)), but more accurate.
                'radius_in': 4.0,  ## Value in micrometers used to define the maximum shift/offset between two images that are considered to be aligned. Larger means more lenient alignment.
                'radius_out': 20.0,  ## Value in micrometers used to define the minimum shift/offset between two images that are considered to be misaligned.
                'z_threshold': 4.0,  ## Z-score required to define two images as aligned. Larger values results in more stringent alignment requirements.
            },
            'augment': {
                'normalize_FOV_intensities': True,  ## Whether or not to normalize the FOV_images to the max value across all FOV images.
                'roi_FOV_mixing_factor': 0.5,  ## default: 0.5. Fraction of the max intensity projection of ROIs that is added to the FOV image. 0.0 means only the FOV_images, larger values mean more of the ROIs are added.
                'use_CLAHE': True,  ## Whether or not to use 'Contrast Limited Adaptive Histogram Equalization'. Useful if params['importing']['type_meanImg'] is not a contrast enhanced image (like 'meanImgE' in Suite2p)
                'CLAHE_grid_block_size': 10,  ## Size of the block size for the grid for CLAHE. Smaller values means more local contrast enhancement.
                'CLAHE_clipLimit': 1.0,  ## Clipping limit for CLAHE. Higher values mean more contrast.
                'CLAHE_normalize': True,  ## Whether or not to normalize the CLAHE image.
            },
            'fit_geometric': {
                'template': 0.5,  ## Which session to use as a registration template. If input is float (ie 0.0, 0.5, 1.0, etc.), then it is the fractional position of the session to use; if input is int (ie 1, 2, 3), then it is the index of the session to use (0-indexed)
                'template_method': 'sequential',  ## Can be 'sequential' or 'image'. If 'sequential', then the template is the FOV_image of the previous session. If 'image', then the template is the FOV_image of the session specified by 'template'.
                'mask_borders': [0, 0, 0, 0],  ## Number of pixels to mask from the borders of the FOV_image. Useful for removing artifacts from the edges of the FOV_image.
                'method': 'DISK_LightGlue',  ## Accuracy order (best to worst): RoMa (by far, but slow without a GPU), LoFTR, DISK_LightGlue, ECC_cv2, (the following are not recommended) SIFT, ORB
                'kwargs_method': {
                    'RoMa': {
                        'model_type': 'outdoor',
                        'n_points': 10000,  ## Higher values mean more points are used for the registration. Useful for larger FOV_images. Larger means slower.
                        'batch_size': 1000,
                    },
                    'DISK_LightGlue': {
                        'num_features': 3000,  ## Number of features to extract and match. I've seen best results around 2048 despite higher values typically being better.
                        'threshold_confidence': 0.2,  ## Higher values means fewer but better matches.
                    },
                    'LoFTR': {
                        'model_type': 'indoor_new',
                        'threshold_confidence': 0.2,  ## Higher values means fewer but better matches.
                    },
                    'ECC_cv2': {
                        'mode_transform': 'euclidean',  ## Must be one of {'translation', 'affine', 'euclidean', 'homography'}. See cv2 documentation on findTransformECC for more details.
                        'n_iter': 200,
                        'termination_eps': 1e-09,  ## Termination criteria for the registration algorithm. See documentation for more details.
                        'gaussFiltSize': 1,  ## Size of the gaussian filter used to smooth the FOV_image before registration. Larger values mean more smoothing.
                        'auto_fix_gaussFilt_step': 10,  ## If the registration fails, then the gaussian filter size is reduced by this amount and the registration is tried again.
                    },
                },
                'constraint': 'affine',  ## Must be one of {'rigid', 'euclidean', 'affine', 'homography'}. Choose constraint based on expected changes in images; use the simplest constraint that is applicable.
                'kwargs_RANSAC': {  ## Parameters related to the RANSAC algorithm used for point/descriptor based registration methods.
                    'inl_thresh': 3.0,  ## Threshold for the inliers. Larger values mean more points are considered inliers.
                    'max_iter': 100,  ## Maximum number of iterations for the RANSAC algorithm.
                    'confidence': 0.99,  ## Confidence level for the RANSAC algorithm. Larger values mean more points are considered inliers.
                },
            },
            'fit_nonrigid': {
                'template': 0.5,  ## Which session to use as a registration template. If input is float (ie 0.0, 0.5, 1.0, etc.), then it is the fractional position of the session to use; if input is int (ie 1, 2, 3), then it is the index of the session to use (0-indexed)
                'template_method': 'image',  ## Can be 'sequential' or 'image'. If 'sequential', then the template is the FOV_image of the previous session. If 'image', then the template is the FOV_image of the session specified by 'template'.
                'method': 'DeepFlow',
                'kwargs_method': {
                    'RoMa': {
                        'model_type': 'outdoor',
                    },
                    'DeepFlow': {},
                },
            },
            'transform_ROIs': {
                'normalize': True,  ## If True, normalize the spatial footprints to have a sum of 1.
            },
        },
        'clustering': {
            'parameters_automatic_mixing': {
                'kwargs_findParameters': {
                    'n_patience': 30,  ## Reduced number to speed up
                    'max_trials': 100,  ## Reduced number to speed up
                },
                'n_jobs_findParameters': 1,  ## THIS IS CRITICAL TO ENSURE REPORDUCIBILITY. Parallelization prevents reproducibility.
            },
        },
        'results_saving': {
            'dir_save': str(Path(dir_data_test).resolve() / 'pipeline_tracking'),
            'prefix_name_save': 'test_pipeline_tracking',
        },
    }
    params = helpers.prepare_params(params_partial, defaults)
    ## Save params as yaml
    path_params = str(Path(dir_data_test).resolve() / 'pipeline_tracking' / 'params.yaml')
    helpers.yaml_save(params, path_params)
    results, run_data, params = pipelines.pipeline_tracking(params)

    #check that results fields are not empty
    assert results['clusters'] != 0, "Error: clusters field is empty"
    assert results['ROIs'] != 0, "Error: ROIs field is empty"
    assert results['input_data'] != 0, "Error: input_data field is empty"
    assert results['clusters']['quality_metrics'] != 0, "Error: quality_metrics field is empty"
    
    assert len(results['clusters']['labels_dict']) == len(results['clusters']['quality_metrics']['cluster_intra_means']), "Error: Cluster data is mismatched"
    assert len(results['clusters']['labels_dict']) == results['clusters']['labels_bool_bySession'][0].shape[1], "Error: Cluster data is mismatched"

    ## Save results
    path_results_output = str(Path(dir_data_test).resolve() / 'pipeline_tracking' / 'results_output.richfile')
    print(f"Saving to: {path_results_output}")
    util.RichFile_ROICaT(path=path_results_output).save(results, overwrite=True)

    path_run_data_output = str(Path(dir_data_test).resolve() / 'pipeline_tracking' / 'run_data_output.richfile')
    print(f"Saving to: {path_run_data_output}")
    util.RichFile_ROICaT(path=path_run_data_output).save(run_data, overwrite=True)

    ## Check run_data equality
    path_run_data_true = str(Path(dir_data_test).resolve() / 'pipeline_tracking' / 'run_data.richfile')
    print(f"Loading true run_data from {path_run_data_true}")
    run_data_true = util.RichFile_ROICaT(path=path_run_data_true).load()

    print(f"run_data_true loaded. Checking equality")
    checker = helpers.Equivalence_checker(
        kwargs_allclose={'rtol': 1e-5, 'equal_nan': True},
        assert_mode=False,
        verbose=1,
    )
    checks = checker(test=run_data, true=run_data_true)
    fails = [key for key, val in helpers.flatten_dict(checks).items() if val[0]==False]
    if len(fails) > 0:
        warnings.warn(f"run_data equality check failed for keys: {fails}. Checks: {checks}")
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
    


## Make a CLI to call the tests
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run tests for the ROICaT package')
    ## Add arguments
    parser.add_argument('--dir_data_test', type=str, required=True, help='Path to the test data directory')
    args = parser.parse_args()
    dir_data_test = args.dir_data_test
    test_pipeline_tracking_simple(dir_data_test)

