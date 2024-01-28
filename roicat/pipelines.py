## Import basic libraries
from pathlib import Path
import copy
import tempfile
from IPython.display import display

# import matplotlib.pyplot as plt
import numpy as np

## Import roicat submodules
from . import data_importing, ROInet, helpers, util, tracking, classification

def pipeline_tracking(params: dict):
    """
    Pipeline for tracking ROIs across sessions.
    RH 2023

    Args:
        params (dict):
            Dictionary of parameters. See
            ``roicat.util.get_default_parameters(pipeline='tracking')`` for
            details.

    Returns:
        (tuple): tuple containing:
            results (dict):
                Dictionary of results.
            run_data (dict):
                Dictionary containing the different class objects used in the
                pipeline.
            params (dict):
                Parameters used in the pipeline. See
                ``roicat.helpers.prepare_params()`` for details.
    """

    ## Prepare params
    defaults = util.get_default_parameters(pipeline='tracking')
    params = helpers.prepare_params(params, defaults, verbose=True)
    display(params)

    ## Prepare state variables
    VERBOSE = params['general']['verbose']
    DEVICE = helpers.set_device(use_GPU=params['general']['use_GPU'])
    SEED = _set_random_seed(
        seed=params['general']['random_seed'],
        deterministic=params['general']['random_seed'] is not None,
    )

    if params['data_loading']['data_kind'] == 'data_suite2p':
        assert params['data_loading']['dir_outer'] is not None, f"params['data_loading']['dir_outer'] must be specified if params['data_loading']['data_kind'] is 'data_suite2p'."
        paths_allStat = helpers.find_paths(
            dir_outer=params['data_loading']['dir_outer'],
            reMatch='stat.npy',
            depth=4,
            find_files=True,
            find_folders=False,
            natsorted=True,
        )[:]
        paths_allOps  = np.array([Path(path).resolve().parent / 'ops.npy' for path in paths_allStat])[:]

        print(f"Found the following stat.npy files:")
        [print(f"    {path}") for path in paths_allStat]
        print(f"Found the following corresponding ops.npy files:")
        [print(f"    {path}") for path in paths_allOps]

        params['data_loading']['paths_allStat'] = paths_allStat
        params['data_loading']['paths_allOps'] = paths_allOps

        ## Import data
        data = data_importing.Data_suite2p(
            paths_statFiles=paths_allStat[:],
            paths_opsFiles=paths_allOps[:],
            verbose=VERBOSE,
            **{**params['data_loading']['common'], **params['data_loading']['data_suite2p']},
        )
        assert data.check_completeness(verbose=False)['tracking'], f"Data object is missing attributes necessary for tracking."
    elif params['data_loading']['data_kind'] == 'roicat':
        paths_allDataObjs = helpers.find_paths(
            dir_outer=params['data_loading']['dir_outer'],
            reMatch=params['data_loading']['data_roicat']['filename_search'],
            depth=1,
            find_files=True,
            find_folders=False,
            natsorted=True,
        )[:]
        assert len(paths_allDataObjs) == 1, f"ERROR: Found {len(paths_allDataObjs)} files matching the search pattern '{params['data_loading']['data_roicat']['filename_search']}' in '{params['data_loading']['dir_outer']}'. Exactly one file must be found."
        
        data = data_importing.Data_roicat()
        data.load(path_load=paths_allDataObjs[0])
    else:
        raise NotImplementedError(f"params['data_loading']['data_kind'] == '{params['data_loading']['data_kind']}' is not yet implemented.")

    ## Alignment
    aligner = tracking.alignment.Aligner(verbose=True)
    FOV_images = aligner.augment_FOV_images(
        ims=data.FOV_images,
        spatialFootprints=data.spatialFootprints,
        **params['alignment']['augment'],
    )
    aligner.fit_geometric(
        ims_moving=FOV_images,  ## input images
        **params['alignment']['fit_geometric'],
    )
    aligner.transform_images_geometric(FOV_images);
    aligner.fit_nonrigid(
        ims_moving=aligner.ims_registered_geo,  ## Input images. Typically the geometrically registered images
        remappingIdx_init=aligner.remappingIdx_geo,  ## The remappingIdx between the original images (and ROIs) and ims_moving
        **params['alignment']['fit_nonrigid'],
    )
    aligner.transform_images_nonrigid(FOV_images);
    aligner.transform_ROIs(
        ROIs=data.spatialFootprints, 
        remappingIdx=aligner.remappingIdx_nonrigid,
        **params['alignment']['transform_ROIs'],
    );


    ## Blur ROIs
    blurrer = tracking.blurring.ROI_Blurrer(
        frame_shape=(data.FOV_height, data.FOV_width),  ## FOV height and width
        plot_kernel=False,  ## Whether to visualize the 2D gaussian
        **params['blurring'],
    )
    blurrer.blur_ROIs(
        spatialFootprints=aligner.ROIs_aligned[:],
    )


    ## ROInet embedding
    dir_temp = tempfile.gettempdir()

    roinet = ROInet.ROInet_embedder(
        device=DEVICE,  ## Which torch device to use ('cpu', 'cuda', etc.)
        dir_networkFiles=dir_temp,  ## Directory to download the pretrained network to
        verbose=VERBOSE,  ## Whether to print updates
        **params['ROInet']['network'],
    )
    roinet.generate_dataloader(
        ROI_images=data.ROI_images,  ## Input images of ROIs
        um_per_pixel=data.um_per_pixel,  ## Resolution of FOV
        pref_plot=False,  ## Whether or not to plot the ROI sizes
        **params['ROInet']['dataloader'],
    );
    roinet.generate_latents();


    ## Scattering wavelet embedding
    swt = tracking.scatteringWaveletTransformer.SWT(
        image_shape=data.ROI_images[0].shape[1:3],  ## size of a cropped ROI image
        device=DEVICE,  ## PyTorch device
        kwargs_Scattering2D=params['SWT']['kwargs_Scattering2D'],
    )
    swt.transform(
        ROI_images=roinet.ROI_images_rs,  ## All the cropped and resized ROI images
        batch_size=params['SWT']['batch_size'],
    );


    ## Compute similarities
    sim = tracking.similarity_graph.ROI_graph(
        frame_height=data.FOV_height,
        frame_width=data.FOV_width,
        verbose=VERBOSE,  ## Whether to print outputs
        **params['similarity_graph']['sparsification']
    )
    s_sf, s_NN, s_SWT, s_sesh = sim.compute_similarity_blockwise(
        spatialFootprints=blurrer.ROIs_blurred,  ## Mask spatial footprints
        features_NN=roinet.latents,  ## ROInet output latents
        features_SWT=swt.latents,  ## Scattering wavelet transform output latents
        ROI_session_bool=data.session_bool,  ## Boolean array of which ROIs belong to which sessions
    #     spatialFootprint_maskPower=1.0,  ##  An exponent to raise the spatial footprints to to care more or less about bright pixels
        **params['similarity_graph']['compute_similarity'],
    );
    sim.make_normalized_similarities(
        centers_of_mass=data.centroids,  ## ROI centroid positions
        features_NN=roinet.latents,  ## ROInet latents
        features_SWT=swt.latents,  ## SWT latents
        device=DEVICE,    
        k_max=data.n_sessions * params['similarity_graph']['normalization']['k_max'],
        k_min=data.n_sessions * params['similarity_graph']['normalization']['k_min'],
        algo_NN=params['similarity_graph']['normalization']['algo_NN'],
    )


    ## Clustering
    clusterer = tracking.clustering.Clusterer(
        s_sf=sim.s_sf,
        s_NN_z=sim.s_NN_z,
        s_SWT_z=sim.s_SWT_z,
        s_sesh=sim.s_sesh,
        verbose=VERBOSE,
    )
    kwargs_makeConjunctiveDistanceMatrix_best = clusterer.find_optimal_parameters_for_pruning(
        seed=SEED,
        **params['clustering']['automatic_mixing'],
    )
    kwargs_mcdm_tmp = kwargs_makeConjunctiveDistanceMatrix_best  ## Use the optimized parameters
    clusterer.make_pruned_similarity_graphs(
        kwargs_makeConjunctiveDistanceMatrix=kwargs_mcdm_tmp,
        **params['clustering']['pruning'],
    )

    def choose_clustering_method(method='automatic', n_sessions_switch=8, n_sessions=None):
        if method == 'automatic':
            method_out = 'hdbscan'.upper() if n_sessions >= n_sessions_switch else 'sequential_hungarian'.upper()
        else:
            method_out = method.upper()
        assert method_out.upper() in ['hdbscan'.upper(), 'sequential_hungarian'.upper()]
        return method_out
    method_clustering = choose_clustering_method(
        method=params['clustering']['cluster_method']['method'],
        n_sessions_switch=params['clustering']['cluster_method']['n_sessions_switch'],
        n_sessions=data.n_sessions,
    )

    if method_clustering == 'hdbscan'.upper():
        labels = clusterer.fit(
            d_conj=clusterer.dConj_pruned,  ## Input distance matrix
            session_bool=data.session_bool,  ## Boolean array of which ROIs belong to which sessions
            **params['clustering']['hdbscan'],
        )
    elif method_clustering == 'sequential_hungarian'.upper():
        labels = clusterer.fit_sequentialHungarian(
            d_conj=clusterer.dConj_pruned,  ## Input distance matrix
            session_bool=data.session_bool,  ## Boolean array of which ROIs belong to which sessions
            **params['clustering']['sequential_hungarian'],
        )
    else:
        raise ValueError('Clustering method not recognized. This should never happen.')

    quality_metrics = clusterer.compute_quality_metrics();

    ## Collect results
    labels_squeezed, labels_bySession, labels_bool, labels_bool_bySession, labels_dict = tracking.clustering.make_label_variants(labels=labels, n_roi_bySession=data.n_roi)

    results = {
        "clusters":{
            "labels": labels_squeezed,
            "labels_bySession": labels_bySession,
            "labels_bool": labels_bool,
            "labels_bool_bySession": labels_bool_bySession,
            "labels_dict": labels_dict,
        },
        "ROIs": {
            "ROIs_aligned": aligner.ROIs_aligned,
            "ROIs_raw": data.spatialFootprints,
            "frame_height": data.FOV_height,
            "frame_width": data.FOV_width,
            "idx_roi_session": np.where(data.session_bool)[1],
            "n_sessions": data.n_sessions,
        },
        "input_data": {
            "paths_stat": data.paths_stat,
            "paths_ops": data.paths_ops,
        },
        "quality_metrics": clusterer.quality_metrics,
    }

    run_data = copy.deepcopy({
        'data': data.serializable_dict,
        'aligner': aligner.serializable_dict,
        'blurrer': blurrer.serializable_dict,
        'roinet': roinet.serializable_dict,
        'swt': swt.serializable_dict,
        'sim': sim.serializable_dict,
        'clusterer': clusterer.serializable_dict,
    })


    ## Visualize results
    print(f'Number of clusters: {len(np.unique(results["clusters"]["labels"]))}')
    print(f'Number of discarded ROIs: {(results["clusters"]["labels"]==-1).sum()}')


    ## Save results
    if params['results_saving']['dir_save'] is not None:

        dir_save = Path(params['results_saving']['dir_save']).resolve()
        name_save = params['results_saving']['prefix_name_save']

        path_save = dir_save / (name_save + '.ROICaT.tracking.results' + '.pkl')
        print(f'path_save: {path_save}')

        helpers.pickle_save(
            obj=results,
            filepath=path_save,
            mkdir=True,
        )

        helpers.pickle_save(
            obj=run_data,
            filepath=str(dir_save / (name_save + '.ROICaT.tracking.rundata' + '.pkl')),
            mkdir=True,
        )
    
    return results, run_data, params

def _set_random_seed(seed=None, deterministic=False):
    """
    Set random seed for reproducibility.
    RH 2023

    Args:
        seed (int, optional):
            Random seed.
            If None, a random seed (spanning int32 integer range) is generated.
        deterministic (bool, optional):
            Whether to make packages deterministic.

    Returns:
        (int):
            seed (int):
                Random seed.
    """
    ### random seed (note that optuna requires a random seed to be set within the pipeline)
    import numpy as np
    seed = int(np.random.randint(0, 2**31 - 1, dtype=np.uint32)) if seed is None else seed

    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    import random
    random.seed(seed)
    import cv2
    cv2.setRNGSeed(seed)

    ## Make torch deterministic
    torch.use_deterministic_algorithms(deterministic)
    ## Make cudnn deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    
    return seed