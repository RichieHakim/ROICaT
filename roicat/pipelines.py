## Import basic libraries
from pathlib import Path
import copy
import tempfile
from IPython.display import display
import time

# import matplotlib.pyplot as plt
import numpy as np

## Import roicat submodules
from . import data_importing, ROInet, helpers, util, visualization, tracking, classification

def pipeline_tracking(params: dict) -> tuple:
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
    ## Start timer
    tic_start = time.time()

    ## Prepare params
    defaults = util.get_default_parameters(pipeline='tracking')
    params = helpers.prepare_params(params, defaults, verbose=True)
    display(params)

    ## Prepare state variables
    VERBOSE = params['general']['verbose']
    DEVICE = helpers.set_device(use_GPU=params['general']['use_GPU'])
    SEED = util.set_random_seed(
        seed=params['general']['random_seed'],
        deterministic=params['general']['random_seed'] is not None,
    )

    
    if params['data_loading']['data_kind'] == 'suite2p':
        assert params['data_loading']['dir_outer'] is not None, f"params['data_loading']['dir_outer'] must be specified if params['data_loading']['data_kind'] is 'suite2p'."
        paths_allStat = helpers.find_paths(
            dir_outer=params['data_loading']['dir_outer'],
            reMatch='stat.npy',
            depth=6,
            find_files=True,
            find_folders=False,
            natsorted=True,
        )[:]
        paths_allOps  = [str(Path(path).resolve().parent / 'ops.npy') for path in paths_allStat][:]

        if len(paths_allStat) == 0:
            raise FileNotFoundError(f"No stat.npy files found in '{params['data_loading']['dir_outer']}'")
        print(f"Found the following stat.npy files:")
        [print(f"    {path}") for path in paths_allStat]
        print(f"Found the following corresponding ops.npy files:")
        [print(f"    {path}") for path in paths_allOps]


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
            find_files=False,
            find_folders=True,
            natsorted=True,
        )[:]
        assert len(paths_allDataObjs) == 1, f"ERROR: Found {len(paths_allDataObjs)} files matching the search pattern '{params['data_loading']['data_roicat']['filename_search']}' in '{params['data_loading']['dir_outer']}'. Exactly one file must be found."
        
        data = data_importing.Data_roicat()
        # data.load(path_load=paths_allDataObjs[0])

        data.import_from_dict(
            dict_load=util.RichFile_ROICaT(path=paths_allDataObjs[0]).load(),
            )
    elif params['data_loading']['data_kind'] == 'custom':
        data = params['data_loading']['data_custom']
    else:
        raise NotImplementedError(f"params['data_loading']['data_kind'] == '{params['data_loading']['data_kind']}' is not yet implemented.")

    assert data.check_completeness(verbose=False)['tracking'], f"Data object is missing attributes necessary for tracking."
    assert data.n_sessions > 1, f"Data object must have more than one session to track (n_sessions={data.n_sessions})."


    ## Alignment
    aligner = tracking.alignment.Aligner(
        um_per_pixel=data.um_per_pixel[0],  ## Single value for um_per_pixel. data.um_per_pixel is typically a list of floats, so index out just one value.
        verbose=VERBOSE,  ## Whether to print updates
        device=DEVICE,
        **params['alignment']['initialization']
    )
    FOV_images = aligner.augment_FOV_images(
        FOV_images=data.FOV_images,
        spatialFootprints=data.spatialFootprints,
        **params['alignment']['augment'],
    )
    aligner.fit_geometric(
        ims_moving=FOV_images,  ## input images
        verbose=VERBOSE,  ## Whether to print updates
        **params['alignment']['fit_geometric'],
    )
    aligner.transform_images_geometric(FOV_images);

    if params['alignment']['fit_nonrigid']['method']:
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
    else:
        aligner.transform_ROIs(
            ROIs=data.spatialFootprints, 
            remappingIdx=aligner.remappingIdx_geo,
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
    if params['clustering']['mixing_method'] == 'automatic':
        kwargs_makeConjunctiveDistanceMatrix_best = clusterer.find_optimal_parameters_for_pruning(
            seed=SEED,
            **params['clustering']['parameters_automatic_mixing'],
        )
    elif params['clustering']['mixing_method'] == 'manual':
        kwargs_makeConjunctiveDistanceMatrix_best = params['clustering']['parameters_manual_mixing']
    else:
        ## Not implemented
        raise NotImplementedError(f"Mixing method '{params['clustering']['mixing_method']}' is not implemented. Select from: ['automatic', 'manual']")

    clusterer.make_pruned_similarity_graphs(
        kwargs_makeConjunctiveDistanceMatrix=kwargs_makeConjunctiveDistanceMatrix_best,
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

    results_clusters = {
        'labels': labels_squeezed,
        'labels_bySession': labels_bySession,
        'labels_dict': labels_dict,
        'quality_metrics': quality_metrics,
    }

    results_all = {
        "clusters":{
            "labels": util.JSON_List(labels_squeezed),
            "labels_bySession": util.JSON_List(labels_bySession),
            "labels_bool": labels_bool,
            "labels_bool_bySession": labels_bool_bySession,
            "labels_dict": util.JSON_Dict(labels_dict),
            "quality_metrics": util.JSON_Dict(clusterer.quality_metrics) if hasattr(clusterer, 'quality_metrics') else None,
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
            "paths_stat": data.paths_stat if hasattr(data, 'paths_stat') else None,
            "paths_ops": data.paths_ops if hasattr(data, 'paths_ops') else None,
        },
    }

    run_data = {
        'data': data.__dict__,
        'aligner': aligner.__dict__,
        'blurrer': blurrer.__dict__,
        'roinet': roinet.__dict__,
        'swt': swt.__dict__,
        'sim': sim.__dict__,
        'clusterer': clusterer.__dict__,
    }
    params_used = {name: mod['params'] for name, mod in run_data.items()}

    ## Print some results
    print(f'Number of clusters: {len(np.unique(results_clusters["labels"]))}')
    print(f'Number of discarded ROIs: {(np.array(results_clusters["labels"])==-1).sum()}')

    ## Save results
    if params['results_saving']['dir_save'] is not None:

        dir_save = Path(params['results_saving']['dir_save']).resolve()
        name_save = str(params['results_saving']['prefix_name_save'])

        print(f'dir_save: {dir_save}')

        paths_save = {
            'results_clusters': str(Path(dir_save) / f'{name_save}.tracking.results_clusters.json'),
            'params_used':      str(Path(dir_save) / f'{name_save}.tracking.params_used.json'),
            'results_all':      str(Path(dir_save) / f'{name_save}.tracking.results_all.richfile'),
            'run_data':         str(Path(dir_save) / f'{name_save}.tracking.run_data.richfile'),
        }

        Path(dir_save).mkdir(parents=True, exist_ok=True)

        helpers.json_save(obj=results_clusters, filepath=paths_save['results_clusters']);
        helpers.json_save(obj=params_used, filepath=paths_save['params_used']);
        util.RichFile_ROICaT(path=paths_save['results_all']).save(obj=results_all, overwrite=True);
        util.RichFile_ROICaT(path=paths_save['run_data']).save(obj=run_data, overwrite=True);

    
        ## Visualize results
        ### Save some figures
        
        #### Save FOV_images as .png files
        def save_image(array, path, normalize=True):
            ## Use PIL to save the image
            from PIL import Image
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray((np.array(array / array.max() if normalize else array) * 255).astype(np.uint8)).save(path)
        [save_image(array, str(Path(dir_save).resolve() / 'visualization' / 'FOV_images' / f'FOV_images_{ii}.png') ) for ii, array in enumerate(data.FOV_images)]
        [save_image(array, str(Path(dir_save).resolve() / 'visualization' / 'FOV_images_aligned_geometric' / f'FOV_images_aligned_geometric_{ii}.png') ) for ii, array in enumerate(aligner.ims_registered_geo)]
        if params['alignment']['fit_nonrigid']['method']:
            [save_image(array, str(Path(dir_save).resolve() / 'visualization' / 'FOV_images_aligned_nonrigid' / f'FOV_images_aligned_nonrigid_{ii}.png') ) for ii, array in enumerate(aligner.ims_registered_nonrigid)]
        [save_image(array, str(Path(dir_save).resolve() / 'visualization' / 'ROIs' / f'ROIs_{ii}.png') ) for ii, array in enumerate(data.get_maxIntensityProjection_spatialFootprints())]
        [save_image(array, str(Path(dir_save).resolve() / 'visualization' / 'ROIs_aligned' / f'ROIs_aligned_{ii}.png') ) for ii, array in enumerate(aligner.get_ROIsAligned_maxIntensityProjection(normalize=True))]
        [save_image(array, str(Path(dir_save).resolve() / 'visualization' / 'ROIs_aligned_blurred' / f'ROIs_aligned_blurred_{ii}.png') ) for ii, array in enumerate(blurrer.get_ROIsBlurred_maxIntensityProjection())]
        
        #### Save the image alignment checker images
        fig_all_to_all, fig_direct = aligner.plot_alignment_results_geometric()
        (Path(dir_save).resolve() / 'visualization' / 'alignment').mkdir(parents=True, exist_ok=True)
        fig_all_to_all.savefig(str(Path(dir_save).resolve() / 'visualization' / 'alignment' / 'all_to_all_geometric.png'))
        fig_direct.savefig(str(Path(dir_save).resolve() / 'visualization' / 'alignment' / 'direct_geometric.png')) if fig_direct is not None else None

        if params['alignment']['fit_nonrigid']['method']:
            fig_all_to_all, _ = aligner.plot_alignment_results_nonrigid()
            fig_all_to_all.savefig(str(Path(dir_save).resolve() / 'visualization' / 'alignment' / 'all_to_all_nonrigid.png'))

        #### Save some sample ROI images
        [save_image(array, str(Path(dir_save).resolve() / 'visualization' / 'ROIs_sample' / f'ROIs_sample_{ii}.png') ) for ii, array in enumerate(roinet.ROI_images_rs[:100])]
        
        #### Save the similarity graphy blocks
        fig = sim.visualize_blocks()
        (Path(dir_save).resolve() / 'visualization' / 'similarity_graph').mkdir(parents=True, exist_ok=True)
        fig.savefig(str(Path(dir_save).resolve() / 'visualization' / 'similarity_graph' / 'blocks.png'))
        
        #### Save the similarity / distance plots for the given conjunctive distance matrix kwargs
        fig = clusterer.plot_distSame(kwargs_makeConjunctiveDistanceMatrix=kwargs_makeConjunctiveDistanceMatrix_best)
        (Path(dir_save).resolve() / 'visualization' / 'clustering').mkdir(parents=True, exist_ok=True)
        fig.savefig(str(Path(dir_save).resolve() / 'visualization' / 'clustering' / 'dist.png'))
        fig, axs = clusterer.plot_similarity_relationships(
            plots_to_show=[1,2,3], 
            max_samples=100000,  ## Make smaller if it is running too slow
            kwargs_scatter={'s':1, 'alpha':0.2},
            kwargs_makeConjunctiveDistanceMatrix=kwargs_makeConjunctiveDistanceMatrix_best
        )
        (Path(dir_save).resolve() / 'visualization' / 'clustering').mkdir(parents=True, exist_ok=True)
        fig.savefig(str(Path(dir_save).resolve() / 'visualization' / 'clustering' / 'similarity_relationships.png'))
        
        #### Save the clustering results
        fig, axs = tracking.clustering.plot_quality_metrics(
            quality_metrics=quality_metrics, 
            labels=labels_squeezed, 
            n_sessions=data.n_sessions,
        )
        (Path(dir_save).resolve() / 'visualization' / 'clustering').mkdir(parents=True, exist_ok=True)
        fig.savefig(str(Path(dir_save).resolve() / 'visualization' / 'clustering' / 'quality_metrics.png'))
        
        ### Save a gif of the ROIs
        FOV_clusters = visualization.compute_colored_FOV(
            spatialFootprints=[r.power(1.0) for r in results_all['ROIs']['ROIs_aligned']],  ## Spatial footprint sparse arrays
            FOV_height=results_all['ROIs']['frame_height'],
            FOV_width=results_all['ROIs']['frame_width'],
            labels=results_all["clusters"]["labels_bySession"],  ## cluster labels
        #     labels=(np.array(results["clusters"]["labels"])!=-1).astype(np.int64),  ## cluster labels
        #     alphas_labels=confidence*1.5,  ## Set brightness of each cluster based on some 1-D array
        #     alphas_labels=(clusterer.quality_metrics['cluster_silhouette'] > 0) * (clusterer.quality_metrics['cluster_intra_means'] > 0.4),
        #     alphas_sf=clusterer.quality_metrics['sample_silhouette'],  ## Set brightness of each ROI based on some 1-D array
        )
        helpers.save_gif(
            array=helpers.add_text_to_images(
                images=[(f * 255).astype(np.uint8) for f in FOV_clusters], 
                text=[[f"{ii}",] for ii in range(len(FOV_clusters))], 
                font_size=3,
                line_width=10,
                position=(30, 90),
            ), 
            path=str(Path(dir_save).resolve() / 'visualization' / 'FOV_clusters.gif'),
            frameRate=10.0,
            loop=0,
        )

        ### Save gifs of the FOVs at different stages of alignment
        helpers.save_gif(
            array=helpers.add_text_to_images(
                images=[((f / np.max(f)) * 255).astype(np.uint8) for f in FOV_images], 
                text=[[f"{ii}",] for ii in range(len(FOV_clusters))], 
                font_size=3,
                line_width=10,
                position=(30, 90),
            ), 
            path=str(Path(dir_save).resolve() / 'visualization' / 'FOV_images' / 'FOV_images.gif'),
            frameRate=10.0,
            loop=0,
        )

        helpers.save_gif(
            array=helpers.add_text_to_images(
                images=[((f / np.max(f)) * 255).astype(np.uint8) for f in aligner.ims_registered_geo], 
                text=[[f"{ii}",] for ii in range(len(FOV_clusters))], 
                font_size=3,
                line_width=10,
                position=(30, 90),
            ), 
            path=str(Path(dir_save).resolve() / 'visualization' / 'FOV_images_aligned_geometric' / 'FOV_images_aligned_geometric.gif'),
            frameRate=10.0,
            loop=0,
        )

        if params['alignment']['fit_nonrigid']['method']:
            helpers.save_gif(
                array=helpers.add_text_to_images(
                    images=[((f / np.max(f)) * 255).astype(np.uint8) for f in aligner.ims_registered_nonrigid], 
                    text=[[f"{ii}",] for ii in range(len(FOV_clusters))], 
                    font_size=3,
                    line_width=10,
                    position=(30, 90),
                ), 
                path=str(Path(dir_save).resolve() / 'visualization' / 'FOV_images_nonrigid' / 'FOV_images_nonrigid.gif'),
                frameRate=10.0,
                loop=0,
            )



    ## End timer
    tic_end = time.time()
    print(f"Elapsed time: {tic_end - tic_start:.2f} seconds")
    
    return results_all, run_data, params