'''
Created on 9 Oct 2021

@author: Bruno Ferrarini
@affiliation Univeristy of Essex, UK

'''

debug = False

import config.argparsing as argparser
from experiment_presets import experiments
#import importlib
import os
import operation_modes as opm

if __name__ == '__main__':
    
    args = argparser.argparser()
    if debug:
        argparser.test_argparser(args)
    
    if args.cpu_only:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        print("CPU ONLY, SET.")
        
    #TRAINING
    if args.mode==argparser.TRAINING:
        if args.tr_preset is None:
            print('Type a valid value for --preset')
        params = experiments[args.tr_preset]
        #train_module = importlib.import_module(params['module'])
        opm.train(
                model_name = params['model_name'],
                epochs = params['epochs'],
                training_data = params['training_data'],
                training_classes = params['classes'],
                validation_data = params['validation_data'],
                val_split = params['val_split'],
                batch_size = params['batch_size'],
                l_rate = params['l_rate'],
                depth_multi = args.depth_multi,
                min_l_rete = params['min_l_rate'],
                patience = params['patience'],
                factor = params['factor'],
                model_save_dir = params['model_save_dir'] if args.models_save_dir is None else args.models_save_dir 
            )
    # MODEL EXPORT
    elif args.mode==argparser.EXPORT:
        if args.tr_preset is None:
            print('Type a valid value for --preset')
        params = experiments[args.tr_preset]
        #export_module = importlib.import_module(params['module'])
        
        opm.export(
            model_name = params['model_name'],
            out_layer = params['out_layer'],
            training_classes = params['classes'],
            out_dir = os.path.join(params['model_save_dir'], params['model_name'], 'export'),
            model_format = args.export_format,
            model_save_dir = params['models_save_dir'] if args.models_save_dir is None else args.models_save_dir
            )        
        
    # IMAGE DESCRIPTOR
    elif args.mode==argparser.DESCRIPTOR:
        if not args.h5_fn is None:         
            opm.descriptor_from_h5(args.target_images, args.features_out, args.h5_fn, verb = True)
        elif not args.tr_preset is None:    
            params = experiments[args.tr_preset]
            #feature_module = importlib.import_module(params['module'])
            opm.descriptor(
                    model_name = params['model_name'],
                    training_classes = params['classes'], #for loading the model
                    images = args.target_images, 
                    out_file = args.features_out,
                    out_layer = params['out_layer'],
                    verb = True,
                    model_save_dir = params['model_save_dir'] if args.models_save_dir is None else args.models_save_dir
                    )
        else:
            print('Not a valid h5 model or preset have given.')
            
    # IMAGE PAIR DISTANCE
    elif args.mode == argparser.DISTANCE:
        
        if not args.h5_fn is None:
            # verb = True -> distance in console
            d = opm.compute_pair_distance(
                    h5_fn = args.h5_fn,
                    image1 = args.image1, 
                    image2 = args.image2, 
                    dist_type = args.dtype,
                    verb = True)
        elif not args.tr_preset is None: 
            params = experiments[args.tr_preset]
            # verb = True -> distance in console
            d = opm.compute_pair_distance(
                    image1 = args.image1, 
                    image2 = args.image2, 
                    model_name = params['model_name'],
                    training_classes = params['classes'],
                    out_layer = params['out_layer'],
                    model_save_dir = params['model_save_dir'] if args.models_save_dir is None else args.models_save_dir,
                    dist_type = args.dtype,
                    verb = True                
                )
        else: 
            print('Not a valid h5 model or preset have given.')
        

    
    # QUERY A DATASET
    elif args.mode==argparser.QUERY:
        
        if not args.h5_fn is None:
            
            m = opm.query_a_dataset(
                    h5_fn = args.h5_fn,
                    query_image = args.image1, 
                    reference_data_dir = args.reference_dataset_dir, 
                    dist_type = args.dtype,
                    verb = True,        
                    top = args.top_n
                )
        
        elif not args.tr_preset is None: 
            m = opm.query_a_dataset(
                    query_image = args.image1, 
                    reference_data_dir = args.reference_dataset_dir, 
                    top = args.top_n,
                    model_name = params['model_name'],
                    training_classes = params['classes'],
                    out_layer = params['out_layer'],
                    model_save_dir = params['model_save_dir'] if args.models_save_dir is None else args.models_save_dir,
                    dist_type = args.dtype,
                    verb = True    
                )
        else: 
            print('Not a valid h5 model or preset have given.')
        
        
    # ACCURACY ON A DATASET
    elif args.mode == argparser.ACCURACY:
        pass