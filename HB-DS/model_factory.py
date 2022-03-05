'''
Created on 11 Oct 2021

@author: Bruno Ferrarini
@affiliation Univeristy of Essex, UK

'''

import experiment_presets as E
import config
import os

def get_model(
            model_name,
            training_classes,
            l_rate=1e-4, 
            resume = False, 
            model_save_dir = os.path.join('.','output','trained_models'),
            depth_multi = 12,
            min_l_rete = 1e-6,
            patience = 1000,
            factor = 0.5
            ):
    
    
    model = None
    
    if model_name == E.FNet_TRO:

        from models.Lce_HybridShallow import QuantizedHShallow as floppynet_wrapper
    
        #Instantiate a model wrapper
        model = floppynet_wrapper(
            model_name = model_name, 
            working_dir = model_save_dir, 
            model_name_2 = 'model', 
            logger_lvl = config.log_lvl,
            nClasses = training_classes,
            fc_units = 256,
            filters = (96,256,256),
            l_rate = l_rate,
            save_weights_only = True,
            enable_monitoring = True,
            tensorboard = True,
            activation_precision = 1, #BNN
            kernel_precision = 1, #BNN
            enable_history = False,
            clean_history = not resume,
            optimizer = None, #Adam will be used with the l_rate as a learning rate
            loss = 'categorical_crossentropy'
            )
        
    elif model_name == E.FNet_FP:
        
        from models.Lce_HybridShallow import QuantizedHShallow as floppynet_fp_wrapper
    
        #Instantiate a model wrapper
        model = floppynet_fp_wrapper(
            model_name = model_name, 
            working_dir = model_save_dir, 
            model_name_2 = 'model', 
            logger_lvl = config.log_lvl,
            nClasses = training_classes,
            fc_units = 256,
            filters = (96,256,256),
            l_rate = l_rate,
            save_weights_only = True,
            enable_monitoring = True,
            tensorboard = True,
            activation_precision = None, #BNN
            kernel_precision = None, #BNN
            enable_history = False,
            clean_history = not resume,
            optimizer = None, #Adam will be used with the l_rate as a learning rate
            loss = 'categorical_crossentropy'
            )
        
    elif model_name == E.HBDS:
        
        from models.Lce_SeparableInputKernels import SeparableInputKernel as network
        
        #Instantiate a model wrapper
        model = network(
            model_name = f"HB{depth_multi}-FN", 
            working_dir = model_save_dir, 
            model_name_2 = 'model', 
            logger_lvl = config.log_lvl,
            nClasses = training_classes,
            
            fc_units = 256,
            layers = ( (96, 11, 4, depth_multi), (256, 5, 1, 0), (256, 3, 1, 0) ),
            save_weights_only = True,
            enable_monitoring = True,
            tensorboard = True,
            
            activation_precision = 1, 
            kernel_precision = 1,
            enable_history= False,
            clean_history = not resume,
            first_bias = True,
            
            l_rate = l_rate,
            min_l_rete = min_l_rete,
            patience = patience,
            factor = factor,
            
            )
        
 
    
    else:
        raise ValueError(f"Invalid model name (preset): {model_name}")
        
    return model

