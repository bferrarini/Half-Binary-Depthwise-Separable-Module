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
            layers_configuration,
            l_rate=1e-4, 
            resume = False, 
            model_save_dir = os.path.join('.','output','trained_models'),
            min_l_rete = 1e-6,
            patience = 1000,
            factor = 0.5,
            fc_units = 256,
            ):
    
    
    model = None
    
    if model_name in (E.FNet_HB1,
                      E.FNet_HB4,
                      E.FNet_HB8,
                      E.FNet_HB12,
                      E.FNet_HB24,
                      E.FNet_HB48,
                      E.FNet_HB60,
                      ):

        from models.Lce_SeparableInputKernels import SeparableInputKernel as network_wrapper
        
        model = network_wrapper(
            model_name = model_name, 
            working_dir = model_save_dir, 
            model_name_2 = 'model', 
            logger_lvl = config.log_lvl,
            nClasses = training_classes,
            fc_units = fc_units,
            layers = layers_configuration,
            save_weights_only = True,
            enable_monitoring = True,
            tensorboard = True,
            activation_precision = 1, 
            kernel_precision = 1,
            enable_history=True,
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

