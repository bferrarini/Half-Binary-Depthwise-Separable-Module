'''
Created on 9 Oct 2021

@author: Bruno Ferrarini
@affiliation Univeristy of Essex, UK

Experimental presets

'''

import os
import dataset_presets as D
import copy

########################################
### working directory for presets ######
###  Change it for your convenience ####
########################################
model_save_dir = os.path.join('.','output','trained_models')
#################################################################

#model_save_dir = os.path.join(r'C:\Users\main\Documents\eclipse-workspace\FloppyNet_TRO','output','trained_models')


experiments = dict()
#########################
### TRO Paper presets ###
#########################

params = dict()

FNet_HB1 = 'HB1-FN'
params['model_name'] = FNet_HB1 
## TRAINING DATA ###
## CHANGE THE PATHs ACCORDINGLY WITH YOUR NEEDS ##
params['training_data'] = D.training_datasets[D.PLACES365]['training_path']
# Set validation data to None to split the training data
params['validation_data'] = D.training_datasets[D.PLACES365]['validation_path']
# val split is ignored unless validation_data is None
params['val_split'] = 0.4
####################
params['classes'] = D.training_datasets[D.PLACES365]['nClass']
params['l_rate'] = 2e-4
params['min_l_rate'] = 1e-5
params['patience'] = 5 #set larger than epochs to don't use patience.
params['factor'] = 0.5
params['batch_size'] = 32
params['epochs'] = 200
params['model_save_dir'] = model_save_dir
params['out_layer'] = 'pool5'
params['layer_config'] = ( (96, 11, 4, 1), (256, 5, 1, 1), (256, 3, 1, 1) )
#params['module'] = 'operation_modes'
experiments[FNet_HB1] = copy.copy(params)



FNet_HB4 = 'HB4-FN'
params['model_name'] = FNet_HB4 
## TRAINING DATA ###
## CHANGE THE PATHs ACCORDINGLY WITH YOUR NEEDS ##
params['training_data'] = D.training_datasets[D.PLACES365]['training_path']
# Set validation data to None to split the training data
params['validation_data'] = D.training_datasets[D.PLACES365]['validation_path']
# val split is ignored unless validation_data is None
params['val_split'] = 0.4
####################
params['classes'] = D.training_datasets[D.PLACES365]['nClass']
params['l_rate'] = 2e-4
params['min_l_rate'] = 1e-5
params['patience'] = 5 #set larger than epochs to don't use patience.
params['factor'] = 0.5
params['batch_size'] = 32
params['epochs'] = 200
params['model_save_dir'] = model_save_dir
params['out_layer'] = 'pool5'
params['layer_config'] = ( (96, 11, 4, 4), (256, 5, 1, 1), (256, 3, 1, 1) )
#params['module'] = 'operation_modes'
experiments[FNet_HB4] = copy.copy(params)



FNet_HB8 = 'HB8-FN'
params['model_name'] = FNet_HB8 
## TRAINING DATA ###
## CHANGE THE PATHs ACCORDINGLY WITH YOUR NEEDS ##
params['training_data'] = D.training_datasets[D.PLACES365]['training_path']
# Set validation data to None to split the training data
params['validation_data'] = D.training_datasets[D.PLACES365]['validation_path']
# val split is ignored unless validation_data is None
params['val_split'] = 0.4
####################
params['classes'] = D.training_datasets[D.PLACES365]['nClass']
params['l_rate'] = 2e-4
params['min_l_rate'] = 1e-5
params['patience'] = 5 #set larger than epochs to don't use patience.
params['factor'] = 0.5
params['batch_size'] = 32
params['epochs'] = 200
params['model_save_dir'] = model_save_dir
params['out_layer'] = 'pool5'
params['layer_config'] = ( (96, 11, 4, 8), (256, 5, 1, 1), (256, 3, 1, 1) )
#params['module'] = 'operation_modes'
experiments[FNet_HB8] = copy.copy(params)


FNet_HB12 = 'HB12-FN'
params['model_name'] = FNet_HB12 
## TRAINING DATA ###
## CHANGE THE PATHs ACCORDINGLY WITH YOUR NEEDS ##
params['training_data'] = D.training_datasets[D.PLACES365]['training_path']
# Set validation data to None to split the training data
params['validation_data'] = D.training_datasets[D.PLACES365]['validation_path']
# val split is ignored unless validation_data is None
params['val_split'] = 0.4
####################
params['classes'] = D.training_datasets[D.PLACES365]['nClass']
params['l_rate'] = 2e-4
params['min_l_rate'] = 1e-5
params['patience'] = 5 #set larger than epochs to don't use patience.
params['factor'] = 0.5
params['batch_size'] = 32
params['epochs'] = 200
params['model_save_dir'] = model_save_dir
params['out_layer'] = 'pool5'
params['layer_config'] = ( (96, 11, 4, 12), (256, 5, 1, 1), (256, 3, 1, 1) )
#params['module'] = 'operation_modes'
experiments[FNet_HB12] = copy.copy(params)



FNet_HB24 = 'HB24-FN'
params['model_name'] = FNet_HB24 
## TRAINING DATA ###
## CHANGE THE PATHs ACCORDINGLY WITH YOUR NEEDS ##
params['training_data'] = D.training_datasets[D.PLACES365]['training_path']
# Set validation data to None to split the training data
params['validation_data'] = D.training_datasets[D.PLACES365]['validation_path']
# val split is ignored unless validation_data is None
params['val_split'] = 0.4
####################
params['classes'] = D.training_datasets[D.PLACES365]['nClass']
params['l_rate'] = 2e-4
params['min_l_rate'] = 1e-5
params['patience'] = 5 #set larger than epochs to don't use patience.
params['factor'] = 0.5
params['batch_size'] = 32
params['epochs'] = 200
params['model_save_dir'] = model_save_dir
params['out_layer'] = 'pool5'
params['layer_config'] = ( (96, 11, 4, 24), (256, 5, 1, 1), (256, 3, 1, 1) )
#params['module'] = 'operation_modes'
experiments[FNet_HB24] = copy.copy(params)




FNet_HB48 = 'HB48-FN'
params['model_name'] = FNet_HB48 
## TRAINING DATA ###
## CHANGE THE PATHs ACCORDINGLY WITH YOUR NEEDS ##
params['training_data'] = D.training_datasets[D.PLACES365]['training_path']
# Set validation data to None to split the training data
params['validation_data'] = D.training_datasets[D.PLACES365]['validation_path']
# val split is ignored unless validation_data is None
params['val_split'] = 0.4
####################
params['classes'] = D.training_datasets[D.PLACES365]['nClass']
params['l_rate'] = 2e-4
params['min_l_rate'] = 1e-5
params['patience'] = 5 #set larger than epochs to don't use patience.
params['factor'] = 0.5
params['batch_size'] = 32
params['epochs'] = 200
params['model_save_dir'] = model_save_dir
params['out_layer'] = 'pool5'
params['layer_config'] = ( (96, 11, 4, 48), (256, 5, 1, 1), (256, 3, 1, 1) )
#params['module'] = 'operation_modes'
experiments[FNet_HB48] = copy.copy(params)



FNet_HB60 = 'HB60-FN'
params['model_name'] = FNet_HB60 
## TRAINING DATA ###
## CHANGE THE PATHs ACCORDINGLY WITH YOUR NEEDS ##
params['training_data'] = D.training_datasets[D.PLACES365]['training_path']
# Set validation data to None to split the training data
params['validation_data'] = D.training_datasets[D.PLACES365]['validation_path']
# val split is ignored unless validation_data is None
params['val_split'] = 0.4
####################
params['classes'] = D.training_datasets[D.PLACES365]['nClass']
params['l_rate'] = 2e-4
params['min_l_rate'] = 1e-5
params['patience'] = 5 #set larger than epochs to don't use patience.
params['factor'] = 0.5
params['batch_size'] = 32
params['epochs'] = 200
params['model_save_dir'] = model_save_dir
params['out_layer'] = 'pool5'
params['layer_config'] = ( (96, 11, 4, 60), (256, 5, 1, 1), (256, 3, 1, 1) )
#params['module'] = 'operation_modes'
experiments[FNet_HB60] = copy.copy(params)

