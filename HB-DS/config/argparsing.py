'''
Created on 9 Oct 2021

@author: Bruno Ferrarini
@affiliation Univeristy of Essex, UK

'''

import argparse
import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

import experiment_presets as EX

TRAINING = 'training'
DESCRIPTOR = 'descriptor'
EXPORT = 'export'
DISTANCE = 'distance'
QUERY = 'query'
ACCURACY = 'accuracy'

ARM64 = 'arm64'
H5 = 'H5'
ALL = 'all'


def argparser():
    
    parser = argparse.ArgumentParser(description='DEMO: Trains and evaluates FloppyNet')
    
    
    #Common
    parser.add_argument('-M','--mode', choices=[TRAINING, EXPORT, DESCRIPTOR, DISTANCE, QUERY, ACCURACY], dest='mode', required=True, help="Execution mode: training, extraction, matching")
    parser.add_argument('-c','--cpu_only', action='store_true', help="set it for CPU ONLY execution")
    parser.add_argument('-O', '--models_save_dir', type=str, dest='models_save_dir', default=None, required=False, help="where trained models are saved. Default: ./output/trained_models.")
    parser.add_argument('-P','--preset', choices=[EX.FNet_TRO,EX.HBDS], dest='tr_preset', default=None, required=False, help="Training preset")
    
    #Export
    parser.add_argument('-f', '--format', choices=[ARM64, H5, ALL], dest='export_format', default=H5, required=False, help="Model Export Format: lce for the RPI4 or H5")
    
    #Feature extraction: These arguments are used in when mode=='extraction'
    parser.add_argument('-I', '--target_images', dest='target_images', required=False, help="It can be either an image or a directory with multiple images.")    
    parser.add_argument('-D', '--output_features_file', type=str, dest='features_out', default=None, required=False, help="Output file for the features. If None, the features will be sent to stdout.")
    parser.add_argument('-H','--h5_model', type=str, dest='h5_fn', default=None, required=False, help="Model to load. It has priority on the preset for feature extraction.")
    
    #Distance
    parser.add_argument('-x', '--img1', dest='image1', required=False, help="An image file")
    parser.add_argument('-y', '--img2', dest='image2', required=False, help="An image file")
    parser.add_argument('-d', '--dtype', dest='dtype', default='COS', required=False, help="Distance type: COS or L2.")

    #QUERY
    parser.add_argument('-R', '--ref_dataset', dest='reference_dataset_dir', required=False, help="The directory where the reference images are stored.")
    parser.add_argument('-n', '--top_n', dest='top_n', type=int, required=False, help="Retrieves the top N images.")
    
    #HDBS
    parser.add_argument('--depth_multi', dest='depth_multi', type=int, required=False, help="Depth Multiplier for HB-DS")
    
    
    return parser.parse_args()



def test_argparser(args):
    
    for attr in dir(args):
        if not attr.startswith("_"):
            print(f"{attr} = {getattr(args,attr)}")
    
if __name__ == '__main__':
    test_argparser(argparser())