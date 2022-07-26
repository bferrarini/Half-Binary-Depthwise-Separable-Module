'''
Created on 10 Oct 2021

@author: Bruno Ferrarini
@affiliation Univeristy of Essex, UK

'''
#import config
import os
import larq
import larq_compute_engine as lce
from training_utility import train_op
from features_helpers import compute_descriptor, get_descriptor
import model_factory as factory
import experiment_presets as EX
import config.argparsing as AP
from match_utility import distance, get_label, get_matches, load_gt_csv


##############################
# EXEPRIMENT CONFIGURATION ###
##############################
#from models.Lce_HybridShallow import QuantizedHShallow as network

'''
    Train the model
    @param epoch: the number of epochs
    @param training_data: path to the training dataset
    @param training_classes: number of classes in the training dataset
    @param validation_data: path to the training dataset
    @param batch_size: the size of the batch
    @param l_reate: learning rate
    @param resume: True to resume the training. Default is False. 
    @param augment: uses augmentation
    @param backup: backups the old model before starting a new training session

'''

def train(model_name,
          epochs, 
          training_data,
          training_classes,
          layers,
          validation_data = None,
          val_split = 0.4,
          batch_size = 24, 
          l_rate=1e-4, 
          depth_multi = 12,
          min_l_rete = 1e-6,
          patience = 1000,
          factor = 0.5,
          augment = False, 
          resume = False, 
          backup = False,
          model_save_dir = os.path.join('.','output','trained_models'),
          ):
    
    #Instantiate a model wrapper
    model_wrapper = factory.get_model(model_name, training_classes, l_rate, resume, model_save_dir,depth_multi,min_l_rete,patience,factor)
    
    train_op(model_wrapper, 
             model_name = model_name,
             train_dir = training_data,
             val_dir = validation_data,
             val_split = val_split,
             batch_size = batch_size, 
             epochs = epochs, 
             augment = augment,
             resume = resume, backup = backup)
    
    
    
def export(
            model_name,
            training_classes, #for loading the model
            out_layer = 'pool5',
            flatten =  False, #False keeps the output layers as it is. True, returns a flatten feature map
            model_save_dir = os.path.join('.','output','trained_models'),
            out_dir = None,
            model_format = 'H5',
            verb = False
        ):
    
    import tensorflow as tf
    
    # Some of the parameters are unnecessary for exporting. Thus they are set arbitrary
    model_wrapper = factory.get_model(model_name, training_classes = training_classes, l_rate = 10, resume = False, model_save_dir = model_save_dir)
    
    out_dir_ = os.path.join('output','trained_models', model_name, 'export') if out_dir is None else out_dir
    if not os.path.exists(out_dir_):
        os.mkdir(out_dir_)
        print(f"Created {out_dir_}")
        
    model_wrapper.load()
        # import traceback
        # import sys
        # print('Weights not available in the default location')
        # print('Empty model exported\n')
        # print(traceback.format_exc())
        
    sub_model = model_wrapper.get_inner_layer_by_name(out_layer, flatten=flatten)
    if verb:
        larq.models.summary(sub_model) 
    
    
    if model_format == AP.H5 or model_format == AP.ALL:
        fn = os.path.join(out_dir_, model_name + ".h5")
        sub_model.save(fn)
        print(f"{model_name} model saved at {fn}")
        
    #PROTOTYPE for FP precision model deployment
    if model_format == AP.ARM64 or model_format == AP.ALL:
        if model_name == EX.FNet_FP: #TFLITE for regular deployment on ARM
            converter = tf.lite.TFLiteConverter.from_keras_model(sub_model)
            # #no optimization as we want AlexNet as a 32-bit model
            # converter.optimizations = [tf.lite.Optimize.DEFAULT]
            # converter.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]
            tflite_model = converter.convert() 
            
            fn = os.path.join(out_dir_, model_name + ".tflite")
            with open(fn, 'wb') as f:
                f.write(tflite_model)
                print(f"{model_name} model for RPI4 is saved at {fn}")
            
        else: #LCE for BNNs
            fn = os.path.join(out_dir_, model_name + ".tflite")
            with open(fn, "wb") as fb:
                fb_bytes = lce.convert_keras_model(sub_model)
                fb.write(fb_bytes)
                print(f"{model_name} model for RPI4 is saved at {fn}")
        
        
def descriptor(
        model_name,
        training_classes, #for loading the model
        images, 
        out_file,
        out_layer = 'pool5',
        model_save_dir = os.path.join('.','output','trained_models'),
        verb = False,
        ):

    model_wrapper = factory.get_model(model_name, training_classes = training_classes, l_rate = 10, resume = False, model_save_dir = model_save_dir)
    
    model_wrapper.load()      
    
    sub_model = model_wrapper.get_inner_layer_by_name(out_layer, flatten=False)
    if verb:
        larq.models.summary(sub_model)   
         
    compute_descriptor(sub_model, images, out_file, limit_to = None, flatten = False, batch_size = 1)
    
    print(f"Feature file written at {out_file}")


def descriptor_from_h5(
        images, 
        out_file,
        h5_fn,
        verb = False,
        ):
    
    import tensorflow as tf
    
    model = tf.keras.models.load_model(h5_fn)
    
    if verb:
        larq.models.summary(model)   
         
    compute_descriptor(model, images, out_file, limit_to = None, flatten = False, batch_size = 1)
    
    print(f"Feature file written at {out_file}")    
    
    
def compute_pair_distance(
        image1, 
        image2, 
        model_name = None,
        training_classes = None,
        h5_fn = None,
        out_layer = 'pool5',
        model_save_dir = os.path.join('.','output','trained_models'),
        dist_type = "COS",
        verb = True
        ):
    
    # this scripts works on image files:
    if os.path.isfile(image1) and os.path.isfile(image2):
    
        if not h5_fn is None:
            import tensorflow as tf
            model = tf.keras.models.load_model(h5_fn, compile=False)
            
        else:
            model_wrapper = factory.get_model(model_name, training_classes = training_classes, l_rate = 10, resume = False, model_save_dir = model_save_dir)
            model_wrapper.load()      
            model = model_wrapper.get_inner_layer_by_name(out_layer, flatten=False)
              
        
        feat, fn = get_descriptor(model, [image1, image2], flatten = True, batch_size = 2)
        
        d = distance(feat[0], feat[1], dist_type)
        
        # d.size should be always when only 1 pair is compared
        assert d.size == 1
        
        d = d[0][0] 
        
        if verb:
            print("Image 1: {}".format(fn[0]))
            print("Image 2: {}".format(fn[1]))
            print("{} distance: {}".format(dist_type, d))
            
        return d
    
    else:
        raise Exception("ERROR: Both {} and {} must be image files".format(image1, image2))


def query_a_dataset(
        query_image, 
        reference_data_dir, 
        model_name = None,
        training_classes = None,
        h5_fn = None,
        out_layer = 'pool5',
        model_save_dir = os.path.join('.','output','trained_models'),
        dist_type = "COS",
        top = 1,
        verb = True        
        ):
    
    # this scripts works with a query image and directory:
    if os.path.isfile(query_image) and os.path.isdir(reference_data_dir):
    
        if not h5_fn is None:
            import tensorflow as tf
            model = tf.keras.models.load_model(h5_fn, compile=False)
            
        else:
            model_wrapper = factory.get_model(model_name, training_classes = training_classes, l_rate = 10, resume = False, model_save_dir = model_save_dir)
            model_wrapper.load()      
            model = model_wrapper.get_inner_layer_by_name(out_layer, flatten=False)
        
        # compute the reference features. This operation is repeated every time. It is inefficient but it is the most simple setup for this demo.
        print("Computing the reference map from {}".format(reference_data_dir))
        ref_feat, ref_fn = get_descriptor(model, reference_data_dir, flatten = True, batch_size = 20)
        print("Computing the image descriptor for the query image {}".format(query_image))
        query_feat, query_fn = get_descriptor(model, query_image, flatten = True, batch_size = 1)
        
        ref_labels = list(map(get_label, ref_fn))
        query_labels = list(map( get_label, query_fn))
        
        
        m, d = get_matches(query_feat, query_labels, ref_feat, ref_labels, top = top, dist_type = dist_type)
        
        print("")
        # get_maeches() works with an arbitrary number of queries. This example is the special case where the query is only one. Hence, index 0
        print("The top {} matches for {}: ".format(top, query_labels[0]))
        print("Labels: {}".format(m[query_labels[0]]))
        print("{}: {}".format(dist_type, d[query_labels[0]]))
        
        return m
    
    else:
        raise Exception("ERROR: query_image {} MUST be an image files, {} MUST be a directory".format(query_image, reference_data_dir))



        
if __name__ == '__main__':
    
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    

    try:
        d = compute_pair_distance(
                h5_fn= r"TRO_pretrained\floppyNet_places365.h5",
                image1 = r"datasets\nordland_sample\winter\17901.jpg", 
                image2 = r"datasets\nordland_sample\summer\17901.jpg", 
                dist_type = "COS",
                verb = True)
    except Exception as e:
        print(e)
    #

    try:    
        d = query_a_dataset(
                h5_fn= r"TRO_pretrained\floppyNet_places365.h5",
                query_image = r"datasets\nordland_sample\winter\17901.jpg", 
                reference_data_dir = r"datasets\nordland_sample\summer", 
                dist_type = "L2",
                verb = False,        
                top = 5
            )
    except Exception as e:
        print(e)    
    
  