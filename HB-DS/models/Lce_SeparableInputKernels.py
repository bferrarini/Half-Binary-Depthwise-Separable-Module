'''
Created on 20 Feb 2022

@author: Bruno Ferrarini
@affiliation Univeristy of Essex, UK


'''

import config

import models.abstract
import larq as lq
import tensorflow as tf
from prettytable import PrettyTable
import tensorflow.keras as keras

class SeparableInputKernel(models.abstract.ModelWrapper):
    
    def __init__(self, 
                 model_name, 
                 working_dir, 
                 nClasses, 
                 fc_units = 512, 
                 #filters = (96,256,256), 
                 # filters, k, stride, depth_multi)
                 layers = ( (96, 11, 4, 1), (256, 5, 1, 1), (256, 3, 1, 1) ),
                 enable_history = False, 
                 activation_precision : int = 1, 
                 kernel_precision : int = 1, 
                 model_name_2 = "checkpoint", 
                 logger_lvl = config.log_lvl, 
                 
                 l_rate = 1e-4,
                 min_l_rete = 1e-5,
                 patience = 10,
                 factor = 0.5,
                 first_bias = False,
                 spp = None,
                 single_fc = False,
                 no_fc = False,
                 
                 **kwargs):
        
        
        self.layers = layers
        super(SeparableInputKernel, self).__init__(model_name, working_dir, model_name_2, logger_lvl = logger_lvl, enable_history = enable_history,**kwargs)    
        
        #Set specific callbacs
        self.nClasses = nClasses
        self.l_rate = l_rate
        self.activation_precision = activation_precision
        self.kernel_precision = kernel_precision 
        self.units = fc_units
        self.l_rate = l_rate
        self.min_lr = min_l_rete
        self.patience = patience
        self.factor = factor
        self.spp = spp
        self.single_fc = single_fc
        self.headless = no_fc
        self.first_bias = first_bias
        
        self.model = self._setup_model(verbose = False) 
        
        

    '''
        Creates the Keras Model.
    '''
    
    def _setup_model(self, **kwargs):
        
        if "verbose" in kwargs:
            verbose = kwargs['verbose']
        else:
            verbose = False
        
        #tf.debugging.set_log_device_placement(True)
        
        self.model_name = self.model_name if self.model_name is not None else 'BinaryVGG16'
        
        input_img = keras.layers.Input(shape = (227, 227, 3))

        cnn = self._cnn(input_tensor = input_img)
            
        net = self._fully_connected(self.nClasses, cnn,
                        units = self.units)
    
        model = keras.models.Model(input_img, net)
        
        if self.min_lr < self.l_rate:
            reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=self.factor, patience=self.patience, verbose=1, min_delta=5e-4, min_lr=self.min_lr)
            self._add_callbacks(reduce_lr)
            
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.l_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07,
            amsgrad=False,
            name='Adam'
            )
    
        model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
        
        self.model = model

        if verbose:
            self._display_layers()
        
        return self.model
    
    '''
        Returns the CNN part of the Model
    '''
    def _cnn(self,input_tensor=None, input_shape=None,              
                use_bias = False,
                momentum = 0.9):
    

        img_input = keras.layers.Input(shape=input_shape) if input_tensor is None else (
            keras.layers.Input(tensor=input_tensor, shape=input_shape) if not keras.backend.is_keras_tensor(input_tensor) else input_tensor
        )
        
        
        i_quant = 'ste_sign' if self.activation_precision == 1 else None
        k_quant = 'ste_sign' if self.kernel_precision == 1 else None
        
        # Block 1
        hyp = self.layers[0]
        x = lq.layers.QuantDepthwiseConv2D(
                        kernel_size = (hyp[1], hyp[1]),
                        strides = (hyp[2], hyp[2]),
                        padding = "valid",
                        pad_values= 1.0,
                        depth_multiplier=hyp[3],
                        data_format=None,
                        activation=None,
                        use_bias=self.first_bias,
                        input_quantizer=None,
                        depthwise_quantizer=None,
                        depthwise_initializer="glorot_uniform",
                        bias_initializer="zeros",
                        depthwise_regularizer=None,
                        bias_regularizer=None,
                        activity_regularizer=None,
                        depthwise_constraint=None,
                        bias_constraint=None,
                        name = "Depth_1"
                    ) (img_input)  
                    
        x = keras.layers.BatchNormalization(name = 'BN_1_intra', momentum = momentum)(x)
                    
        
        x = lq.layers.QuantConv2D(hyp[0], kernel_size = (1,1), strides = (1,1), padding='valid', name='conv1',
                                          input_quantizer = i_quant,
                                          kernel_quantizer = k_quant,
                                          kernel_constraint = lq.constraints.WeightClip(clip_value=1.0) if self.kernel_precision == 1 else None ,
                                          use_bias = use_bias,
                                          pad_values = 1.0)(x)              
                    
        x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(x)            
        
        #block2
        x = keras.layers.BatchNormalization(name = 'bn2', momentum = momentum)(x)
        
        hyp = self.layers[1]
        x = lq.layers.QuantConv2D(hyp[0], kernel_size = (hyp[1],hyp[1]), strides = (hyp[2], hyp[2]), padding='same', name='conv2',
                                          input_quantizer = 'ste_sign',
                                          kernel_quantizer = 'ste_sign',
                                          kernel_constraint = lq.constraints.WeightClip(clip_value=1.0),
                                          use_bias = use_bias,
                                          pad_values = 1.0)(x)                              
        x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(x)
              
        #block3
        hyp = self.layers[2]
        x = keras.layers.BatchNormalization(name = 'bn3', momentum = momentum)(x)
        x = lq.layers.QuantConv2D(hyp[0], kernel_size = (hyp[1],hyp[1]), strides = (hyp[2], hyp[2]), padding='same', name='conv3',
                                          input_quantizer = 'ste_sign',
                                          kernel_quantizer = 'ste_sign',
                                          kernel_constraint = lq.constraints.WeightClip(clip_value=1.0),
                                          use_bias = use_bias,
                                          pad_values = 1.0)(x)               
                                          
        if self.spp is None:                                                 
            x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool5')(x)     
        else: #No SPP for this release
            #x = SpatialPyramidPooling(pool_list = self.spp, flatten = False, name='pool5')(x) 
            x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool5')(x)       

        x = keras.layers.BatchNormalization(name = 'bnf', momentum = momentum)(x)
        x = keras.layers.Flatten(name='flatten')(x)        
        
        return x
     
     
    '''
        Returns the FC part of the Model
    ''' 
    def _fully_connected(self, nClasses, cnn, units):
        
        x = cnn
        
        if not self.headless:
            #Attenzione, nella rete originale il batchnorm veniva messo a monte della convoluzione
            if not self.single_fc: 
                x = lq.layers.QuantDense(units, name="fc6")(cnn)
                x = keras.layers.BatchNormalization(name='bn6')(x)
                x = keras.layers.Activation('relu', name='act_6')(x)    
        
            x = lq.layers.QuantDense(units, name="fc7")(x)
            x = keras.layers.BatchNormalization(name='bn7')(x)
            x = keras.layers.Activation('relu', name='act_7')(x)     
            
        x = lq.layers.QuantDense(nClasses, name="fc8")(x)
        x = keras.layers.BatchNormalization(name='bn8')(x)
        x = keras.layers.Activation('softmax', name='act_8')(x)
        
        return x

    '''
        Conv block
    '''
    def _conv_block(self, 
                    filters, 
                    name,
                    kernel_size = (3,3), 
                    strides = (1,1), 
                    padding = 'same',
                    pad_value = 1.0,
                    #activation=None, name = None,
                    input_quantizer = 'ste_sign',
                    kernel_quantizer = 'ste_sign',
                    kernel_constraint=lq.constraints.WeightClip(clip_value=1),
                    use_bias = False,
                    batch_norm = True,
                    momentum = 0.9):
        
        def layer_wrapper(inp):
            x = inp
            if batch_norm:
                x = keras.layers.BatchNormalization(name = name + '_bn', momentum = momentum)(x)
            x = lq.layers.QuantConv2D(filters, kernel_size = kernel_size, strides = strides, padding=padding, pad_values = pad_value, name=name,
                                      input_quantizer = input_quantizer,
                                      kernel_quantizer = kernel_quantizer,
                                      kernel_constraint = kernel_constraint,
                                      use_bias = use_bias)(x)
            
            #x = keras.layers.Activation(activation, name = name + '_act')(x)
            return x

        return layer_wrapper

    
    '''
        Dense block
    '''
    def _dense_block(self, units, activation='relu', name='fc1', use_batch_norm = True):

        def layer_wrapper(inp):
            x = lq.layers.QuantDense(units, name=name)(inp)
            if use_batch_norm:
                x = keras.layers.BatchNormalization(name='bn_{}'.format(name))(x)
            x = keras.layers.Activation(activation, name='act_{}'.format(name))(x)
            #x = keras.layers.Dropout(dropout, name='dropout_{}'.format(name))(x)
            return x

        return layer_wrapper  
        

    '''
        Returns a runnable model to extract inner feature. If it not exists, then it is instantiated and added to self.inner_layers
        layer_name: the layer name as defined in the model
    '''
        
    def get_inner_layer_by_name(self, layer_name, k_bit = None, activation = None, flatten = False):
        layer = self.layer_output_by_name(layer_name)
        # Quantized depends on the network. For LarqAlex is ste_sign
        if activation:
            out = keras.layers.Activation(activation)(layer.output)
        else:
            out = layer.output
        if k_bit:
            pass
        if flatten:
            out = keras.layers.Flatten()(out)
        model = tf.keras.models.Model(inputs=self.model.input, outputs = out)
    
        return model
    
    def _display_layers(self):
        c = 0
        t = PrettyTable(['#','Layer','in','out','Trainable'])
        for l in self.model.layers: 
            t.add_row([str(c), l.name, l.input_shape, l.output_shape, str(l.trainable)])
            c += 1
        print(t)         

if __name__ == '__main__':
    
    pass