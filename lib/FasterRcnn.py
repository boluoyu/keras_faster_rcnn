import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets

from keras.applications import vgg16
from keras import backend as K

from math import floor
import pprint

import matplotlib.image as mpimg
import numpy as np

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten,Lambda,Conv2D,Input
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils.data_utils import get_file
from keras.engine.topology import Layer


WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

anchor_box_scales = [128, 256, 512]
anchor_box_ratio = [[1,1],[1,2],[2,1]]

def vgg16(input_tensor):
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(input_tensor)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    
    return x

def load_vgg_weights(model):
    
    weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
                            WEIGHTS_PATH_NO_TOP,
                            cache_subdir='models')
    
    model.load_weights(weights_path,by_name=True)
    
    return model 


def rpn(layer, num_anchors):
    rpn_conv = Conv2D(512, (3, 1), activation='relu', name='rpn_conv_3x3', padding="same")(layer)
    
    rpn_class =  Conv2D(num_anchors, (1, 1), 
                        activation='sigmoid', 
                        name='rpn_class', 
                        padding="same", 
                        kernel_initializer='uniform')(rpn_conv)
    
    rpn_regr = Conv2D(num_anchors*4, (1, 1), 
                      activation='linear', 
                      name='rpn_regr', 
                      padding="same", 
                      kernel_initializer='zero')(rpn_conv)
    
    return rpn_class, rpn_regr, rpn_conv

class RoiPoolingLayer(Layer):
    '''
        pool out the selected anchor
    '''
    def __init__(self, convLayers, proposalLayer, **kwargs):
        self.convLayers = convLayers
        self.proposalLayer = proposalLayer
        super(ProposalLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        super(MyLayer, self).build(input_shape)  
        
    def call(self,x):
        pass
    
    def compute_output_shape(self, input_shape):
        # return pooled rois
        return (None, 7, 7, 3)


class ProposalLayer(Layer):
    '''
        select the proposal that are valid
    '''
    def __init__(self,rpn_regr, rpn_conv, **kwargs):
        self.rpn_regr = rpn_regr
        self.rpn_conv = rpn_conv
        super(ProposalLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        super(MyLayer, self).build(input_shape)  
        
    def call(self,x):
        # ensure there are two input node to this layer
        assert(len(x) == 2)
        featureMap = x[0]
        rois = x[1]
        
        # take top n proposal
        print(rois.shape)
        
        # apply NMS
        
        # take top n proposal 
        
        
        
        pass
    
    def compute_output_shape(self, input_shape):
        # return selected proposal
        return (None, 4)

def rcnn(convLayers, rpn_regr, rpn_conv, nb_rois, nb_classes= 21, trainable=False):
    #select roi that relavent 
    proposalLayer = ProposalLayer(rpn_regr, rpn_conv)
    # pass the heatmap and roi to roi pooling layer
    roi_pooling = RoiPoolingLayer(convLayers, proposalLayer)
    
    x = Flatten()(roi_pooling)
    x = Dense(4096, name="rcnn_fc6")(x)
    x = Dense(4096, name="rcnn_fc7")(x)
    cls_score = Dense(nb_rois, name="cls_score")(x)
    cls_score = softmax(cls_score)
    
    bbox_pred = Dense(nb_rois * 4, name="bbox_pred")(x)
    
    return cls_score, bbox_pred

TEST_FULL_IMG = mpimg.imread("test1.jpg")
imgs = np.array([TEST_FULL_IMG])

nb_anchors = len(anchor_box_scales) * len(anchor_box_ratio)

print("imgs shape", imgs.shape)

img_input = Input(shape=(None,None,3))

block5_conv3 = vgg16(input_tensor=img_input)
rpn_class, rpn_regr, rpn_conv = rpn(block5_conv3,nb_anchors)

cls_score, bbox_pred = rcnn(block5_conv3, rpn_regr, rpn_conv, 21, 21, trainable=True)

rpn_model = Model(img_input, [rpn_regr,rpn_conv], name='rpn')
fasterRcnn = Model(img_input, [cls_score, bbox_pred], name="fasterRcnn")
model = load_vgg_weights(model)


res = model.predict(imgs)
print("res shape", res.shape)
