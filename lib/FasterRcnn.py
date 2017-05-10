import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets
from math import floor
import pprint

def poposal_layer_func(x):
  return []

def roi_pooling_layer_func(x):
  return []

class FasterRcnn:
    def __init__(self, config, train=True):
        self.config = config
        self.train = train
        self._createNetwork()

    # in order to reduce complexity, one support one img
    def _createNetwork(self):
        ''' create the graph of the network '''

        width = self.config.input_shape[0]
        hieght = self.config.input_shape[1]

        print(self.config.input_shape)
        img = tf.placeholder("float", shape=self.config.input_shape)


        vgg, end_points = nets.vgg.vgg_16(img, num_classes=30, is_training=True)
        # print(end_points)
        vgg_last_conv = end_points["vgg_16/conv5/conv5_1"]
        self.vgg_last_conv = vgg_last_conv

        # for restore weigths
        vgg_variables_to_restore = slim.get_variables_to_restore(exclude=['fc6', 'fc7', 'fc8'])
        self.restorer_cov = tf.train.Saver(vgg_variables_to_restore)

        print("vgg_last_conv", vgg_last_conv.get_shape())
        conv3_3 = slim.conv2d(vgg_last_conv, 512, [3, 3], scope='conv3_3')
        print("conv3_3", conv3_3.get_shape())
        feature_map = tf.nn.relu(conv3_3, name="feature_map")

        #  ---- rpn reg
        rpn_reg_conv = tf.nn.conv2d(vgg_last_conv, 36, [1,1], padding='SAME', name="rpn_reg_conv")
        #  the shape of the anchor box is [x, y, anchorId, xywh]
        rpn_bbox_reg = tf.nn.reshape(rpn_reg_conv,[floor(width/4), floor(hieght/4), 9, 4])

        # ----- rpn class
        rpn_cls_conv = tf.nn.conv2d(vgg_last_conv, 18, [1,1], padding='SAME', name="rpn_cls_conv")
        rpn_bbox_cls = tf.nn.reshape(rpn_reg_conv,[floor(width/4), floor(hieght/4), 9, 2])

        # ----- rpn poposal layer
        # This layer do the selection of the anchor, output => [None, 4]
        rpn_poposal_layer = tf.py_func(poposal_layer_func,[rpn_bbox_reg, rpn_bbox_cls], [tf.float32, tf.float32], tf.float32)

        # rcnn
        # Each roi is 14X14X512
        # output is None,None,14,14,512
        poi_pooling = tf.py_func(roi_pooling_layer_func, [rpn_bbox_cls, rpn_poposal_layer], [ tf.float32, tf.float32])
        fc6 = slim.fully_connected(poi_pooling, 4069, scope='fc/fc_6')
        fc7 = slim.fully_connected(fc6, 4069, scope='fc/fc_7')

        #rcnn cls
        rcnn_cls_fc = slim.fully_connected(fc7, self.config.nb_classes, scope='rcnn_cls_fc')
        rcnn_cls = tf.contrib.layers.softmax(rcnn_cls_fc)

        #rcnn reg
        #predict x,y,w,h
        rcnn_reg_fc = slim.fully_connected(fc7, 4, scope='rcnn_reg_fc')

        # init. session
        self.sess = tf.Session()

    def initConvLayer(self):
        self.restorer_cov.restore(self.sess, "/tmp/model.ckpt")
        raise RuntimeError('Not yet implement error')

    def initRandomRpn(self):
        raise RuntimeError('Not yet implement error')

    def trainRpn(self, generator):
        raise RuntimeError('Not yet implement error')

    def saveRpn(self):
        raise RuntimeError('Not yet implement error')

    def trainDetector(self):
        raise RuntimeError('Not yet implement error')

    def saveDetector(self):
        raise RuntimeError('Not yet implement error')

    def saveConv(self):
        raise RuntimeError('Not yet implement error')

    def trainRpn(self):
        raise RuntimeError('Not yet implement error')

    def save(self):
        raise RuntimeError('Not yet implement error')

    def load(self):
        raise RuntimeError('Not yet implement error')

    def loadRpn(self):
        raise RuntimeError('Not yet implement error')

    def loadDetector(self):
        raise RuntimeError('Not yet implement error')

    def loadConv(self):
        raise RuntimeError('Not yet implement error')

    def predict(self):
        '''
            predict the bounding box and the class
            Return:
                [
                    {
                        classId: int,
                        className: string,
                        x: int,
                        y: int,
                        w: int,
                        h: int
                    },
                    ...
                ]
        '''
        # check the img. size and resize
        raise RuntimeError('Not yet implement error')

    def preview(self, img, predictedResult=[]):
        '''
            img: the rbg img
            predictedResult: the result reutrn by predict
        '''
        raise RuntimeError('Not yet implement error')
    def release(self):
        self.sess.close()
