from error import RuntimeError
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets.vgg as vgg

class FasterRcnn:
    def __init__(self, config):
        self.config = config
        self._createNetwork()

    def _createNetwork(self):
        ''' create the graph of the network '''
        vgg, end_points = vgg.vgg16(self.config.input_shape, is_training=True)
        vgg_last_conv = end_points["vgg_16/conv5"]
        self.vgg_last_conv = vgg_last_conv
        vgg_variables_to_restore = slim.get_variables_to_restore(exclude=['fc6', 'fc7', 'fc8'])
        self.restorer_cov = tf.train.Saver(vgg_variables_to_restore)



        # init. session
        self.sess = tf.Session()

        pass

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
