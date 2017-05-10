# -*- coding: utf-8 -*-
#from lib.DataGenerator import DataGenerator
from lib import ConfigObject
from lib import FasterRcnn


# prepare generator
# dataGenerator = DataGenerator()

configObject = ConfigObject.Config()
#configObject.setModelPath(rpn="",conv="", detection="")
#
## create rastRcnnObject
fasterRcnn = FasterRcnn.FasterRcnn(configObject, train=True)
#
## step 1 train RPN
#fasterRcnn.initConvLayer("vgg16Path")
#fasterRcnn.initRandomRpn()
#fasterRcnn.trainRpn(dataGenerator)
#fasterRcnn.saveRpn()
#
## step 2 train detector network
#fasterRcnn.initConvLayer("vgg16Path")
#fasterRcnn.initRandomDetector()
#fasterRcnn.trainDetector(dataGenerator)
#fasterRcnn.saveDetector()
#fasterRcnn.saveConv()
#
## step 3 fine tune rpn
#fasterRcnn.trainRpn(dataGenerator)
#fasterRcnn.saveRpn()
#fasterRcnn.saveConv()
#
## step 4 fine tune detector
#fasterRcnn.trainDetector(dataGenerator)
#fasterRcnn.saveDetector()
#fasterRcnn.saveConv()
#
#
## in test mode
#imgs = []
#fasterRcnn = FasterRcnn(configObject, train=False)
#res, preview = fasterRcnn.predict(imgs, preview=True)
