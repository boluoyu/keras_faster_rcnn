# Faster RCNN keras implementation 
The project aim to implement faster rcnn in keras with tf. And it can create a high level wrapping to make faster rcnn simple to use.

## Task list
- [X] Design the flow 
- [ ] Data generator from kiti
- [ ] Conv tensor
- [ ] Conv layer (init, Load and save)
- [ ] RPN tensor
- [ ] RPN layer (init, sliding windows, Load and save)
- [ ] RPN muti-scale Anchors 
- [ ] RPN Loss 
- [ ] RPN training
- [ ] Detection model 
- [ ] Detection(Faster-RCNN) (init, Load and save)
- [ ] Detection Loss 
- [ ] Detection training
- [ ] Detection preview 
- [ ] Cal mAP

-------
  
#### The planeed use case
```python 
dataGenerator = DataGenerator()
configObject = ConfigObject()
configObject.setModelPath(rpn="",conv="", detection="")


# create rastRcnnObject
fasterRcnn = FasterRcnn(configObject, train=True)

# step 1 train RPN
fasterRcnn.initConvLayer("vgg16Path")
fasterRcnn.initRandomRpn()
fasterRcnn.trainRpn(dataGenerator)
fasterRcnn.saveRpn()

# step 2 train detector network
fasterRcnn.initConvLayer("vgg16Path")
fasterRcnn.initRandomDetector()
fasterRcnn.trainDetector(dataGenerator)
fasterRcnn.saveDetector()
fasterRcnn.saveConv()

# step 3 fine tune rpn 
fasterRcnn.trainRpn(dataGenerator)
fasterRcnn.saveRpn()
fasterRcnn.saveConv()

# step 4 fine tune detector
fasterRcnn.trainDetector(dataGenerator)
fasterRcnn.saveDetector()
fasterRcnn.saveConv()


# in test mode 
imgs = []
fasterRcnn = FasterRcnn(configObject, train=False)
res, preview = fasterRcnn.predict(imgs, preview=True)
```

