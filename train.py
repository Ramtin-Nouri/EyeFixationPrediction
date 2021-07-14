import TF2_Keras_Template as template
import CNN,cv2,numpy as np

batchsize = 16

class CustomDataset(template.ImageDataset):
    def augmentate(self, batchIn, batchOut, isTrain):
        if isTrain:
            return self.normCropReshape(batchIn,batchOut,(192,192))
        else:
            ins = []
            outs = []
            for in_,out_ in zip(batchIn,batchOut):
                ins.append(cv2.resize(in_/255,(224,224)))
                outs.append(cv2.resize(out_/255,(224,224)))
            return (np.array(ins),np.array(outs))

#Get data generator
ds = CustomDataset(batchsize)
ds.addDataFromTXT("data/train_images.txt","data/train_fixations.txt","data/val_images.txt","data/val_fixations.txt")
ds.addDataFromTXT("data/CAT2000/images.txt","data/CAT2000/fixations.txt",splitTrain=True)
ds.addDataFromTXT("data/MIT/imgs.txt","data/MIT/fixs.txt",splitTrain=True)
trainGenerator = ds.getGenerator(isTrain=True)
valGenerator = ds.getGenerator(isTrain=False)


#Get Model
net = CNN.NeuralNetwork()
model,epoch = net.getModel((None,None,3),(None,None,1)) #(None,None) basically means we don't care. Because it is a CNN the output shape will be determined by the architecture


#Get Loggers
logger = template.Logger("savedata/",model)
logger.setTestImages("data/images/test")
callbacks = logger.getCallbacks(period=20)


#Train
trainSteps = int(len(ds.trainData)/batchsize)
validSteps = int(len(ds.valData)/batchsize)
model.fit(trainGenerator,
                steps_per_epoch=trainSteps,
                epochs=1000,
                shuffle=True,
                initial_epoch=epoch,
                validation_steps=validSteps,
                validation_data=valGenerator,
                callbacks=callbacks)
