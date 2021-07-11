import TF2_Keras_Template as template
import CNN

batchsize = 16


#Get data generator
ds = template.ImageDataset(batchsize)
ds.addDataFromTXT("data/train_images.txt","data/train_fixations.txt","data/val_images.txt","data/val_fixations.txt")
ds.addDataFromTXT("data/CAT2000/images.txt","data/CAT2000/fixations.txt",splitTrain=True)
ds.addDataFromTXT("data/MIT/imgs.txt","data/MIT/fixs.txt",splitTrain=True)
trainGenerator = ds.getGenerator(outputsize=(192,192))
valGenerator = ds.getGenerator(outputsize=(192,192),isTrain=False)


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
