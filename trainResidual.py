import TF2_Keras_Template as template
import CNN
from train import CustomDataset

def makeTrainable(blocks):
    global model
    for layer in model.layers[1:20]:
        layer.trainable = blocks[0]
    for layer in model.layers[20:39]:
        layer.trainable = blocks[1]
    for layer in model.layers[39:58]:
        layer.trainable = blocks[2]
    model.compile(optimizer='adam', loss='mean_squared_error')


def train(currentEpoch=0,epochLength=500):
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(trainGenerator,
                steps_per_epoch=trainSteps,
                epochs=currentEpoch+epochLength,
                shuffle=True,
                initial_epoch=currentEpoch,
                validation_steps=validSteps,
                validation_data=valGenerator,
                callbacks=callbacks)    
    

batchsize = 64


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
callbacks = logger.getCallbacks(period=10)


#Train
trainSteps = int(len(ds.trainData)/batchsize)
validSteps = int(len(ds.valData)/batchsize)
firstRun = True
    
print("Starting Phase 1")
makeTrainable((True,False,False))
train(50)

print("Starting Phase 2")
makeTrainable((False,True,False))
train(50,50)

print("Starting Phase 3")
makeTrainable((False,False,True))
train(100,50)

print("Starting End Phase")
makeTrainable((True,True,True))
train(150,250)
