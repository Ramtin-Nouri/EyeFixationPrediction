import TF2_Keras_Template as template
from nets import ResFreeze
from dataManager import CustomDataset

def makeTrainable(i):
    global model
    preBlock=True
    postBlock = False
    for layer in model.layers:
        if preBlock:
            #Everything until the start layer will be not trainable
            if layer.name == F"BlockStart_{i}":
                layer.trainable=True
                preBlock=False
            else:
                layer.trainable=False
        elif postBlock:
            #Everthing after the end layer will be not trainable
            layer.trainable=False
        else:
            #If were not before not after the block we will set layer trainable and look for the end
            layer.trainable=True
            if layer.name == F"BlockEnd_{i}":
                postBlock = True

    model.compile(optimizer='adam', loss='mean_squared_error',metrics=["mae"])


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
    

batchsize = 16


#Get data generator
ds = CustomDataset(batchsize)
ds.addDataFromTXT("data/train_images.txt","data/train_fixations.txt","data/val_images.txt","data/val_fixations.txt")
ds.addDataFromTXT("data/CAT2000/images.txt","data/CAT2000/fixations.txt",splitTrain=True)
ds.addDataFromTXT("data/MIT/imgs.txt","data/MIT/fixs.txt",splitTrain=True)
trainGenerator = ds.getGenerator(isTrain=True)
valGenerator = ds.getGenerator(isTrain=False)


#Get Model
net = ResFreeze.NeuralNetwork()
model,epoch = net.getModel((None,None,3),(None,None,1)) #(None,None) basically means we don't care. Because it is a CNN the output shape will be determined by the architecture


#Get Loggers
logger = template.Logger("savedata/",model)
logger.setTestImages("data/images/test")
callbacks = logger.getCallbacks(period=10)


#Train
trainSteps = int(len(ds.trainData)/batchsize)
validSteps = int(len(ds.valData)/batchsize)
firstRun = True
    
print("Starting Phase 0")
makeTrainable(0)
train(0,50)

print("Starting Phase 1")
makeTrainable(1)
train(50,50)

print("Starting Phase 2")
makeTrainable(2)
train(100,50)

print("Starting End Phase")
for layer in model.layers: layer.trainable=True
model.compile(optimizer='adam', loss='mean_squared_error',metrics=["mae"])
train(150,250)