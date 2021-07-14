from tensorflow.keras.layers import Conv2D, Dropout, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Sequential

import TF2_Keras_Template as template

class NeuralNetwork(template.nnBase.NNBase):
    
    def __init__(self):
        #Only sets the name of this class
        self.networkName = "DeepResidial"
            
    def makeModel(self,inputShape,outputShape):
        """
            overrides base function
            Create and return a Keras Model
        """
        
        input_img = Input(shape=inputShape)
        block1 = self.resBlock(input_img,width,height)
        block2 = self.resBlock(block1,width,height)
        block3 = self.resBlock(block2,width,height)
        finalConv = Conv2D(1, (3, 3), activation='relu',padding='same')(block3)
        out = Reshape((height,width,))(finalConv)
        
        model = Model(input_img,out)
        model.compile(optimizer='adam', loss='mean_squared_error',metrics=["mae"])
        return model
    
    def resBlock(self,input_,width,height):
        conv1 = Conv2D(32, (3, 3), activation='relu',padding='same')(input_)
        pool1 = MaxPooling2D((2, 2))(conv1)
        conv2 = Conv2D(64, (3, 3), activation='relu',padding='same')(pool1)
        pool2 = MaxPooling2D((2, 2))(conv2)
        conv3 = Conv2D(128, (3, 3), activation='relu',padding='same')(pool2)
        pool3 = MaxPooling2D((2, 2))(conv3)
        conv4 = Conv2D(256, (3, 3), activation='relu',padding='same')(pool3)
        
        dense1 = Dense(500)(conv4)

        conv5 = Conv2D(256, (3, 3), activation='relu',padding='same')(dense1)
        add1 = Concatenate()([conv5,conv4])
        conv6 = Conv2D(128, (3, 3), activation='relu',padding='same')(add1)
        up2 = UpSampling2D((2,2))(conv6)
        add2 = Concatenate()([up2,conv3])
        conv7 = Conv2D(64, (3, 3), activation='relu',padding='same')(add2)
        up3 = UpSampling2D((2,2))(conv7)
        add3 = Concatenate()([up3,conv2])
        conv8 = Conv2D(32, (3, 3), activation='relu',padding='same')(add3)
        up4 = UpSampling2D((2,2))(conv8)
        add4 = Concatenate()([up4,conv1])
        return add4
                


