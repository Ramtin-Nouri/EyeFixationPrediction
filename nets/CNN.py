from tensorflow.keras.layers import Conv2D, Dropout, MaxPooling2D, UpSampling2D, Input,concatenate, BatchNormalization
from tensorflow.keras.models import Model

import TF2_Keras_Template as template

class NeuralNetwork(template.nnBase.NNBase):
    
    def __init__(self):
        #Only sets the name of this class
        self.networkName = "CNN"
            
    def makeModel(self,inputShape,outputShape):
        """
            overrides base function
            Create and return a Keras Model
        """
        dropoutStrength = 0.1

        _input = Input(shape=inputShape)

        #1
        x = Conv2D(128, (3, 3), activation='relu',padding='same', use_bias=False)(_input)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(dropoutStrength)(x)

        x = Conv2D(64, (3, 3), activation='relu',padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(dropoutStrength)(x)

        x = Conv2D(32, (3, 3), activation='relu',padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(dropoutStrength)(x)
                
        x = Conv2D(32, (3, 3),strides=(2,2), activation='relu', use_bias=False,padding='same')(x)
        x = BatchNormalization()(x)
        x1 = Dropout(dropoutStrength)(x)

        #2
        x = Conv2D(128, (5,5), activation='relu',padding='same', use_bias=False)(_input)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(dropoutStrength)(x)

        x = Conv2D(64, (5,5), activation='relu',padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(dropoutStrength)(x)

        x = Conv2D(32, (5,5), activation='relu',padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(dropoutStrength)(x)
                
        x = Conv2D(32, (5,5),strides=(2,2), activation='relu',padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x2 = Dropout(dropoutStrength)(x)

        #3
        x = Conv2D(128, (8,8), activation='relu',padding='same', use_bias=False)(_input)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(dropoutStrength)(x)

        x = Conv2D(64, (8,8), activation='relu',padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(dropoutStrength)(x)

        x = Conv2D(32, (8,8), activation='relu',padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(dropoutStrength)(x)
                
        x = Conv2D(32, (8,8),strides=(2,2), activation='relu',padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x3 = Dropout(dropoutStrength)(x)

        #Merge
        concat = concatenate([x1,x2,x3])
        concat = Conv2D(128, (3, 3), activation='relu',padding='same', use_bias=False)(concat)
        concat = BatchNormalization()(concat)
        concat = Conv2D(64, (3, 3), activation='relu',padding='same', use_bias=False)(concat)
        concat = BatchNormalization()(concat)
        concat = Conv2D(32, (3, 3), activation='relu',padding='same', use_bias=False)(concat)
        concat = BatchNormalization()(concat)

        #1
        x = Conv2D(32, (3, 3), activation='relu',padding='same', use_bias=False)(concat)
        x = BatchNormalization()(x)
        x = UpSampling2D((2,2))(x)
        x = Dropout(dropoutStrength)(x)

        x = Conv2D(32, (3, 3), activation='relu',padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = UpSampling2D((2,2))(x)
        x = Dropout(dropoutStrength)(x)

        x = Conv2D(64, (3, 3), activation='relu',padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = UpSampling2D((2,2))(x)
        x = Dropout(dropoutStrength)(x)

        x = Conv2D(128, (3, 3), activation='relu',padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = UpSampling2D((2,2))(x)
        x1 = Dropout(dropoutStrength)(x)

        #2
        x = Conv2D(32, (5, 5), activation='relu',padding='same', use_bias=False)(concat)
        x = BatchNormalization()(x)
        x = UpSampling2D((2,2))(x)
        x = Dropout(dropoutStrength)(x)

        x = Conv2D(32, (5, 5), activation='relu',padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = UpSampling2D((2,2))(x)
        x = Dropout(dropoutStrength)(x)

        x = Conv2D(64, (5,5), activation='relu',padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = UpSampling2D((2,2))(x)
        x = Dropout(dropoutStrength)(x)

        x = Conv2D(128, (5,5), activation='relu',padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = UpSampling2D((2,2))(x)
        x2 = Dropout(dropoutStrength)(x)

        #3
        x = Conv2D(32, (8,8), activation='relu',padding='same', use_bias=False)(concat)
        x = BatchNormalization()(x)
        x = UpSampling2D((2,2))(x)
        x = Dropout(dropoutStrength)(x)

        x = Conv2D(32, (8,8), activation='relu',padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = UpSampling2D((2,2))(x)
        x = Dropout(dropoutStrength)(x)

        x = Conv2D(64, (8,8), activation='relu',padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = UpSampling2D((2,2))(x)
        x = Dropout(dropoutStrength)(x)

        x = Conv2D(128, (8,8), activation='relu',padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = UpSampling2D((2,2))(x)
        x3 = Dropout(dropoutStrength)(x)

        #Merge
        concat = concatenate([x1,x2,x3])
        
        x = Conv2D(128, (3, 3), activation='relu',padding='same', use_bias=False)(concat)
        x = BatchNormalization()(x)
        x = Conv2D(64, (3, 3), activation='relu',padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Conv2D(32, (3, 3), activation='relu',padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Conv2D(16, (3, 3), activation='relu',padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Conv2D(8, (3, 3), activation='relu',padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        out = Conv2D(1, (3, 3), activation='relu',padding='same')(x)

        model = Model(_input,out)        
        model.compile(optimizer='adam', loss='mse',metrics=["mae"])
        return model
