import TF2_Keras_Template as template

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
