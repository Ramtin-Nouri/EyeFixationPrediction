import TF2_Keras_Template as  template
import cv2,os,numpy as np,math

class CustomLogger(template.Logger):

    def setTestImages(self,testImageFolder):
        imgpaths = os.listdir(testImageFolder)[:8]
        #Pray they are actually images
        for img in imgpaths:
            self.testImages.append(cv2.imread(F"{testImageFolder}/{img}")/255)

    def stack(self,imgs):
        sqrt = int(math.sqrt(len(imgs)))
        rowLength = math.ceil(len(imgs)/sqrt)
        rows=[]
        shape = imgs[0].shape
        for row in range(sqrt):
                thisRow=[]
                for col in range(rowLength):
                        try:
                            img = imgs[row*rowLength+col]
                            if img.shape[2] < 3:
                                img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
                            thisRow.append(img)
                        except:
                                thisRow.append( np.zeros((shape[0],shape[1],3)) )
                rows.append(np.hstack(thisRow))
        return np.vstack(rows)*255