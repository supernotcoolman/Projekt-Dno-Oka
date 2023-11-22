import cv2
import skimage as sk
import numpy as np
import matplotlib.pyplot as plt

################### FUNKCJE

def toSquare(image):
    
    h = image.shape[0]
    w = image.shape[1]
    
    a = max(h,w)
    
    delta_h = (a - h)//2
    delta_w = (a - w)//2
    
    new = np.zeros((a,a))
    new[delta_h:delta_h+h, delta_w:delta_w+w] = image
    new = np.asarray(new, dtype = np.uint8)
    #new = cv2.resize(new, (800,800))
    
    return new

def toGray(images):
    for i in range(len(images)):
        
        tmp = np.asarray(images[i], dtype = np.uint8)
        images[i] = np.asarray(tmp)
        #images[i] = np.asarray(images[i], dtype=np.uint8)
        images[i][:,:,0] = 0
        images[i][:,:,2] = 0
        images[i] = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)
        images[i] = toSquare(images[i])
        
    return images

def imageProcessing(images):
    for i in range(len(images)):
        images[i] = cv2.medianBlur(images[i], 3)
        
        # images[i] = cv2.Laplacian(images[i], cv2.CV_8U, ksize=3) # wyostrzenie obrazu, które zniekształca i psuje obraz >:(
        # images[i] = cv2.Laplacian(images[i], cv2.CV_64F)
        # images[i] = cv2.convertScaleAbs(images[i])
        
        images[i] = cv2.equalizeHist(images[i])
        
        #_,images[i] = cv2.threshold(images[i], 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        images[i] = cv2.adaptiveThreshold(images[i], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,121,18)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        images[i] = cv2.morphologyEx(images[i], cv2.MORPH_OPEN, kernel)
        
        # images[i] = sk.filters.frangi(images[i])
        
        # images[i] = cv2.normalize(images[i], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    return images


def getResults(images, hands, fovmask):
    
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    
    for i in range(fovmask.shape[0]):
        for j in range(fovmask.shape[1]):
            if fovmask[j][i] == 255:
                if images[j][i] == hands[j][i] and hands[j][i] == 255:
                    TP += 1
                if images[j][i] == hands[j][i] and hands[j][i] == 0:
                    TN += 1
                if images[j][i] != hands[j][i] and hands[j][i] == 255:
                    FN += 1 
                if images[j][i] != hands[j][i] and hands[j][i] == 0:
                    FP += 1 
                    
                       
    accuracy = ((TP+TN) / (TP+FP+FN+TN)) * 100 
    sensitivity = (TP / (TP+FN)) * 100
    specificity = (TN / (FP+TN)) * 100  
    
      
    return accuracy, sensitivity, specificity
   
   
def finalsResults(images, hands, fovmask, inputs):
    
    for i in range(len(images)):
        acc, sens, spec = getResults(images[i], hands[i], fovmask)
        
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))

        axes[0, 0].imshow(inputs[i], cmap='gray')
        axes[0, 0].set_title('Obraz wejściowy')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(images[i], cmap='gray')
        axes[0, 1].set_title('Przetworzony obraz')
        axes[0, 1].axis('off')

        axes[1, 0].imshow(hands[i], cmap='gray')
        axes[1, 0].set_title('Porównanie z ręcznie oznaczonymi naczyniami')
        axes[1, 0].axis('off')

        axes[1, 1].axis('off')
        axes[1, 1].text(0.1, 0.4, f'Accuracy: {acc}\nSensitivity: {sens}\nSpecificity: {spec}',fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
        
        plt.tight_layout()
        plt.show()
    


################### MAIN

image1 = cv2.imread('healthy/01_h.jpg', cv2.IMREAD_UNCHANGED)
image2 = cv2.imread('healthy/02_h.jpg', cv2.IMREAD_UNCHANGED)
image3 = cv2.imread('healthy/03_h.jpg', cv2.IMREAD_UNCHANGED)
image4 = cv2.imread('healthy/04_h.jpg', cv2.IMREAD_UNCHANGED)
image5 = cv2.imread('healthy/05_h.jpg', cv2.IMREAD_UNCHANGED)

inputs = [image1,image2,image3,image4,image5]
images = [image1,image2,image3,image4,image5]
#images = [image1]

hand1 = cv2.imread('healthy_manual/01_h.tif', cv2.IMREAD_UNCHANGED)
hand2 = cv2.imread('healthy_manual/02_h.tif', cv2.IMREAD_UNCHANGED)
hand3 = cv2.imread('healthy_manual/03_h.tif', cv2.IMREAD_UNCHANGED)
hand4 = cv2.imread('healthy_manual/04_h.tif', cv2.IMREAD_UNCHANGED)
hand5 = cv2.imread('healthy_manual/05_h.tif', cv2.IMREAD_UNCHANGED)

hands = [hand1,hand2,hand3,hand4,hand5]
#hands = [hand1]

for i in range(len(hands)):
    hands[i] = toSquare(hands[i])
    

fovmask = cv2.imread('healthy_fovmask/01_h_mask.tif', cv2.IMREAD_UNCHANGED)
fovmask = cv2.cvtColor(fovmask, cv2.COLOR_BGR2GRAY)
fovmask = toSquare(fovmask)


toGray(images)
imageProcessing(images)

     
finalsResults(images, hands, fovmask, inputs)



#cv2.imshow('image',images[0])
#cv2.waitKey(0)
