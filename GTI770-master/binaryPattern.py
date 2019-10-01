#Refference inspirer de www.pyimagesearch.com/2015/12/07/local-binary-paterns-with-python-opencv/
from skimage import feature
from scipy.stats import entropy as scipy_entropy 
import numpy as np
from numpy import unique
from color import crop_center
import cv2
 """
   Class binary Paterne. En premier lieu, elle permet de caculer les forme de la surface d'une image, 
   en second lieu elle calcule l'entropy de celle ci 
    
    input :
        numPoint = integer, fournie lkes nombre de point d'interet du relief de l'image
        raduis = rayon en bit de l'image, represente la zone dans laquel on cehrcher les point
        img = image a identifier
        
    output : 
        entropy de chaque image 
    
    """
class GalaxyBinaryPatterns:
    def __init__ (self,numPoints, radius):
       #Enregistre les point et radius 
        # va permettre de construire un histograme
        self.numPoints = numPoints
        self.radius = radius
        
   # MX=cropImage(240,240,X_train)    
    def Galaxy_description(self,image , eps=1e-7): 
        lbp = feature.local_binary_pattern(image , self.numPoints , self.radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(),bins = np.arange(0 , self.numPoints +3),
                                range=(0,self.numPoints +2))
        hist = hist.astype("float")
        hist/= (hist.sum()+eps)
        
        return hist

def binaryPatterns(img,numPoints,radius):
    Patern = GalaxyBinaryPatterns(numPoints,radius)
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    Hist = Patern.Galaxy_description(gris)

    _,counts = unique(Hist,return_counts=True)

    return int(100 * scipy_entropy(counts,base=2))
        
    
# Histograme = []
# Donne = []
# #Patern = GalaxyBinaryPatterns(24,8)
# #Patern = GalaxyBinaryPatterns(60,60)
# # MX=crop_center(X_train,125,125) 
# # for image in MX:
# #     gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# #     Hist = Patern.Galaxy_description(gris)
# #     Histograme.append(Hist)
    
    
# print(Histograme)  

# MEntropy = []
# def entropy(matrix , base=2):
#     for i in matrix:
#         _,counts = unique(i,return_counts=True)
#         MEntropy.append(scipy_entropy(counts,base=base))
#     return MEntropy    

# entropy(Histograme,2)