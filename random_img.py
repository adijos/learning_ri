# random_img.py
# process images by random indexing

# libraries
from pylab import *
import numpy as np
from mnist_loader import load_mnist
import rimgs
import matplotlib.pyplot as plt

# load images
images, labels = load_mnist('training')
print images.shape
images = images[0:100,:,:]

# initialize random indexing object
N,k,b = (1000,100,10)
window = 1
RIM = rimgs.RIImages(N,k,b)
#print RIM.RI_letters.shape
#print RIM.RI_letters

# learn bases over image set
RIM.learn_basis(images,window=window)
print RIM.basis


