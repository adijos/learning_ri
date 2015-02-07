import os
import Image
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

home_dir = os.getcwd()
lena = home_dir + '/lena.png'

img = Image.open(lena)
img_grey = img.convert('L')
img_arr = np.array(img_grey)
plt.imshow(img_grey, cmap=cm.Greys_r)
#plt.show()

# decide patch size parameters
# extract patches
# random index randomly chosen number of patches
# visualize learnt "local" features
