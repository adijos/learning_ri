# random_img.py
# process images by random indexing

# libraries
import numpy as np
from mnist_loader import load_mnist
import rimgs
import matplotlib.pyplot as plt
import utils

limit_images = 60000
# load images
images, labels = load_mnist('training')
print images.shape
random_shuffle = np.random.permutation(np.r_[0:images.shape[0]])
images = images[random_shuffle[0:limit_images],:,:]
print images.shape
digits = []
for i in xrange(10):
    digits.append(str(i))

# initialize random indexing object
s,t = 2,5 # factored out number of bases for plotting purposes
N,k,b = (256*4,128*4,s*t)
window = 1
RIM = rimgs.RIImages(N,k,b)
#print RIM.RI_letters.shape
#print RIM.RI_letters

# learn bases over image set
RIM.learn_basis(images,window=window)
print "successfully learnt basis"

reps = RIM.find_reps(image_set=images,window=window)
print "learnt representations"
flattened_reps = RIM.flatten_reps()
print flattened_reps[0,:]
# clustering plots
#cosangles = utils.cosangles(RIM.basis)
#print cosangles
#utils.plot_clusters(cosangles, digits)

# plot reps
f, axarr = plt.subplots(s, t)
for i in xrange(s):
    for j in xrange(t):
        print i,j, (j)+(t)*i
        axarr[i,j].imshow(reps[j+(t)*i], cmap='gray')

print "~~~~"
#axarr[0, 0].plot(x, y)
#axarr[0, 0].set_title('Axis [0,0]')
#axarr[0, 1].scatter(x, y)
#axarr[0, 1].set_title('Axis [0,1]')
#axarr[1, 0].plot(x, y ** 2)
#axarr[1, 0].set_title('Axis [1,0]')
#axarr[1, 1].scatter(x, y ** 2)
#axarr[1, 1].set_title('Axis [1,1]')
## Fine-tune figure; hide x ticks for top plots and y ticks for right plots
for i in xrange(s):
    plt.setp([a.get_xticklabels() for a in axarr[i, :]], visible=False)
for i in xrange(t):
    plt.setp([a.get_yticklabels() for a in axarr[:, i]], visible=False)

g, axarr2 = plt.subplots(10,10)
for i in xrange(10):
    for j in xrange(10):
        print (j)+(10)*i
        axarr2[i,j].imshow(images[j+(10)*i,:,:], cmap='gray')
for i in xrange(s):
    plt.setp([a.get_xticklabels() for a in axarr2[i, :]], visible=False)
for i in xrange(t):
    plt.setp([a.get_yticklabels() for a in axarr2[:, i]], visible=False)

plt.show()
