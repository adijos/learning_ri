# rimgs.py
# creates random index vectors for images

# libraries
import sys
import numpy as np
import string
import utils
import pandas as pd
import os
import string
from tqdm import trange
from tqdm import tqdm
import random_idx as ridx


def gen_pixelv(view=0):
    alphabet = []
    for i in xrange(0,256):
        alphabet.append(str(i))
    if view: print alphabet
    return alphabet

class RIImages:

    def __init__(self, N, k, b):
        self.N = N # N-dimensional space
        self.k = k # 2k sparse vector
        self.b = b # number of basis elements to learn
        self.basis = np.random.rand(N,b)
        self.alphabet = gen_pixelv()
        self.RI_letters = self.generate_letter_id_vectors(self.N,self.k, self.alphabet)
        self.sampled_pixelw = []

        self.representations = None

    def nearest_k(self, theta):
        diff = 1000
        best_k = -1
        for k in xrange(1,self.N):
            diff_k = abs((2*np.pi*k)/self.N - theta)
            if diff_k < diff:
                diff = diff_k
                best_k = k
        return best_k

    def extract_pixelw(self, image, window=0):
    # image-extract pixel window

        m,n = image.shape
        # total
        image_vec = np.zeros((1, self.N))
        # pad image
        padded_img = np.lib.pad(image,window,mode='constant')

        # randomly choose pixel window
        i = np.random.randint(window, high=window+m)
        j = np.random.randint(window, high=window+n)
        pixelw = padded_img[i-window:i+window + 1, j-window:j+window+1]

        self.sampled_pixelw.append(pixelw)

        return pixelw


    def pixelw_vectorize(self, pixelw, window=0):
    # pixel_window vectorize

        if window == 0:
            # TEST
            self.RI_letters[pixelw]

        #print pixelw
        # useful size constants
        x_sz,y_sz =  pixelw.shape
        fl_x = int(np.floor(x_sz/2))
        fl_y = int(np.floor(y_sz/2))

        # vector initializations
        pixelw_vec = np.zeros((1, self.N))
        pix_vec = np.ones((1,self.N))
        orig = np.array([0,1])
        orig_z = np.array([0,1,0])

        # complicated angle encoding on complex plane of pixel relations
        for x in xrange(-fl_x, x_sz - fl_x):
            for y in xrange(-fl_y, y_sz - fl_y):

                if x == 0 and y == 0:
                    pixelw_vec += self.RI_letters[pixelw[x,y],:]

                # calculate vector angles by dot product
                pixel_vec = np.array([x,y])
                #print orig, pixel_vec
                pixel_vec_z = np.array([x,y,0])
                pixel_vec_normed = pixel_vec/np.linalg.norm(pixel_vec)

                # use cross product to orient arccos angle correctly [0,2pi]
                cross = np.cross(orig_z, pixel_vec_z)
                cosangle = pixel_vec_normed.dot(orig)
                angle = np.arccos(cosangle)
                if cross[2] < 0:
                    angle = -angle
                if angle < 0:
                    angle = angle + 2*np.pi

                #print 'cosangle', cosangle,'angle', angle,'cross', cross

                # find nearest angle
                shift = self.nearest_k(angle)
                #print shift

                pixy = self.RI_letters[pixelw[x,y],:]
                pix_vec = np.roll(pixy,shift)

                pixelw_vec += pix_vec

        return pixelw_vec


    def learn_basis(self, image_set, eps=10e-8, window=None, sample_num=1000):
        if not window: print "no window size specified"; return;

        num_imgs, m, n = image_set.shape
        for k in trange(sample_num):

            #image_vec = self.image_vectorize(image_set[k,:,:], window=window)
            #image_vec = image_vec/np.linalg.norm(image_vec)
            # select random image number
            img_num = np.random.randint(0,high=10)
            #print "sample: ", k, "; image: ", img_num

            # random index random pixel window in image
            pixelw = self.extract_pixelw(image_set[img_num,:,:], window=window)
            pixelw_vec = self.pixelw_vectorize(pixelw, window=window)

            #find weights with relation to pixelw vector
            weights = pixelw_vec.dot(self.basis)

            # make 1 where max and 0 otherwise
            maxy = (weights == np.max(weights))
            maxed_weights = maxy*weights
            maxed_weights[maxed_weights != 0] = 1

            for i in xrange(self.b):
                self.basis[:,i] = np.reshape(self.basis[:,i],(1,self.N)) +  maxed_weights[0][i]*pixelw_vec

            # normalize basis
            for i in xrange(self.b):
                self.basis[:,i] = np.reshape(self.basis[:,i],(1,self.N))/np.linalg.norm(self.basis[:,i])

            # hard threshold basis values
            #self.basis[self.basis < eps] = 0

        return

    def find_reps(self, image_set=None, lim=0, max_iter=100, window=None):
        """ find_reps means finds pixel representations of high dimensional basis vectors!"""

        if not window: print "no window size specified"; return;

        m = 2*window+1
        n = 2*window+1
        representations = 0.01*np.random.rand(self.b, m, n)

        # normalize basis in case
        for i in xrange(self.b):
            self.basis[:,i] = np.reshape(self.basis[:,i],(1,self.N))/np.linalg.norm(self.basis[:,i])

        if image_set == None:
            random = 1
        else:
            random = 0

            # limits iterations by number of images if image_set is defined

        for i in trange(max_iter):

            if random:
                pixelw = np.random.rand(m,n)
            else:
                img_num = np.random.randint(0,high=10)
                img = image_set[img_num,:,:]
                pixelw = self.extract_pixelw(img, window=window)

            pixelw_vec = self.pixelw_vectorize(pixelw, window=window)
            cosangles = pixelw_vec.dot(self.basis)
            rep_basis = np.argmax(cosangles)
            coslist = cosangles.tolist()[0]
            #print coslist
            representations[rep_basis,:,:] = np.reshape(representations[rep_basis,:],(m, n)) + coslist[rep_basis]*pixelw

        self.representations = representations
        return representations

    def flatten_reps(self):
        if self.representations==None: return;
        b,m,n = self.representations.shape
        flattened_reps = np.zeros((self.b,m*n))
        for i in xrange(self.b):
            flattened_reps[i,:] = self.representations[i,:,:].flatten()
        return flattened_reps

    def generate_letter_id_vectors(self, N, k, alph):
        # build row-wise k-sparse random index matrix
        # each row is random index vector for letter
        num_letters = len(alph)
        #print num_letters
        RI_letters = np.zeros((num_letters,N))
        for i in xrange(num_letters):
            rand_idx = np.random.permutation(N)
            RI_letters[i,rand_idx[0:k]] = 1
            RI_letters[i,rand_idx[k:2*k]] = -1
        return RI_letters
