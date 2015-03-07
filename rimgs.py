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
        self.basis = 0.01*np.random.rand(N,b)
        self.alphabet = gen_pixelv()
        self.RI_letters = self.generate_letter_id_vectors(self.N,self.k, self.alphabet)

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

    def image_vectorize(self, image, window=0):

        m,n = image.shape
        # total
        image_vec = np.zeros((1, self.N))
        # pad image
        padded_img = np.lib.pad(image,window,mode='constant')
        # iterate over pixels
        for i in xrange(window,window+m):
            if i != 14: continue
            for j in xrange(window,window+n):
                if (j != 13): continue
                # pixel is (i,j)
                #print (i,j)
                if window == 0:
                    image_vec += self.RI_letters[padded_img[i,j]]
                pixelw = padded_img[i-window:i+window + 1, j-window:j+window+1]
                #print pixelw
                x_sz,y_sz =  pixelw.shape
                fl_x = int(np.floor(x_sz/2))
                fl_y = int(np.floor(y_sz/2))
                flat_pixelw = pixelw.flatten()
                #print flat_pixelw
                pixnum = len(flat_pixelw)
                pix_vec = np.ones((1,self.N))
                orig = np.array([0,1])
                orig_z = np.array([0,1,0])

                # complicated angle encoding on complex plane of pixel relations
                for x in xrange(-fl_x, x_sz - fl_x):
                    for y in xrange(-fl_y, y_sz - fl_y):

                        if x == 0 and y == 0: continue

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

                '''
                # just sequentially shifting, no angle encoding
                for shift in xrange(-pixnum/2+1, pixnum/2):
                    #print flat_pixelw[pix_idx]
                    pixy = self.RI_letters[flat_pixelw[pix_idx], :]
                    #print 'pixy', shift
                    #print pixy
                    pix_vec *= np.roll(pixy,shift)
                    pix_idx += 1
                '''
                #print pix_vec
                #print pix_vec.shape
                #print "~~~~~~~~~~~~~~~~"

                image_vec += pix_vec
        return image_vec

    def learn_basis(self, image_set, eps=10e-8, window=None):
        if not window: print "no window size specified"; return;

        num_imgs, m, n = image_set.shape

        for k in trange(num_imgs):

            image_vec = self.image_vectorize(image_set[k,:,:], window=window)
            image_vec = image_vec/np.linalg.norm(image_vec)
            weights = image_vec.dot(self.basis)

            for i in xrange(self.b):
                self.basis[:,i] = np.reshape(self.basis[:,i],(1,self.N)) +  weights[0][i]*image_vec

            # normalize basis
            for i in xrange(self.b):
                self.basis[:,i] = np.reshape(self.basis[:,i],(1,self.N))/np.linalg.norm(self.basis[:,i])

            #self.basis[self.basis < eps] = 0

        return

    def find_reps(self, image_set=None, lim=0, max_iter=100,pixel_m=28, pixel_n=28, window=None):
        """ find_reps means finds pixel representations of high dimensional basis vectors!"""

        if not window: print "no window size specified"; return;
        representations = np.zeros((self.b,pixel_m, pixel_n))

        # normalize basis in case
        for i in xrange(self.b):
            self.basis[:,i] = np.reshape(self.basis[:,i],(1,self.N))/np.linalg.norm(self.basis[:,i])

        if image_set == None:
            random = 1
        else:
            random = 0
            num_imgs, m, n = image_set.shape

            # limits iterations by number of images if image_set is defined
            if num_imgs < max_iter:
                max_iter = num_imgs

        for i in trange(max_iter):

            if random:
                img = np.random.rand(m,n)
            else:
                img = image_set[i,:,:]

            image_vec = self.image_vectorize(img, window=window)
            cosangles = image_vec.dot(self.basis)
            rep_basis = np.argmax(cosangles)
            coslist = cosangles.tolist()[0]
            representations[rep_basis,:,:] = np.reshape(representations[rep_basis,:],(pixel_m, pixel_n)) + coslist[rep_basis]*img

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
