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
        self.wordz = {} # dictionary to store words to random index
        self.b = b # number of basis elements to learn
        self.basis = np.random.rand(N,b)
        self.alphabet = gen_pixelv()

        self.RI_letters = self.generate_letter_id_vectors(self.N,self.k, self.alphabet)

    def image_vectorize(self, image, window=0):

        m,n = image.shape
        # total
        image_vec = np.zeros((1, self.N))
        # pad image
        padded_img = np.lib.pad(image,window,mode='constant')
        # iterate over pixels
        for i in xrange(window,window+m):
            for j in xrange(window,window+n):
                # pixel is (i,j)
                #print (i,j)
                if window == 0:
                    image_vec += self.RI_letters[padded_img[i,j]]
                pixelw = padded_img[i-window:i+window + 1, j-window:j+window+1]
                flat_pixelw = pixelw.flatten()
                #print flat_pixelw
                pixnum = len(flat_pixelw)
                pix_idx = 0
                pix_vec = np.ones((1,self.N))
                for shift in xrange(-pixnum/2+1, pixnum/2):
                    #print flat_pixelw[pix_idx]
                    pixy = self.RI_letters[flat_pixelw[pix_idx], :]
                    #print 'pixy', shift
                    #print pixy
                    pix_vec *= np.roll(pixy,shift)
                    pix_idx += 1
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

            self.basis[self.basis < eps] = 0

        return 

    def find_reps(self,lim=0, max_iter=100,pixel_m=28, pixel_n=28, window=None):
        if not window: print "no window size specified"; return;
        representations = {}
        for i in xrange(self.b):
            representations[alphabet[i]] = [np.random.rand(pixel_m,pixel_n)]

        random_matrix = np.random.rand(pixel_m,pixel_n)
        image_vec = self.image_vectorize(random_matrix, window=window)
        cosangles = image_vec.dot(self.basis)

        rep_basis = np.argmax(cosangles)
        representations[str(rep_basis)] += cosangles[rep_basis]*random_matrix
        return representations

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
