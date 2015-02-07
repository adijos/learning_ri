# random_words.py
# creates random index vectors for words in text

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

alphabet = string.lowercase

class RIGrammar:

    def __init__(self, N, k, b):
        self.N = N # N-dimensional space
        self.k = k # 2k sparse vector
        self.wordz = {} # dictionary to store words to random index
        self.b = b # number of basis elements to learn
        self.basis = np.random.rand(N,b)
        self.RI_letters = ridx.generate_letter_id_vectors(self.N,self.k, alph=alphabet)

    def add_words(self, text_path=None, byletter=0):
        if not text_path: print "no text file to load"; return;

        words = utils.extract_words(text_path, length=8)
        for word in words:
            if word in self.wordz: continue

            # generate ri vector 
            if byletter:
                rand_vector = np.zeros((1,self.N))
                rand_idx = np.random.permutation(self.N)
                rand_vector[0,rand_idx[0:self.k]] = 1
                rand_vector[0,rand_idx[self.k:2*self.k]] = -1
            else:
                rand_vector = np.zeros((1,self.N))
                for letter in list(word):
                    letter_idx = alphabet.index(letter)
                    letter_vec = self.RI_letters[:,letter_idx]
                    rand_vector = np.multiply(rand_vector, letter_vec[:,np.newaxis])

            # add to wordz
            self.wordz[word] = rand_vector
        return

    def learn_basis_over(self, eps=10e-8, text_path=None, block_sz=0):
        if not text_path: print "no text file off which to learn"; return;

        text_words = utils.extract_words(text_path, length=8)
        print 'len textwords', len(text_words)

        for k in trange(len(text_words)):
            if k < block_sz or k+ block_sz >= len(text_words): continue;

            word_vec = np.ones((1,self.N))
            for bidx in xrange(k-block_sz,k+block_sz+1):
                word_vec = np.multiply(word_vec,self.wordz[text_words[bidx]])

            word_vec = word_vec/np.linalg.norm(word_vec)
            weights = word_vec.dot(self.basis)

            for i in xrange(self.b):
                self.basis[:,i] = np.reshape(self.basis[:,i],(1,self.N)) +  weights[0][i]*word_vec

            # normalize basis
            for i in xrange(self.b):
                self.basis[:,i] = np.reshape(self.basis[:,i],(1,self.N))/np.linalg.norm(self.basis[:,i])

            self.basis[self.basis < eps] = 0

        #print self.basis[:,1]
        return

    def find_reps(self,lim=0):
        representations = {}
        for i in xrange(self.b):
            representations[alphabet[i]] = []

        count = 0
        for word in tqdm(self.wordz):
            word_vec = self.wordz[word]
            word_vec = word_vec/np.linalg.norm(word_vec)
            cosangles = word_vec.dot(self.basis)
            #print 'cosine angles', cosangles
            rep_basis = np.argmax(cosangles)
            representations[alphabet[rep_basis]].append(word)
            if lim > 0 and count > lim:
                break
            count += 1
        return representations
