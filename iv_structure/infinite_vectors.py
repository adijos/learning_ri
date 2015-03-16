# infinite_vectors.py
# implements "infinite-dimensional" vectors as sets of tuples, with various algebraic operations
# intended to be used for a generalized form of random indexing

#libraries
import numpy as np

class Vector(object):
    """ pseudo infinite-dimensional vectors implemented in a tractable manner """

    def __init__(self, beta=None, beta2=None, N=None, empty=0):
        """ N - dimensionality of vector, usually N >> 1
            beta - rational or irrational iterator parameter for first phase
            beta2 - rational or irrational iterator parameter for second phase (optional)
            empty - (don't) generate entries upon initialization
        """

        if N == None:
            self.N = 100.
        else:
            self.N = N

        if beta == None:
            beta = 1/self.N
        else:
            self.beta = beta

        if beta2 == None:
            self.beta2 = None
        else:
            self.beta2 = beta2

        self.entries = {}
        if not empty:
            self.generate_entries()

    def __repr__(self):
        """ information about vector in string form""" 
        param_str = "N= " +  str(self.N) + ", beta= " + str(self.beta)
        if self.beta2:
            param_str += ", beta2 = " + str(self.beta2)
        info_str = ""
        for phase in self.entries.keys():
            info_str += "(" + str(phase) + ", " + str(self.entries[phase]) + "), "
        return info_str

    def generate_entries(self):
        """ generate new entries for vector based on integer multiples of beta """
        print "generating entry"
        for k1 in xrange(1,int(self.N+1)):
            #print k1

            phase1 = (k1*self.beta) % 1
            #print phase1
            if self.beta2:
                for k2 in xrange(1,int(self.N+1)):
                    phase2 = (k2*self.beta2) % 1
                    if (phase1,phase2) not in self.entries.keys():
                        print "creating in generation"
                        self.create_entry((phase1,phase2))

            else:
                self.create_entry(phase1)

    def clear_entries(self):
        """ clear all (phase, entry) tuples """
        self.entries.clear()

    def create_entry(self, phase_tuple):
        """ randomly creates entry coupled to phase as element from {-1, 1} with uniform distribution """
        #print "adding phase", phase
        exponent = np.random.randint(1, high=3) #randomly initialize exponent
        self.entries[phase_tuple] = (-1)**exponent


    def __add__(self, w):
        """ add another Vector element-wise, creates entries if phases don't exist"""

        new_vec = Vector(N=self.N, beta=self.beta, empty=1)
        for phase_tuple in w.entries.keys():

            if phase_tuple not in self.entries.keys():
                print "creating in __add__"
                self.create_entry(phase_tuple)

            new_vec.entries[phase_tuple] = w.entries[phase_tuple] + self.entries[phase_tuple]

        return new_vec

    def __radd__(self,w):
        """ for use with the built-in sum function """
        if w == 0:
            return self
        else:
            return self.__add__(w)

    def __mul__(self, c):
        """ multiplies by scalar pointwise, creates entries if phases don't exist"""

        new_vec = Vector(N=self.N, beta=self.beta, empty=1)
        for phase_tuple in self.entries.keys():

            print "multiplying by scalar", phase_tuple

            new_vec.entries[phase_tuple] = c*self.entries[phase_tuple]

        return new_vec

    def __rmul__(self,c):
        """ for use with built-in mul function """
        if c == 1:
            return self
        else:
            return self.__mul__(c)

    def mult(self, w):
        """ multiple with another Vector element-wise, creates entries if phases don't exist"""

        new_vec = Vector(N=self.N, beta=self.beta, empty=1)
        for phase_tuple in w.entries.keys():

            if phase_tuple not in self.entries.keys():
                print "creating in mult"
                self.create_entry(phase_tuple)

            new_vec.entries[phase_tuple] = w.entries[phase_tuple]*self.entries[phase_tuple]

        return new_vec

    def __div__(self,c):
        if c == 0:
            raise ZeroDivisionError
        else:
            return self.__mul__(1/float(c))

    def permute(self,theta, phase_axis=1):
        """ permute Vector by phase, creates entries if phases don't exist"""

        permuted_vec = Vector(N=self.N, beta=self.beta, beta2=self.beta2, empty=1)
        for phase_tuple in self.entries.keys():
            if not self.beta2:
                new_phase_tuple = (phase_tuple + theta) % 1
            else:
                permuted_angle = (phase_tuple[phase_axis-1] + theta) % 1
                if phase_axis==1:
                    new_phase_tuple = (permuted_angle, phase_tuple[1])
                elif phase_axis==2:
                    new_phase_tuple = (phase_tuple[0], permuted_angle)

            permuted_vec.entries[new_phase_tuple] = self.entries[phase_tuple]

        return permuted_vec

    def roll(self, roll_int, phase_axis=1):
        """ multiplies Vector phases by roll_int, creatse entries if phases don't exist """
        permuted_vec = Vector(N=self.N, beta=self.beta, beta2=self.beta2, empty=1)
        for phase_tuple in self.entries.keys():
            if not self.beta2:
                new_phase_tuple = (phase_tuple*roll_int) % 1
            else:
                permuted_angle = (phase_tuple[phase_axis-1]*roll_int) % 1
                if phase_axis==1:
                    new_phase_tuple = (permuted_angle, phase_tuple[1])
                elif phase_axis==2:
                    new_phase_tuple = (phase_tuple[0], permuted_angle)

            permuted_vec.entries[new_phase_tuple] = self.entries[phase_tuple]

        return permuted_vec

    def norm(self):
        total = 0
        #if not self.beta2:
        #print "num o phases" ,len(self.entries.keys())
        for phase_tuple in self.entries.keys():
            total += np.abs(self.entries[phase_tuple])**2
        #else:
        #    for phase_tuple in self.entries.keys():

        return float(np.sqrt(total))
