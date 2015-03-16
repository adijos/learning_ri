import os
import unittest
import shutil
import string
import numpy as np

from iv_structure.infinite_vectors import Vector

class TestVector(unittest.TestCase):

    def setUp(self):
        os.chdir(os.path.dirname(__file__))
        if os.path.exists("vectors"):
            shutil.rmtree("vectors")
        os.mkdir("vectors")

    def tearDown(self):
        if os.path.exists("vectors"):
            shutil.rmtree("vectors")

    def test_vectors(self):
        print " ONE PHASE! "
        print '\nparameters'
        print "========================"
        print "===   rational case  ==="
        print "========================"
        N = 10.
        beta = 1/N
        print 'N:', N, ', beta:', beta
        print "~~~~~~~"

        v = Vector(N=N, beta=beta)
        w = Vector(N=N, beta=beta)
        print 'v', v, 'norm', v.norm(), '\n'
        print 'w', w, 'norm', w.norm()

        print 'v_normed', v/v.norm()

        print "~~~~~~~"
        print "testing addition"
        print "v+w", v + w

        print "~~~~~~~"
        print "testing multiplication"
        print "v*w", v.mult(w)

        print "~~~~~~~"
        print "testing permutation"
        print "permute(v)", v.permute(1/N)

        print "========================"
        print "===  irrational case ==="
        print "========================"
        N = 10.
        beta = 1/np.exp(1)
        print 'N:', N, ', beta:', beta
        print "~~~~~~"

        v = Vector(N=N, beta=beta)
        w = Vector(N=N, beta=beta)
        print 'v', v, 'norm', v.norm(), '\n'
        print 'w', w, 'norm', w.norm()
        print 'v_normed', v/v.norm(), 'norm', (v/v.norm()).norm()

        print "~~~~~~~"
        print "testing addition"
        print "v+w", v + w

        print "~~~~~~~"
        print "testing multiplication"
        print "v*w", v.mult(w)

        print "~~~~~~~"
        print "testing permutation"
        print "permute(v)", v.permute(1/100.)
        print 'v', v

        print "++++++++++++++++++++++++++"
        print "~~~~~~~~~~~~~~~~~~~~~~~~~~"
        print "++++++++++++++++++++++++++"
        print " TWO PHASES! "
        print '\nparameters'
        print "========================"
        print "===   rational case  ==="
        print "========================"
        N = 10.
        beta = 1/N
        beta2 = 1/N
        print 'N:', N, ', beta:', beta, ', beta2:', beta2
        print "~~~~~~~"

        v = Vector(N=N, beta=beta, beta2=beta2)
        w = Vector(N=N, beta=beta, beta2=beta2)
        print 'v', v, '\n'
        print 'w', w

        print "~~~~~~~"
        print "testing addition"
        print "v+w", v + w

        print "~~~~~~~"
        print "testing multiplication"
        print "v*w", v.mult(w)

        print "~~~~~~~"
        print "testing permutation over first axis"
        print "permute(v)", v.permute(1/N, phase_axis=1)

        print "~~~~~~~"
        print "testing permutation over second axis"
        print "permute(v)", v.permute(1/N, phase_axis=2)

        print "\n========================"
        print "===  irrational case ==="
        print "========================"
        N = 2.
        beta = 1/np.exp(1)
        beta2 = (1/np.exp(2)) % 1
        print 'N:', N, ', beta:', beta, ' beta2: ', beta2
        print "~~~~~~"

        v = Vector(N=N, beta=beta, beta2=beta2)
        w = Vector(N=N, beta=beta, beta2=beta2)
        print 'v', v, 'norm', v.norm(), '\n'
        print 'w', w, '\n'
        print 'v_normed', v/v.norm(), 'normed norm', (v/v.norm()).norm()

        print "~~~~~~~"
        print "testing addition"
        print "v+w", v + w

        print "~~~~~~~"
        print "testing multiplication"
        print "v*w", v.mult(w)

        print "~~~~~~~"
        print "testing permutation"
        print "permute(v)", v.permute(1/100.)
        print 'v', v
