import gc
from random import expovariate
from math import gamma
import os, sys, pickle, numpy, platform
from copy import copy
from math import pi, exp, log, sqrt, floor
from time import time

# Experiment Gram-Schmidt long-norms of HKZ reduced basis of determinant 1
rk_ln  = (0.547259105204595, 0.540657007681950, 0.520464961279540, 0.489722661108787, 0.482669740508587, 0.457847170113922, 0.434100555585611, 0.403051719769461, 0.383429240245152, 0.360998736849332, 0.338251899884233, 0.318529874678345, 0.287405382133861, 0.272276343099372, 0.235039538262977, 0.212492233803718, 0.191337170979269, 0.164067149717650, 0.135986009595065, 0.111745176577936, 0.0768666500439220, 0.0470135132302541, 0.0189095515409488, -0.0162619245891062, -0.0222172542126025, -0.0651787804167555, -0.0894916002054040, -0.122663057177440, -0.145149008613024, -0.184285649940731, -0.207272719472678, -0.242143063604887, -0.263692706719984, -0.296250692877672, -0.329206564266224, -0.367465512664053, -0.389288938563704, -0.424212169950291, -0.463723098558426, -0.494745397565877, -0.522661938691620, -0.560485531088975, -0.596060307036430, -0.613074793032250, -0.614590682646335)

# an example Gram-Schmidt log-norms of LLL reduced 100-dimensional basis (outputted by fplll code)
l = (17.64878,17.676655,17.520402,17.562236,17.518547,17.610344,17.473419,17.320197,17.35881,17.090739,17.208694,17.145931,17.018854,16.816176,16.882262,16.797166,16.585451,16.460784,16.400937,16.325103,16.259435,16.151939,15.995972,15.974628,15.977863,15.892194,15.787931,15.82978,15.679813,15.470669,15.420349,15.309477,15.308292,15.311121,15.082396,15.154497,14.956673,14.742087,14.681312,14.763462,14.667364,14.469891,14.478223,14.444558,14.209345,14.140888,13.939096,13.779748,13.682736,13.813108,13.868752,13.788298,13.670763,13.496685,13.538829,13.360446,13.260553,13.118916,13.238494,12.980362,12.896577,12.7718,12.984951,12.816311,12.600272,12.458473,12.465237,12.4061,12.53681,12.473932,12.212868,12.031393,11.857065,12.02787,11.889548,11.700284,11.619152,11.516841,11.497575,11.359637,11.530566,11.328076,11.16768,11.016612,11.080511,11.138445,11.001056,10.848521,10.910883,10.874056,10.761911,10.638186,10.476838,10.319336,10.237403,9.981492,9.8707546,10.035203,10.063214,9.9421573)

def gso_analyze_vol_n (n, R):
    return float(pi**(n/2)*(R**n)/gamma(n/2+1))

# bkz_simulation_stochastic (l, 45, 10, rk_ln, l_format=2)
def bkz_simulation_stochastic (l, beta, Ntours, rk, l_format=2):
    """
    This version terminates when no substantial progress is made
    anymore or at most Ntours tours were simulated.
    
    INPUT:
        -  ``l`` - natural-log norms of the GSO vectors of the basis;
        -  ``l_format`` - if the input is FPLLL format, need to set to 2;
        -  ``beta`` - blocksize of BKZ reduction
        -  ``Ntours`` - maximum number of tours to simulate. This version terminates
           early if no substantial progress is made.
        -  ``rk`` - log norms of GSO vectors of typical HKZ
    """

    n = len(l) # dimension of the input Gram-Schmidt log-norms
    
    # if in FPLLL format, l[i] should be halved
    l_fixed = [l[i]/l_format for i in range(n)]

    # setup l1, l2
    l1 = [float(0) for i in range(n)]
    l2 = [float(0) for i in range(n)]
    for i in range(n):
	l1[i] = l_fixed[i]
	l2[i] = l_fixed[i]
	
    # used for return the result
    l1_double = [l1[i]*l_format for i in range(n)]

    # (log of) multiplier in front of det^(1/n) of GH
    c = []

    for d in xrange(1, beta + 1):
        vol = gso_analyze_vol_n(d,1)
        extra_common = 0 # for future use.
        common = -log(vol)/d + extra_common
        c.append(common)

    old_touched = [True for i in range(n)]
    for tours in xrange(Ntours):
        print "# stochastic tour ", tours

	##########################################
        ### 1. front blocks
        ##########################################
        new_touched = [False for i in range(n)]
	
	for start in xrange(n-45):
            # [start, end-1]
            bs = min(beta, n-start) # bs is current block-size starting at k, thus [k, k+bs-1]
            end = start + bs        # end of local block (exclusively)

	    #current log-determinant
	    logdet = sum(l2[:end]) - sum(l2[:start])
            GH_0 = logdet/bs + c[bs-1]                

            to_be_changed = False
            for k in range(start, end):
                to_be_changed = (to_be_changed or old_touched[k])

	    radius = 0 # radius for enumeration
            if (to_be_changed):

	        #random variable distributed according to Expo[1/2]
		eps = log(expovariate(0.5))/bs
		GH = GH_0 + eps # our statistical model
                decrease = l2[start] - GH # current value compared to chosen value

		if (decrease > 0):		                                                
                    # new 2nd is set to be (bs-1)/bs times the old 1st; and average improve [start+2,end)
		    #l2_temp = l2[start+1]
		    l2[start] = GH
                    prop = (bs-1)*(bs**(-1))

		    l2[start+1] = l1[start] + log(sqrt(prop))

		    decrease2 =  decrease - (l2[start+1]-l1[start+1])

                    for k in range(start+2,end):
                        l2[k] = l2[k] + decrease2/(bs-2)
                        new_touched[k] = True
                    
	            new_touched[start] = True
	            new_touched[start+1] = True

            # update l1 for the use of updating next block
            for i in range(n): 	
	        l1[i] = l2[i]
        ##########################################
        ### 2. last 45 blocks
        ##########################################
	t = 45
	logdet = sum(l2[:n]) - sum(l2[:n-t])
        # essentially builds an HKZ reduced bases for n < t = 45
        if n < t:
            tmp = sum(rk[-n:])/n
            rk1 = [r-tmp for r in rk[-n:]]
            K = range(n)
            for k,r in zip(K, rk1):
                l2[k] = logdet/min(t,n) + r
        else:
            rk1 = rk
            K = range(n-t,n)
            to_be_changed = False 
            for i in range(n-t, n):
                to_be_changed = (to_be_changed or new_touched[i])
            change_touch = False
            if (to_be_changed):
	    	for k,r in zip(K, rk1[:45]):
		    l2[k] = logdet/min(t,n) + r
                    if ((l1[k] - l2[k]) > 1e-6):
                        change_touch = True
                if (change_touch):
                    for k in range(n-t,end):
                        new_touched[k]=True

        ##########################################
        ###  early termination (optional)
        ##########################################
	if l1 == l2:
            print "# early-abort happends since no change"
            l1_double = [l1[i]*l_format for i in range(n)]
	    
            return l1_double
        
	##########################################
        # 3. done this tour, copy
        ##########################################
        old_touched = new_touched        
	for i in range(n): 	
	    l1[i] = l2[i]
	
	l1_double = [l1[i]*l_format for i in range(n)]
	
        print "done: ", tours

    # all done, final adjustment
    l1_double = [l1[i]*l_format for i in range(n)]
    return l1_double

print sum(l)
print bkz_simulation_stochastic (l, 60, 1000, rk_ln, 2)