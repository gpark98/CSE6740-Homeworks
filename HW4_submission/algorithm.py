import matplotlib.pyplot as plt
import numpy as np


def algo(q, Y):
    # init
    p = 0.0
    fig, ax = plt.subplots()

    # TODO implement your algorithm and return the (i) prob p and (ii) a matplotlib Figure object for the plot

    Y = Y.reshape(39,)

    # forward
    memo1 = {}
    memo2 = {}

    memo1[1] = 0.2*(q+(2*q-1)/2*(Y[0]-1))
    memo2[1] = 0.8*(q-(2*q-1)/2*(Y[0]+1))

    def alpha(l):
        if l in memo1:
            return memo1[l]
        else:
            memo1[l] = (alpha(l-1)*0.8 + beta(l-1)*0.2)*(q+(2*q-1)/2*(Y[l-1]-1))
            return memo1[l]


    def beta(l):
        if l in memo2:
            return memo2[l]
        else:
            memo2[l] = (alpha(l-1)*0.2 + beta(l-1)*0.8) *(q-(2*q-1)/2*(Y[l-1]+1))
            return memo2[l]


    p_d = alpha(39) + beta(39)


    # backward
    memo3 = {}
    memo4 = {}

    memo3[39] = 1
    memo4[39] = 1

    def gamma(l):
        if l in memo3:
            return memo3[l]
        else:
            memo3[l] = 0.8*gamma(l+1)*(q+(2*q-1)/2*(Y[l]-1))+ 0.2*delta(l+1)*(q-(2*q-1)/2*(Y[l]+1))
            return memo3[l]


    def delta(l):
        if l in memo4:
            return memo4[l]
        else:
            memo4[l] = 0.2*gamma(l+1)*(q+(2*q-1)/2*(Y[l]-1))    + 0.8*delta(l+1)*(q-(2*q-1)/2*(Y[l]+1))
            return memo4[l]


    prob = {}

    for t in range(1, 40):
        prob[t] = alpha(t)*gamma(t)/p_d


    a = prob.values()

    plt.plot(a)

    p = prob[39]


    return p, fig