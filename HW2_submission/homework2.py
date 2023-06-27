import scipy.io
from AccMeasure import acc_measure
from mycluster import cluster
import numpy as np
# from mycluster_extra import cluster_extra
# from show_topics import display_topics

mat = scipy.io.loadmat('data.mat')
mat = mat['X']
X = mat[:, :-1]


idx = cluster(X, 4)

acc = acc_measure(idx)

print('accuracy %.4f' % (acc))
