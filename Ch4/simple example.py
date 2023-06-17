import numpy as np
import pylab as pl

import mlp

x = [[0.3,0.4],[0.1,0.6],[0.9,0.4]]
t = [[0.88,0.82,0.57]]

net = mlp.mlp(x,t,2)

net.mlptrain(x,t,0.2,200)
