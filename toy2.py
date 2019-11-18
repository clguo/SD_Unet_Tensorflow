from __future__ import division, print_function
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
plt.rcParams['image.cmap'] = 'gist_earth'
np.random.seed(98765)
from tf_SDUnet import image_gen
from tf_SDUnet import unet
from tf_SDUnet import util
nx = 572
ny = 572
generator = image_gen.GrayScaleDataProvider(nx, ny, cnt=20)

x_test, y_test = generator(1)
print(x_test[0,...,0])