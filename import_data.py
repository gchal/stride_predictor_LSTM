
import numpy as np
import torch
import os

import scipy.io as scpio


np.random.seed(2)



# data = scpio.loadmat('data_strides'+'.mat')
# data=np.array(list(data['dataN'])).astype(np.float)
data = scpio.loadmat('data_phi'+'.mat')
data=np.array(list(data['phi'])).astype(np.float)
a=np.reshape(data, (10, -1))
print(a.shape)


# torch.save(a, open('traindata.pt', 'wb'))

torch.save(a, open('traindata1.pt', 'wb'))
