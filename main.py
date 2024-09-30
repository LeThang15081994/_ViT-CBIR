import torch
import matplotlib.pyplot as plt
import numpy as np

if torch.cuda.is_available():
    print('available')
    print(torch.cuda.get_device_name())
    
