import torch
import matplotlib.pyplot as plt
import numpy as np
from transformers import ViTImageProcessor, ViTForImageClassification
import cv2
import pickle
import os
import logging
from datetime import datetime
from vit_cls_extract import vit_feature_extract