# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 15:43:17 2023

@author: jseys
"""

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))