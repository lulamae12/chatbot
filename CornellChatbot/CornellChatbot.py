import pandas as pd #pandas datareader
import numpy as np
import tensorflow as tf
import re
import time
tf.__version__

#loads cornell data
lines = open('movie_lines.txt', encoding='utf-8', errors='ignore').read().split('\n')
conv_lines = open('movie_conversations.txt', encoding='utf-8', errors='ignore').read().split('\n')

#sentences to train
lines[:10]
conv_lines[:10]
