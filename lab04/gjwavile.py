#############################################################################
# wavfile collection
# def writewav(filename, samplerate, data, maxval=1.0)
#   save wavfile
#
#def showwav(filename)
#   show wavfile play widget in jupyter notebook
#
# DEEE725 Speech Signal Processing Lab
# 2023 Spring, Kyungpook National University
# Instructor: Gil-Jin Jang
# Lab 03 FIR filter design
# references:
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.firwin.html
# https://coding-yoon.tistory.com/23
#############################################################################

import scipy 
import numpy as np
import IPython

def writewav(filename, samplerate, data, maxval=1.0):
    data = data/maxval*(2**15)    # 16 bit
    scipy.io.wavfile.write(filename, samplerate, data.astype(np.int16))

def showwav(filename):
    IPython.display.Audio(filename)
