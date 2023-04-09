#############################################################################
# class definition for trapezoidal FIR filter overlap-add
# def getLPHFIRFFT(H, order, winname=None)
# class firoverlapadd
'''
# usage example
# create an instance of class firoverlapadd
# shift 10ms, overlap 3ms
fola = firoverlapadd()
fola.setbyLPHFIRtab(len(h_a[0]), int(Fs*0.010), int(Fs*0.003))
y = fola.doFilterAll(h_a, sinusoid)
# y is the online output stream (with delay 3ms)
'''
#
# DEEE725 Speech Signal Processing Lab
# 2023 Spring, Kyungpook National University
# Instructor: Gil-Jin Jang
# Lab 03 FIR filter design
# references:
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.firwin.html
# https://coding-yoon.tistory.com/23
#############################################################################

import numpy as np
import matplotlib.pyplot as plt
import librosa

#######################################################
# re-definition of getFIRFFT to
# 1. make sure that it generates linear phase (symmetric real) filter
# 2. be renamed to indiate the above
# 3. return the linear phase filter and the delay incurred by the filtering
def getLPHFIRFFT(H, order, winname=None):
    # H: NFFT/2+1
    NFFT = (len(H)-1)*2
    H = np.concatenate((H, H[-2:0:-1])) + np.zeros(NFFT)*(1j)
    h = np.fft.ifft(H, NFFT)

    # adjust order if necessary
    order = min(NFFT-1, order)  # at most NFFT samples
    delay = order//2     # delay by the filtering is half the order
    order = delay*2      # odd order -> even so that the filter is symmetric w.r.t. the center sample

    #h = np.concatenate((h[len(h)//2:], h[0:len(h)//2]))
    h = np.concatenate((h[(len(h)-delay):], h[0:delay+1]))
    if winname != None:
        win = librosa.filters.get_window(winname, order+1, fftbins=False)
        h = h*win
    return h.real, delay

class firoverlapadd:
    def __init__(self):
        self.initializeAllVar()

    def initializeAllVar(self):
        # initially, all empty
        self.Ns = 0
        self.Nov = 0
        self.M = 0
        self.FIRshift = 0
        self.rwin = np.zeros(0)
        self.lwin = np.zeros(0)
        self.lastNx = 0   # length of last x input
        self.lastNy = 0   # length of last y output
        self.xbuf = np.zeros(0)
        self.ybuf = np.zeros(0)
        self.yout = np.zeros(0)

    # set the parameters using number of samples
    # Ns, Nov: do not change when the given values are less than or equal to 0
    # order, FIRshift: do not change when the given values are less than 0 (0 is valid)
    def set(self, Ns=0, Nov=0, order=-1, FIRshift=-1):
        if Ns>0: self.Ns = Ns
        if Nov>0: self.Nov = Nov
        if order>=0: self.M = order
        if FIRshift>=0: self.FIRshift = FIRshift
        self.allocBuffer()

    # find order and FIRshift from the linear phase filter
    def setbyLPHFIRtab(self, ntab, Ns=0, Nov=0):
        self.set(Ns, Nov, self.getOrderByFilter(ntab), self.getLinearPhaseDelay(ntab))

    # set the order by an FIR filter
    def getOrderByFilter(self, ntab): 
        return ntab-1
    # set the FIR shift by a linear phase FIR filter
    def getLinearPhaseDelay(self, ntab):
        return (ntab-1)//2
    # delay of the output
    def getDelay(self): 
        return self.Nov+self.FIRshift
        #return self.Nov

    # allocate memory and buffers, assuming that the numbers are already set properly
    def allocBuffer(self):
        self.rwin = np.linspace(1,0,self.Nov+2)[1:-1]
        self.lwin = np.linspace(0,1,self.Nov+2)[1:-1]
        self.xbuf = np.zeros(self.M+self.Nov+self.Ns)
        self.ybuf = np.zeros(self.M+self.Nov+self.Ns)
        self.yout = np.zeros(self.Ns)
        self.lastNx = 0
        self.lastNy = self.getDelay()

    # clear all the memory and buffers
    def resetBuffer(self):
        self.xbuf[:] = 0
        self.ybuf[:] = 0
        self.yout[:] = 0
        self.lastNx = 0
        self.lastNy = self.getDelay()

    # single frame processing
    def doFilterMem(self, h, x):
        ############################################
        # elaborate input x
        # x-1. shift buffer
        for ii in range(self.M+self.Nov):
            self.xbuf[ii] = self.xbuf[ii+self.Ns]
            
        # x-2. copy the new input
        self.lastNx = len(x)
        if len(x) > 0:
            for ii in range(len(x)):
                self.xbuf[ii+self.M+self.Nov] = x[ii]
            for ii in range(len(x),self.Ns):
                self.xbuf[ii+self.M+self.Nov] = 0
        else:   # no input, clear buffer
            self.xbuf[(self.M+self.Nov):] = np.zeros(self.Ns)

        ############################################
        # generate output y
        # y-1. copy the overlapped old output, overlapped
        self.yout[:self.Nov] = self.ybuf[(-self.Nov):]
        
        # y-2. do filtering (if necessary)
        if len(h) > 1: 
            # if do assignment as follows, exception unless 
            # the lengths of left and right match (to prevent bug)
            self.ybuf[:] = signal.lfilter(h, [1], self.xbuf)
        elif len(h) == 1:  # simple amplifier
            self.ybuf[:] = h[0]*self.xbuf[:]
        else:   # no filter at all
            self.ybuf[:] = np.zeros(self.M+self.Nov+self.Ns)
        
        # y-3. multiply trapezoidal window
        self.ybuf[self.M:(self.M+self.Nov)] *= self.lwin
        self.ybuf[(-self.Nov):] *= self.rwin
        
        # y-4. overlap-add
        self.yout[:self.Nov] += self.ybuf[self.M:(self.M+self.Nov)]
        self.yout[self.Nov:self.Ns] = self.ybuf[(self.M+self.Nov):(self.M+self.Ns)]

        '''
        plt.figure(figsize=FIG_SIZE*np.array([1.8,0.5]))
        plt.subplot(2,1,1)
        plt.plot(self.ybuf[self.M:])
        plt.subplot(2,1,2)
        plt.plot(self.yout)
        '''
        
        # y-5. update output buffer length
        self.lastNy = self.lastNy + self.lastNx
        len_out = min(self.lastNy, self.Ns)
        #print('lastNx = %d, lastNy = %d, Ns = %d, len_out = %d' % (self.lastNx, self.lastNy, self.Ns, len_out))
        self.lastNy -= len_out

        # output and actual length
        return self.yout, len_out
    
    # all audio signal processing
    def doFilterAll(self, H, X):
        y = np.zeros(0)   # start from empty signal
        Ly, Lx, n = 0, len(X), 0
        while True:
            # 1. input
            t1,t2 = min(Lx,n*self.Ns),min(Lx,(n+1)*self.Ns)
            x = X[t1:t2]

            # 2. filtering with memory and overlap add
            if n >= len(H): h = []
            else: h = H[n]
            [yt,Lyt] = self.doFilterMem(h, x)

            # 3. process one frame and append it
            if Lyt > 0:
                Ly += Lyt
                y = np.concatenate((y, yt[:Lyt]))
            else: break
            n+=1
        return y

