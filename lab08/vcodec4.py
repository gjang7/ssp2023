# vocoder LPC / LSF
# 2021 1 22 Gil-Jin Jang
#
# updated, version 2 on 2021 2 13
# LSF, LPC coef, Residual dimension tranposed 
# to match sklearn's and plot methods, which saves computation/memoy
# ex) LSF: lpcorder x N -> N x lpcorder
#
# updated, version 3 on 2021 2 14
# - fix: excitation is re-encoded by mapped (coded) LSF
#
# updated, version 4 on 2021 2 14
# - using class 
# - lsfvoc.py -> vcodec.py

import sys
import numpy as np
from scipy.io import wavfile
import librosa
from matplotlib import pyplot as plt
import pickle


####################################################
# Non-member method
####################################################
def readfile_lsfcodebook(codebookfile):
    if codebookfile != None:
        with open(codebookfile, 'rb') as f: return pickle.load(f)
    return None # failure to load file

def writefile_lsfcodebook(codebookfile, codebook):
    if codebookfile != None: 
        with open(codebookfile, 'wb') as f: 
            pickle.dump(codebook,f)
            return True
        print("cannot open codebookfile '%s'"%codebookfile)
    else:
        print('codebookfile is None')
    return False

def LPCOrderFs(self,Fs): 
    # find LPC order according to Fs
    # these values are just suggestion
    # 10 LSFs for 16kHz are also fine
    if Fs == 8000: return 10
    elif Fs == 16000: return 14
    elif Fs == 44100 or Fs == 48000: return 20
    else: 
        print("unknown sampling rate %d"%Fs)
        return 10

def NSamplesPerShift(SamplingRate,FrameShiftInSeconds):
    return int(float(SamplingRate)*float(FrameShiftInSeconds))

####################################################
# class definition for shared configuration
####################################################
class Config:
    # default initialization values
    alpha = 0.98
    xepsilon = 1.0e-3
    #xepsilon = 1.0e-5
    #xepsilon = 1.0e-10
    minpsd_plot =1.0e-10    # minimum power spectral density, drawing only
    cmap_plot = plt.cm.bone # default colormap for spectrogram, gray

    # added on 2/14/2021
    Fs = 0  # if 0, it means uninitialized
    Tsht = 0.01    # 10ms, de factor standard
    M = 0   # LPC order = LSF dimension, 0 for being uninitialized
    LCFile = None  # name of the codebook file

    # added on 2/20/2021
    REncMode = 0   # do not encode residual, initially

    # reserved methods
    def __init__(self): pass    #NOP

    def __repr__(self):
        s = "vcodec:Config:Fs=%d Tsht=%.4f"%(self.Fs,self.Tsht)
        s += " Ns=%d lpcorder=%d"%(self.Ns(),self.M)
        if self.LCFile != None: s += (" LSFcodebook="+self.LCFile)
        return s

    # methods
    def Ns(self):   # number of samples per shift
        #return int(float(self.Fs)*self.Tsht)
        return NSamplesPerShift(self.Fs,self.Tsht)

    def load_lsfcodebook(self): 
        return readfile_lsfcodebook(self.LCFile)

    def set(self, Fs=0, Tsht=0.0, lpcorder=0, lsfcodefile=None, remode=0):
        # do not change when 0 -> enables selective changing
        if Tsht>0.0: self.Tsht = Tsht   
        if Fs>0: self.Fs = Fs 
        if lpcorder>0: self.M = lpcorder
        # for lsfcodefile, assign None means no code mapping
        # so do not check if it is None
        self.LCFile = lsfcodefile
        self.REncMode = remode

    def setbyfile(self, config_file):
        # read the configuration from file 
        # only fileds that exist in the file will be updated
        # format example:
        #   Fs = 16000 --> self.Fs = 16000
        #   # adding comments is allowed
        #   M = 14 --> self.M = 14
        #   ...
        num_configs = 0
        with open(config_file, 'r') as f: 
            for line in f:
                #while True:
                #line = f.readline()
                if not line: break
                elif line[0] != '#':  # not comment lines
                    line = line.rstrip('\n')
                    line = line.rstrip('\r')
                    print(line)
                    exec("self."+line)  # no 'eval' but 'exec'
                    num_configs += 1
                print(self)
        return num_configs

####################################################
# end of class Config
####################################################


####################################################
# class definition for FIR/IIR filter memory
####################################################
class FilterMemory:
    # default initialization values
    preem = 0.0  # preemphasis memory
    deem = 0.0   # deemphasis memory
    lpcxpn_in = None
    lpcxpn_hat = None
    lpcxbuf = None
    xpnbuf = None
    xpnwbuf = None

    # 2021 2 14, for LSF-coded re-encoding
    xpn_prev = None
    lpcxpn_in_prev = None

    # 2021 2 14, for decoding
    framebuf = None

    # reserved methods
    def __init__(self): pass    #NOP

    def __repr__(self):
        s = "vcodec:FilterMemory: "
        s += "preem="+str(self.preem)+" deem="+str(self.deem) + '\n'
        s += 'lpcxpn_in = ' + str(self.lpcxpn_in) + '\n'
        s += 'lpcxpn_hat' + str(self.lpcxpn_hat) + '\n'
        s += 'lpcxbuf' + str(self.lpcxbuf) + '\n'
        s += 'xpnbuf' + str(self.xpnbuf) + '\n'
        s += 'xpnwbuf' + str(self.xpnwbuf) + '\n'
        s += 'framebuf' + str(self.xpnwbuf) + '\n'
        return s

    # ordinary methods
    def alloc(self, lpcorder, Ns):
        self.preem = 0.0  # preemphasis memory
        self.deem = 0.0   # deemphasis memory
        self.lpcxpn_in = np.zeros(lpcorder,dtype='float')
        self.lpcxpn_hat = np.zeros(lpcorder,dtype='float')
        self.lpcxbuf = np.zeros(lpcorder+Ns,dtype='float')

        # 21 01 21 bugfix, for hamming windowing
        # before multiplication, to save memory for the previous frame,  and 
        self.xpnbuf = np.zeros(Ns*2,dtype='float')
        # to store window multiplication
        self.xpnwbuf = np.zeros(Ns*2,dtype='float')

        # 2021 2 14, for LSF-coded re-encoding
        self.xpn_prev = np.zeros(Ns,dtype='float')
        self.lpcxpn_in_prev = np.zeros(lpcorder,dtype='float')

        # 2021 2 14, for decoding
        self.framebuf = np.zeros(Ns,dtype='float')

    def clear(self):
        self.preem = 0.0  # preemphasis memory
        self.deem = 0.0   # deemphasis memory
        self.lpcxpn_in.fill(0.0)
        self.lpcxpn_hat.fill(0.0)
        self.lpcxbuf.fill(0.0)

        # 21 01 21 bugfix, for hamming windowing
        # before multiplication, to save memory for the previous frame,  and 
        self.xpnbuf.fill(0.0)
        # to store window multiplication
        self.xpnwbuf.fill(0.0)

        # 2021 2 14, for LSF-coded re-encoding
        self.xpn_prev.fill(0.0)
        self.lpcxpn_in_prev.fill(0.0)

        # 2021 2 14, for decoding
        self.framebuf.fill(0.0)

####################################################
# end of class FilterMemory
####################################################


####################################################
# Encoder class definition 
####################################################
class Encoder:
    # transmitting information
    T = 0           # 1 int, length of the encoded signal
    K = 0           # 1 int, number of frames
    LSFcode = None  # LSFcode: K int
    G = None        # excitation gain: K real
    VUV = None      # vuv flag: K boolean
    P = None        # pitch of voiced excitation: K int

    # The following variables are 
    # LSF = None: when lsfcode != None
    # R = None: when encoded by vuv, gain, pitch, etc.
    LSF = None  # envelope: LSF KxM real
    R = None    # excitation: residual KxNs real

    # not for transmission, but for encoding/decoding
    cfg = None  # class Config object
    mem = None  # class FilterMemory
    lsfcodebook = None  # sklearn's KMeans

    #################################
    # auto: create config and memory first
    def __init__(self):
        self.cfg = Config()
        self.mem = FilterMemory()

    def __repr__(self): 
        return str(cfg) + str(mem) + str(lsfcodebook)

    #################################
    # methods
    def getREmode(self): return self.cfg.REncMode

    # returns raw LSF / coded LSF depending on lsfcodebook
    def getLSFc(self,k): 
        if self.lsfcodebook == None: return self.LSF[k,:]
        else: return self.lsfcodebook.cluster_centers_[self.LSFcode[k],:]

    # shared by set and setbyfile
    def _set_sub(self):
        # alloc(self, lpcorder, nsamplespershift):
        self.mem.alloc(self.cfg.M,self.cfg.Ns())
        self.lsfcodebook = self.cfg.load_lsfcodebook()
        print(self.lsfcodebook)

    #################################
    # step 1: these set methods should be called at least once
    def set(self, Fs, Ts, M, lcfile=None, REncMode=0):
        # set(self, Fs=16000, Tsht=0.01, lpcorder=10, lsfcodefile=None):
        self.cfg.set(Fs,Ts,M,lcfile,REMode)
        self._set_sub()

    # set by config file
    def setbyfile(self, config_file):
        self.cfg.setbyfile(config_file)
        self._set_sub()

    ####################################################
    # (x) Env,Exitation,residual = lsfencoder(sig, config, memory, lsfcodebook)
    # residual = lsfencoder(sig)
    # step 2. this method should be called after set() or setbyfile()
    ####################################################
    def lsfencoder(self, sig):
        M = self.cfg.M  # LPC order
        Ns = self.cfg.Ns()  # N shift
        self.T = sig.shape[0]    # signal length
        self.K = (self.T+Ns-1)//Ns    # ceiling
        K = self.K

        #self.enc.T = sig.shape[0]
        # moved to Config
        #Ns = int(float(Fs)*Tsht)  # number of samples per shift
        #memory = initmem(lpcorder,Ns)
        #config = initconfig(lpcorder,Ns,Fs)

        # clear (initialize) memory contents 
        self.mem.clear()
        print('%s: Fsamp=%d, len(sig)=%d, shift=%d no_frames=%d'%(__name__,self.cfg.Fs,self.T,self.cfg.Ns(),self.K))

        # allocate envelope parameters 
        #A = np.zeros([K,M+1],dtype='float')   # LPC order + 1
        #E = np.zeros([K,Ns],dtype='float')   # same as shift size
        if self.lsfcodebook != None:
            # LSF code: integer/frame
            self.LSFcode = np.zeros(K,dtype='int') 
            self.LSF = None
        else:
            # LSF: LPC order/frame
            self.LSF = np.zeros([K,M],dtype='float') 
            self.LSFcode = None

        # allocate residual parameters 
        if self.getREmode() == 0:
            self.R = np.zeros([K,Ns],dtype='float')   # same as shift size
        else:
            self.R = None   # do not store residual signal 
            self.G = np.zeros(K,dtype='float')   # frame gain
            self.VUV = np.zeros(K,dtype='bool')   # Voiced/Unvoiced flag
            self.P = np.zeros(K,dtype='int')   # pitch values

        # random seed
        np.random.seed(self.T)

        # A short-time framing
        #frame = np.zeros(Ns,dtype='float')
        frame = self.mem.framebuf
        residual = np.zeros(sig.shape,dtype='float')

        # LPC encoding and obtaining residual signals 
        for k in range(K):
            istart = k*Ns
            iend = min((k+1)*Ns,self.T)
            #Ns_last = iend-istart   # keep the length for the last frame
            frame[0:(iend-istart)] = sig[istart:iend]
            frame[(iend-istart):] = 0  # zero-padding for the last frame only
            #A[:,k], E[:,k] = lpcencode_frame(frame,M,self.cfg,self.mem)
            #residual[istart:iend] = E[:(iend-istart),k]
            #a_, e_ = lpcencode_frame(frame,lpcorder,self.cfg,self.mem)
            #a_, e_ = self.lpcencode_frame(frame,M)

            # extract basic LPC and residual from a frame
            a_, e_ = self.lpcencode_frame(frame)

            # convert to LSFcode, LSF, residual
            c_, l_, e_ = self.lpc2lsf_residual(a_,e_)

            # store code or LSF, not both
            if c_ != -1: self.LSFcode[k] = c_
            else: self.LSF[k,:] = l_

            # store residual 
            if self.getREmode() == 0: self.R[k,:] = e_
            elif self.getREmode() == 1:
                # TODO: encode residual
                pass
            else: 
                print("unknown residual encoding method: %d"%(self.getREmode()))

            # 1-dim residual to return
            residual[istart:iend] = e_[:(iend-istart)]

        #return self.Env,self.Ext,residual
        return residual

    ####################################################
    def lpcencode_frame(self,x):
        Ns = len(x) # shift size
        lpcorder = self.cfg.M

        # B pre-emphasis
        #xp,memory['preem'] = preemphasis(x,self.cfg.alpha,self.mem.preem)
        #xp,xmem = preemphasis(x,self.cfg.alpha,self.mem.preem)
        xp,xmem = preemphasis(x,self.cfg.alpha,self.mem.preem)
        self.mem.preem = xmem

        # C tiny noise addition
        #print(max(abs(x)))
        #print(max(abs(xp)))
        #xpn = xp + np.random.normal(0,self.cfg.xepsilon,Ns)
        xpn = self.mem.xpnbuf
        xpn[:Ns] = xpn[Ns:]  # shift
        xpn[Ns:] = xp + np.random.normal(0,self.cfg.xepsilon,Ns)
        self.mem.xpnbuf = xpn  # may not be necessary

        # D hamming windowing
        #xpnw = np.multiply(xpn,np.hamming(Ns)
        # bugfix 21/1/21 
        xpnw = self.mem.xpnwbuf
        np.multiply(xpn,np.hamming(Ns*2),out=xpnw)

        # E LPC
        a = librosa.lpc(xpnw,lpcorder)  # a: [lpcorder+1,1]

        # 2021 2 14, for LSF-coded re-encoding
        self.mem.xpn_prev[:] = xpn[Ns:]
        self.mem.lpcxpn_in_prev[:] = self.mem.lpcxpn_in

        # F LPC encoding  e[t] = a[t]*x[t]
        #e,xmem = lpcresidual_frame(xpn,a,self.mem.lpcxpn_in,self.mem.lpcxbuf)
        e,xmem = lpcresidual_frame(xpn[Ns:],a,self.mem.lpcxpn_in,self.mem.lpcxbuf)
        self.mem.lpcxpn_in = xmem

        return a,e

    ####################################################
    def lpc2lsf_residual(self,a_,e_):
        # convert to LSF
        l_ = poly2lsf(a_)

        # convert to LSF code, if possible
        if self.lsfcodebook != None: 
            c_ = self.lsfcodebook.predict(np.reshape(l_,(1,-1)))
            lc_ = self.lsfcodebook.cluster_centers_[c_,:]
            # get residual from the converted LPC
            a_ = lsf2poly(np.reshape(lc_,-1))
            # re-encode exitation (residual)
            e_,_ = lpcresidual_frame(self.mem.xpn_prev,a_,
                    self.mem.lpcxpn_in_prev,self.mem.lpcxbuf)
        else:
            c_ = -1     # not encoded 
            
        # returns code, residual, lsf (optional)
        return c_, l_, e_

    ####################################################
####################################################
# end of class Encoder
####################################################


####################################################
# Decoder class definition 
####################################################
class Decoder:
    ####################################################
    # decoded, A_decoded = lsfdecoder(Env, Ext, T, config, memory)
    ####################################################
    #def lsfdecoder(Env, Ext, T, config, memory, lsfcodebook):
    def lsfdecoder(self, enc):
        # G decoding
        # H deemphasis
        Ns = enc.cfg.Ns()
        K = enc.K

        # clear (initialize) memory contents 
        enc.mem.clear()
        print('%s: Fsamp=%d, len(sig)=%d, shift=%d no_frames=%d'%(__name__,enc.cfg.Fs,enc.T,enc.cfg.Ns(),enc.K))

        #frame = np.zeros(Ns,dtype='float')
        frame = enc.mem.framebuf
        decoded = np.zeros(enc.T,dtype='float')
        for k in range(enc.K):
            # code to LSF, LSF to LPC
            l_ = enc.getLSFc(k)
            a_ = lsf2poly(np.reshape(l_,-1))

            # decode residual 
            if enc.getREmode() == 0: e_ = np.reshape(enc.R[k,:],-1)
            elif self.getREmode() == 1:
                # TODO: encode residual
                pass
            else: 
                print("unknown residual encoding method: %d"%(self.getREmode()))

            #lpcdecode_frame(frame,A[:,k],E[:,k],config,memory)
            #lpcdecode_frame(frame,A[k,:],E[k,:],config,memory)
            #lpcdecode_frame(frame,a_,e_,config,memory)
            self.lpcdecode_frame(enc,frame,a_,e_)
            istart = k*Ns
            iend = min((k+1)*Ns,enc.T)
            decoded[istart:iend] = frame[0:(iend-istart)]
            #print(np.fix(decoded[istart:iend]))
            #print(sig[istart:iend])

        return decoded


    ####################################################
    # extract LPC coefficients for drawing purposes
    def getLSFLPCMat_plot(self, enc):
        M = enc.cfg.M
        #L = np.zeros([M,enc.K],dtype='float')
        #A = np.zeros([M+1,enc.K],dtype='float')
        L = np.zeros([enc.K,M],dtype='float')
        A = np.zeros([enc.K,M+1],dtype='float')
        for k in range(enc.K):
            # code to LSF, LSF to LPC
            l_ = enc.getLSFc(k)
            a_ = lsf2poly(np.reshape(l_,-1))
            #L[:,k] = np.reshape(l_,[M])
            #A[:,k] = np.reshape(a_,[M+1])
            L[k,:] = np.reshape(l_,[M])
            A[k,:] = np.reshape(a_,[M+1])

        return L,A


    ####################################################
    ####################################################
    def lpcdecode_frame(self,enc,xout,a,e):
        # xout: saving results
        # a: lpcorder+1 (lpcorder is the order of lpc)
        # e: Ns (number of samples per shift), residual signal

        Ns = len(e)
        M = len(a)-1

        # filtering
        #xpn = np.zeros(Ns+M,dtype='float')
        # re-use buffer
        xbuf = enc.mem.lpcxbuf    # shape [Ns+M,]
        xbuf[range(M)] = enc.mem.lpcxpn_hat
        #print(enc.mem.lpcxpn_hat)
        #print(xbuf)
        for t in range(len(e)):
            #xbuf[M+t] = e[t]-np.multiply(a[2:],xbuf[np.arange(t+M-1,t,-1)]).sum()
            #xbuf[M+t] = e[t]-np.multiply(a[range(M,0,-1)],xbuf[t:(M+t)]).sum()
            xbuf[M+t] = e[t]-np.multiply(a[range(M,0,-1)],xbuf[t:(M+t)]).sum()/a[0]
        #print(xbuf)
        enc.mem.lpcxpn_hat = xbuf[(len(xbuf)-M):]

        # de-emphasis
        xout[:], xdmem = deemphasis(xbuf[M:],enc.cfg.alpha,enc.mem.deem)
        enc.mem.deem = xdmem
        #xout[:] = xbuf[M:]

        return xout

    ####################################################
    # end of class Decoder
    ####################################################


####################################################
# Encoding-Decoding (codec) demo class definition 
####################################################
class VCodecDemo:
    # encoder and decoder
    enc = None
    dec = None

    # just for SNR and drawing
    #sig = None
    #residual = None

    def __init__(self):
        self.enc = Encoder()
        self.dec = Decoder()

    ####################################################
    # Env,Exitation,residual = wavfileencoder(file)
    ####################################################
    # TODO: check Fs to be matched lsfcodebook
    def wavfileencoder(self, infile, cfgfile):
        # initialize configuration
        self.enc.setbyfile(cfgfile)

        # read data from a file
        Fs, sig = wavfile.read(infile)

        if sig.ndim>1 and sig.shape[1]>1:
            print('warning: file "' + infile + '" has multiple channles ('
                    + str(sig.shape) + '), taking only the first channel')
            sig = np.reshape(sig[:,0],-1)

        if self.enc.cfg.Fs <= 0:    # Fs is 0 --> uninitialized
            #self.enc.cfg.set(Fs,self.enc.cfg.Ts)
            print('error: Fs is not set')
            exit()
        elif Fs != self.enc.cfg.Fs: 
            print('error: Fs mismatch %d != %d'%(Fs,self.enc.cfg.Fs))
            exit()

        ######## F F-2, F-3 LPC, LPC2LSF, LSF2code ########
        # Env: LSF or LSF code (when lsfcodebook is not None)
        #Env,Ext,residual = lsfencoder(sig, config, memory, lsfcodebook)
        #enc = (Env,Ext,T,memory,config)

        # input: sig, output: Env, Ext, residual
        residual = self.enc.lsfencoder(sig) 

        return sig, residual
        ################################################################
        # ENCODING IS FINISHED
        ################################################################

    ####################################################
    def wavfiledecoder(self, enc, outfile=None):
        ################################################ 
        # (x)enc = (L,E,T,memory,config) or (C,E,T,memory,config)
        # enc: class Encoder

        """
        Env = enc[0]
        Ext = enc[1]
        T = enc[2]
        memory = enc[3]
        config = enc[4]
        lpcorder = config["lpcorder"]
        Ns = config["Ns"]
        Fs = config["Fs"]
        print("lpcorder=%d Ns=%d Fs=%d"%(lpcorder,Ns,Fs))
        """

        ######## G-0 G-1 G-2 LSF code to LPC conversion ########
        #decoded = lsfdecoder(Env, Ext, T, config, memory, lsfcodebook)
        decoded = self.dec.lsfdecoder(enc)

        # save to outfile unless it is None
        if outfile != None:
            print('writing output to %s'%(outfile))
            wavfile.write(outfile, enc.cfg.Fs, decoded.astype(np.int16))

        return decoded


    ####################################################
    def wavfilevocoder(self,infile, cfgfile=None, outfile=None):
        ##################################################
        # 1) encoding test
        sig, residual = self.wavfileencoder(infile, cfgfile)

        ##################################################
        # 2) decoding test
        decoded = self.wavfiledecoder(self.enc, outfile)

        ##################################################
        # 3) Compute SNR
        #print(decoded)
        # compute SNR: signal = input, noise = input-output (prediction error)
        # snr = sqrt((signal)^2/noise^2
        # SNR in dB = 10*log10(s^2/e^2)
        #           = 20*log10(abs(s)/abs(e))
        reconerr = sig - decoded  # reconstruction error
        snr = 10*np.log10((sig**2).sum()/(reconerr**2).sum())
        print('M = %d SNR = %.3f dB'%(self.enc.cfg.M,snr))
        ##################################################

        #wavfilevocoder_plot(enc, sig, residual, decoded, 
        #lsfcodebook=lsfcodebook, wavfilename=infile)
        self.wavfilevocoder_plot(sig, residual, decoded, infile)

        return snr, sig, sig, residual, decoded


    ################################################################
    ######## PLOTTING ENCODING/DECODING RESULTS ####################
    ################################################################
    def wavfilevocoder_plot(self, sig, residual, decoded, wavfile=None):
        #########################################################
        # minpsd_plot = 1e-10 minimum power spectral density to prevent log(0)
        # cmap_plot = plt.cm.bone_r reversed gray scale colormap
        #########################################################

        Ns = self.enc.cfg.Ns()
        Fs = self.enc.cfg.Fs
        T = self.enc.T
        lpcorder = self.enc.cfg.M
        '''
        Env = enc[0]
        E = enc[1]
        memory = enc[3]
        config = enc[4]
        lpcorder = config["lpcorder"]
        print("lpcorder=%d Ns=%d Fs=%d"%(lpcorder,Ns,Fs))

        # LSF VQ code to LPC coefficients
        # enc = (L,E,T,memory,config) or (C,E,T,memory,config)
        """
        if lsfcodebook == None: L = enc[0]
        else: L = code2lsfmat(enc[0], lsfcodebook)
        """
        L, A = code2lpcmat(Env, lsfcodebook)
        '''
        L, A = self.dec.getLSFLPCMat_plot(self.enc)
        minpsd = self.enc.cfg.minpsd_plot
        cmap = self.enc.cfg.cmap_plot

        ################################################################
        # preparing plots
        fig, axes = plt.subplots(nrows=7)
        if wavfile != None: axes[0].set_title(wavfile)
        xticks = np.arange(T)/Fs    # x-axis tick values in seconds
        sigmax = np.maximum(np.max(abs(sig)),np.max(abs(decoded)))
        sigmax = np.minimum(2**15,1.2*sigmax)   # for good display range 
        axi = 0 # subplot index

        ################################################################
        # draw signal and residual
        axes[axi].plot(xticks,sig,label='input signal')
        axes[axi].plot(xticks,residual,linestyle='--',label='LP residual')
        #axes[axi].legend(['signal','residual'])
        axes[axi].legend()
        axes[axi].axis([ xticks[0], xticks[-1], -sigmax, sigmax ])
        axes[axi].set_xlabel('time (seconds)')
        axes[axi].set_ylabel('value')
        #plt.xlim(xticks[0],xticks[-1])  # not working, don't know why
        #plt.ylim(-2**15,2**15)  # not working, don't know why

        axi+=1   # drawing done, move to the next

        ################################################################
        # draw spectrogram of input signal
        [pxx,freq,t,cax] = axes[axi].specgram(sig,Fs=Fs,
                window=np.hamming(Ns*2),
                NFFT=Ns*2,noverlap=80,
                scale_by_freq=True,
                mode='psd',scale='dB',
                cmap=cmap)

        lab = 'input signal, PSD %.1f+/-%.1f'%(pxx.mean(),pxx.std())
        print(lab)
        axes[axi].text(T/Fs*0.05,Fs/8,lab)
        axes[axi].set_xlabel('time (seconds)')
        axes[axi].set_ylabel('frequency (Hz)')
        #axes[axi].set_label(lab)
        #axes[axi].legend(lab)
        axi+=1   # drawing done, move to the next

        ################################################################
        # draw spectrograms of the LP coefficients
        # 1) FFT on LP coef
        #Af = np.fft.fft(A,n=Ns*2,axis=0)
        #Af = Af[0:(Ns+1),:]     # take 0-pi only
        # Note: the following two lines are equivalent to 
        # Af = np.fft.rfft(A,n=Ns*2,axis=1) , see the reference manual 
        Af = np.fft.fft(A,n=Ns*2,axis=1)
        Af = Af[:,0:(Ns+1)]     # take 0-pi only

        """
        # power spectral densities (PSD) of 1/|A|
        #inv_aAf = 1.0/np.sqrt(Af*np.conjugate(Af))    
        inv_aAf = 1.0/np.sqrt(np.real(Af*np.conjugate(Af)))
        print(inv_aAf)
        print(type(inv_aAf))
        print(inv_aAf.shape)

        # log with minimum to avoid 1/0
        #log_inv_aAf = np.real(np.log(np.maximum(minpsd,inv_aAf)))
        log_inv_aAf = np.log(np.maximum(minpsd,np.real(inv_aAf)))
        """
        """
        # log power spectral densities (PSD) of the envelope, log(1/|A|^2)
        #inv_aAf = 1.0/np.sqrt(Af*np.conjugate(Af))    
        log_inv_aAf = -np.log(np.maximum(minpsd,np.real(Af*np.conjugate(Af))))
        print(log_inv_aAf)
        """
        # power spectral densities (PSD) of 1/|A|
        inv_aAf = 1.0/np.maximum(minpsd,np.real(Af*np.conjugate(Af)))

        # log with minimum to avoid 1/0
        #log_inv_aAf = np.real(np.log(np.maximum(minpsd,inv_aAf)))
        log_inv_aAf = np.log(np.real(inv_aAf))

        # spectrogram xmin,xmax,ymin,ymax = 0,T,0,Fs/2
        specgram_axis = [0,float(T)/float(Fs),0,float(Fs)/2]

        # summary of A(f)
        lab = '1/A(f), PSD %.1f+/-%.1f'%(inv_aAf[:].mean(),inv_aAf[:].std())
        print(lab)

        axes[axi].imshow(np.transpose(log_inv_aAf),
                cmap=cmap,origin='lower',aspect='auto',
                extent=specgram_axis)
        #axes[axi].text(Af.shape[0]*0.05,Af.shape[1]/8,lab)
        axes[axi].text(specgram_axis[1]*0.05,specgram_axis[3]/8,lab)
        #fig.colorbar(cax).set_label('Intensity [dB]')
        axi+=1   # drawing done, move to the next

        # draw LSFs on the spectrum
        # LSFs are in [0,pi], rescale it to [0,Ns]
        axes[axi].imshow(np.transpose(log_inv_aAf),
                cmap=cmap,origin='lower',aspect='auto',
                extent=specgram_axis)
        Lscaled = L/np.pi*float(Fs)/2    # scale to 0-Fs/2 (max freq)
        print(L)
        print(Lscaled)
        print(Fs)
        #axes[axi].plot(np.arange(Lscaled.shape[1])*Ns/Fs,np.transpose(Lscaled))
        axes[axi].plot(np.arange(Lscaled.shape[0])*Ns/Fs,Lscaled)
        axi+=1   # drawing done, move to the next

        ################################################################
        # draw spectrogram of the residual signal
        [pxx,freq,t,cax] = axes[axi].specgram(residual,Fs=Fs,
                window=np.hamming(Ns*2),
                NFFT=Ns*2,noverlap=80,
                scale_by_freq=True,
                mode='psd',scale='dB',
                cmap=cmap)
        lab = 'residual signal, PSD %.1f+/-%.1f'%(pxx.mean(),pxx.std())
        print(lab)
        axes[axi].text(T/Fs*0.05,Fs/8,lab)
        axes[axi].set_xlabel('time (seconds)')
        axes[axi].set_ylabel('frequency (Hz)')
        #fig.colorbar(cax).set_label('Intensity [dB]')
        axi+=1   # drawing done, move to the next

        ################################################################
        ######## For decoding results

        #print(decoded)
        # compute SNR: signal = input, noise = input-output (prediction error)
        # snr = sqrt((signal)^2/noise^2
        # SNR in dB = 10*log10(s^2/e^2)
        # = 20*log10(abs(s)/abs(e))
        reconerr = sig - decoded  # reconstruction error
        print(sig.shape)
        print(decoded.shape)
        print(np.max(np.abs(decoded)))
        Ps = (sig**2).mean()
        Pn = (reconerr**2).mean()
        print('Energies of signal, noise = %.3f, %.3f'%(Ps,Pn))
        snr = 10*np.log10((sig**2).sum()/(reconerr**2).sum())
        print('M = %d SNR = %.3f dB'%(lpcorder,snr))

        ######## PLOT ########
        # draw decoded signal
        lab = 'decoded SNR %.2f dB'%snr
        axes[axi].plot(xticks,decoded[:(T)],label=lab)
        axes[axi].plot(xticks,reconerr[:(T)],label='reconstruction error')
        axes[axi].legend()
        axes[axi].axis([ xticks[0], xticks[-1], -sigmax, sigmax ])
        axes[axi].set_xlabel('time (seconds)')
        axes[axi].set_ylabel('value')
        axi+=1   # drawing done, move to the next

        [pxx,freq,t,cax] = axes[axi].specgram(decoded,Fs=Fs,
                window=np.hamming(Ns*2),
                NFFT=Ns*2,noverlap=80,
                scale_by_freq=True,
                mode='psd',scale='dB',
                cmap=cmap)
        lab = 'decoded signal, PSD %.1f+/-%.1f'%(pxx.mean(),pxx.std())
        print(lab)
        axes[axi].text(T/Fs*0.05,Fs/8,lab)
        axes[axi].set_xlabel('time (seconds)')
        axes[axi].set_ylabel('frequency (Hz)')
        ######## END OF PLOT ########

        plt.show()
    ################################################################



####################################################
def preemphasis(x,alp,xmem):
    xp = np.zeros(len(x),dtype='float')
    xp[0] = x[0] - alp*xmem
    xp[1:] = x[1:] - alp*x[:-1]
    xmem = x[-1]    # last sample, for the next frame
    return xp,xmem

####################################################
def deemphasis(x,alp,xdmem):
    xd = np.zeros(len(x),dtype='float')
    xd[0] = x[0] + alp*xdmem
    #xd[1:] = x[1:] + alp*xd[:-1]   # wrong IIR filter, previous output required
    for t in range(1,len(x)):
        xd[t] = x[t]+alp*xd[t-1]
    xdmem = xd[-1]    # last sample, for the next frame
    return xd,xdmem

####################################################
def lpcresidual_frame(x,a,xmem,xbuf=None):
    # x: Ns (number of samples per shift)
    # a: lpcorder+1 (lpcorder is the order of lpc)
    # xmem: lpcorder, x memory 
    # returns e: residual signal

    # to save memory re-allocation, re-use xbuf
    np.concatenate((xmem,x),axis=0,out=xbuf)
    # xbuf = np.concatenate((xmem,x),axis=0)
    # xbuf: [Ns+lpcorder,]
    # x: [Ns,]
    # a: [lpcorder+1,]

    # filtering
    e = np.zeros(len(x),dtype='float')
    for t in range(len(x)):
        #print('t=%d len(x)=%d len(xbuf)=%d'%(t,len(x),len(xbuf)))
        #print('t=%d '%(t), end='')
        #print(np.arange(t+len(a)-1,t-1,-1))
        e[t] = np.multiply(a,xbuf[np.arange(t+len(a)-1,t-1,-1)]).sum()

    xmem[:] = x[(len(x)-(len(a)-1)):]
    return e,xmem


####################################################
# LSF line spectral frequencies
####################################################

####################################################
# source: pyvocoder ################################
# https://github.com/shamidreza/pyvocoder

# for poly2lsf and lsf2poly
from scipy.signal import deconvolve, convolve
#from scipy.signal import lfilter, hamming, freqz
#from numpy.fft import fft, ifft, rfft, irfft

# alternatives: 
# spectrum.poly2lsf
# https://pyspectrum.readthedocs.io/en/latest/ref_others.html#module-spectrum.linear_prediction
# https://github.com/johnhw/lpc_vocoder/blob/master/LPC.ipynb

def poly2lsf(a):
    a = a / a[0]
    A = np.r_[a, 0.0]
    B = A[::-1]
    P = A - B
    Q = A + B

    P = deconvolve(P, np.array([1.0, -1.0]))[0]
    Q = deconvolve(Q, np.array([1.0, 1.0]))[0]

    roots_P = np.roots(P)
    roots_Q = np.roots(Q)

    angles_P = np.angle(roots_P[::2])
    angles_Q = np.angle(roots_Q[::2])
    angles_P[angles_P < 0.0] += np.pi
    angles_Q[angles_Q < 0.0] += np.pi
    lsf = np.sort(np.r_[angles_P, angles_Q])
    return lsf

# alternative: spectrum.lsf2poly
# https://pyspectrum.readthedocs.io/en/latest/ref_others.html#module-spectrum.linear_prediction
def lsf2poly(lsf):
    order = len(lsf)
    Q = lsf[::2]
    P = lsf[1::2]
    poles_P = np.r_[np.exp(1j*P),np.exp(-1j*P)]
    poles_Q = np.r_[np.exp(1j*Q),np.exp(-1j*Q)]

    P = np.poly(poles_P)
    Q = np.poly(poles_Q)

    P = convolve(P, np.array([1.0, -1.0]))
    Q = convolve(Q, np.array([1.0, 1.0]))

    a = 0.5*(P+Q)
    return a[:-1]
####################################################

####################################################
# main method
####################################################
def main():
    demo = VCodecDemo()

    if len(sys.argv) > 1:   # having extra arguments
        infile = sys.argv[1] 
        if (len(sys.argv)>2): cfgfile = sys.argv[2]
        else: cfgfile = 'def2.conf'
        if (len(sys.argv)>3): outfile = sys.argv[3]
        else: outfile = 'decoded.wav'
        print(infile + " " + cfgfile + " " + outfile)
        demo.wavfilevocoder(infile,cfgfile,outfile)
    else:
        print('usage: ' + sys.argv[0] + ' wavfile [cfgfile] [outfile]')

        #wavfilevocoder('s2.wav',lpcorder)
        #wavfilevocoder('s3.wav',lpcorder)
        #wavfilevocoder('s4.wav',lpcorder)
        #wavfilevocoder('s5.wav',lpcorder)

        #wavfilevocoder('speech/timit_merged_16k.wav',10)
        #wavfilevocoder('speech/timit_merged_16k.wav',14)
        #wavfilevocoder('speech/timit_wav_selected_20110419/mdac0/sx451.wav',10)


####################################################
# check if main
if __name__ == '__main__': main()

####################################################
# end of file
####################################################
