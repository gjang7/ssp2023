# DEEE725 Speech Signal Processing Lab
## 2023 Spring, Kyungpook National University

# Lab 02 Draw spectrogram
- attached: `lab02_spectrum.ipynb`

## assignments
- Complete 80% or more quantile minimum value spectrum drawing
- Upsampling (interpolation)
  - from 16 kHz audio file, resample it to 32 kHz, 48 kHz, and 44.1 kHz
  - use your own recording (_not `gjang-kdigits0-3.wav` given as a reference_)
  - need to implement low-pass FIR filter
  - check if there is no frequency contents over 8 kHz frequency by spectrogram
  - listen to the outputs
  - save them as `xxx_32k.wav`, `xxx_48k.wav`, `xxx_44k.wav`
- Decimation
  - for each of `xxx_32k.wav`, `xxx_48k.wav`, `xxx_44k.wav`
  - resample them to 8 kHz and 11.025 kHz
  - save them as `xxx_8k.wav` and `xxx_11k.wav`
