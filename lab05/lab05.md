# DEEE725 Speech Signal Processing Lab
## 2023 Spring, Kyungpook National University

# Lab 05 probabilistic noise suppression methods

- attached: [`lab05_ns_prob_models.ipynb`](lab04_ns_epd.ipynb)

1. probabilstic voice activity detection (VAD) for noise spectrum estimation
    - fixed thresholding 으로 noise 구간을 구하면 noise 의 크기에 크게 
    영향받을 수 밖에 없다. 따라서 noise 구간을 유연하게 추정할 수 있는
    (adaptive thresholding) 확률 모델을 사용하여 본다.

    1. 확률 모델을 이용하여 probabilistic voice activity detection 수행
    2. 각 frame 별로 noise 확률 계산 - find P(voice|y), y 는 한 frame 
    3. (deterministic decision) 확률값을 thresholding 하여 binary classification, 그리고 검출된 noise frame 들의 평균 제곱 Fourier 성분으로 noise spectrum 예측 
    4. (soft decision and maximum a posteriori estimation) 각 frame 별로 posterior probability ( P( voice | y ) ) 를 계산함. 그리고 noise spectrum 을 posterior probability로 weighted estimation 한다.  E[N] = sum (1-P(v|y)) y
    5. 두 가지 방법(deterministic/soft)으로 추정한 noise spectrum 으로 suppression 한 결과 비교 
    6. noise 차감은 lab04 의 time-domain Wiener filtering 이용

2. time domain VAD
    1. time domain signal 에 대해서 dual Gaussian mixture model 을 이용하여 probabilistic voice activity detection 
    2. speech 의 크기가 noise 의 크기보다 작다고 가정하고 
    작은 Gaussian 을 noise 분포로 가정 
    3. P(noise|y) = p(noise) p(y|noise), y 는 한 frame

3. frequency domain VAD
    1. Fourier transform on y(t) -> Y(w)
    2. Rayleigh distribution 으로 |X(w)|^2, |N(w)|^2 의 
    dual Rayleigh mixture model 추정
    3. 각 Rayleigh distribution 의 sigma parameter 로 E[N^2] 추정

4. log-frequency domain VAD
    1. Fourier transform on y(t) -> Y(w)
    2. Compute log PSD - log|Y(w)|^2
    3. Gaussian distribution 으로 log|X(w)|^2, log|N(w)|^2 의 
    mixture model 추정 (mean and variance)
    4. noise Gaussian 으로 E[ log|N(w)|^2 ]
    5. E[|N(w)|^2] = exp E[ log|N(w)|^2 ]

5. uniform filter bank energies 

6. mel-scale filter bank energies
