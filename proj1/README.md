# DEEE725 Speech Signal Processing Lab
## 2023 Spring, Kyungpook National University

# Project 1 Isolated digit recognition in noisy environments

- attached: 
    2. [`proj1_nidr.ipynb`](proj1_nidr.ipynb)

- __목적__:

- Train/Validation/Test 환경으로 나누어 음성인식 실험해 보기 
- EPD 를 이용해서 음성구간 검출 및 학습/인식
- Noise suppression 을 이용해서 잡음을 제거하여 음성인식 성능 높인다.
- 모델: HMM (hmmlearn) 이용
- __Challenge__: RNN 시도해 본다.

1. Train and Validation
    1. `segmented-train.zip` 으로 HMM training
    2. `segmented-val.zip` 으로 HMM test 및 최적화 
    > 현재 성능이 매우 안 나온다. 문제점을 파악해서 올린다.
        - `org`: original recordings, segmented.
2. Test
    2. `unsegmented-test.zip` 으로 HMM test
        - `org`: original recordings, unsegmented.
    3. 파일이 발성단위로 나뉘어 있지 않기 때문에 EPD 를 이용하여 10개로 분리하여야 한다.

3. Noisy test
    2. `unsegmented-test.zip` 의 noisy mixture 로 HMM test
        - `nbnSNR10`, `nbnSNR0`, `nbnSNR-10`: narrowband noise 를 10, 0, -10 dB 로 섞은 것
        - `wbnSNR10`, `wbnSNR0`, `wbnSNR-10`: wideband noise 를 10, 0, -10 dB 로 섞은 것
    3. Wiener filtering 으로 잡음을 억제하고 제거한다.
    4. 모델은 clean training 으로 학습되어야 한다.
        - 이러한 환경을 unmatched condition 이라고 한다.
    2. `segmented-val.zip` 에도 noisy data 가 있다.
        - `nbnSNR10`, `nbnSNR0`, `nbnSNR-10`: narrowband noise 를 10, 0, -10 dB 로 섞은 것
        - `wbnSNR10`, `wbnSNR0`, `wbnSNR-10`: wideband noise 를 10, 0, -10 dB 로 섞은 것

4. HMM and RNN
    2. 모델은 기본적으로 `hmmlearn` 을 사용
    3. __challenge__: RNN 가능하면 해 본다.
    

4. Requirements
    - 반칙사항
        - validation data, test data 는 절대로 training 에는 사용하면 안 됨
        - EPD threshold 는 validation data 만을 사용
        - test data 는 unknown condition, 따라서 모든 학습과정, parameter tuning 이 종료된 후 한번만 test 해 봐야 함
    - 현재 성능이 매우 안 나옴
        - Data 를 눈과 귀로 보고 확인해 보고 문제점 공유
    - code 는 본인이 작성한 것을 쓰면 best 이지만 이 github 에 공개된 코드나 외부 코드를 써도 된다. 
        - 그럴 경우 반드시 명시. 명시되어 있지 않으면 본인이 작성한 것으로 가정되며 그럼 다른 코드와 유사하면 안 됨
    - toolkit 사용은 최소한으로.
        - GPU 없다고 가정
        - 최신 버전
    - 제출: A4 4쪽 이내 report 
