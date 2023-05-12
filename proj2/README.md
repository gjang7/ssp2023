# DEEE725 Speech Signal Processing Lab
## 2023 Spring, Kyungpook National University

# Project 2 Voiced, Unvoiced, and Silence detection on TIMIT dataset

- attached: 
    2. `timit_wav.tar`: see `Projec 1` at [`LMS1`](lms1.knu.ac.kr)

- __목적__:
    1. Voiced (유성음), Unvoiced (무성음), Silence (묵음) 을 구분하는 분류기를 만든다.
    2. 모든 모음은 유성음, 자음중 /s/, /f/, /th/, /ch/, /k/, /p/, /t/ 은 무성음

1. TIMIT transcripts explanation
    모든 wav file 들은 phoneme 단위로 시간정보가 기재되어 있음 

    > Phonetic symobol list: [phoneme classification, see tables 2 and 3](https://doi.org/10.3390/app11010428)

    __Example:__ `fmem0/sa1.wav` 에는 다음의 transcripts 가 포함됨

    1. `sa1.txt`: 
    > 0 65434 She had your dark suit in greasy wash water all year.
    2. `sa1.wrd`: 
    <pre><code>
2360 7021 she
7021 11462 had
11462 13640 your
13640 20720 dark
20720 28490 suit
28490 31450 in
31450 40128 greasy
41734 47306 wash
48476 53010 water
54324 58032 all
58032 63222 year
    </code></pre>
    3. `sa1.phn`: 
    <pre><code>
0 2360 h#
2360 5263 sh
5263 7021 iy
7021 8370 hv
8370 10234 eh
10234 11084 dcl
11084 11462 d
11462 11989 y
11989 12699 uh
12699 13640 r
13640 15590 dcl
15590 15746 d
    </code></pre>


2. Train/Validation/Test
    1. Train/Validation: `timit/train`의 화자들을 8:2, 혹은 다른 비율로 적당히 나눔
    `dr_`를 dialect region 을 나타냄. 따라서 각 dialect 안의 화자들을 나누는 것이 적합
    2. Test: `timit/test` 

3. 실험
    2. SA (dialect sentences) 로 학습-validation-test
    3. SX (phonetically compact) 로 학습-validation-test

3. 방법
    1. EPD or VAD 를 이용하여 1차 구분: 음성구간 앞뒤 확장 필요
    2. autocorrelation 을 이용하여 voiced/unvoiced rough 구분,  `a[1]/a[0] <> 0.5`, threshold 0.5 변경하면서 
    3. ZCR, median filter, heuristic 적용
    4. RNN 사용

4. 제출
    1. __Report__: 사용한 방법 상세히 설명, 20% test set 에 대한 성능
    2. __Code__: 실행 가능한 `.py` or `.ipynb`, argument는 TIMIT wav 위치.

4. Requirements
    - TIMIT 을 포함하여 data 제출 금지
    - Copyrights
        - TIMIT 은 LDC (linguistic data consortium) 의 소유권이 있으며, 경북대는 institutional license 를 
	구입하여 사용중 (관리 및 구매: 장길진)
	- 경북대 구성원은 모두 사용 가능하고, 문서에 관련 내용을 적을 수 있음
	- 외부에 유출되지 않도록 주의
    - 반칙사항
        - validation data, test data 는 절대로 training 에는 사용하면 안 됨
        - EPD threshold 는 validation data 만을 사용
        - test data 는 unknown condition, 따라서 모든 학습과정, parameter tuning 이 종료된 후 한번만 test 해 봐야 함
    - code 는 본인이 작성한 것을 쓰면 best 이지만 이 github 에 공개된 코드나 외부 코드를 써도 된다. 
        - 그럴 경우 반드시 명시. 명시되어 있지 않으면 본인이 작성한 것으로 가정되며 그럼 다른 코드와 유사하면 안 됨
    - toolkit 사용은 최소한으로.
        - GPU 없다고 가정
        - 최신 버전
    - 제출: A4 4쪽 이내 report 
