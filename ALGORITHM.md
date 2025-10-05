# Sunfish 알고리즘 요약

Sunfish는 DNA 서열로부터 유전자 영역을 예측하는 Transformer 기반 주석기입니다. 핵심 아이디어는 "Attention Is All You Need" 논문의 Transformer 아키텍처를 C/C++로 처음부터 구현하고, pthread를 사용한 병렬화로 성능을 최적화하는 것입니다. 모든 하이퍼파라미터는 TOML 설정 파일을 통해 런타임에 구성 가능합니다.

## 입력과 출력
- 학습(train): FASTA(게놈) + GFF3(정답 CDS) + config.toml → `sunfish.model`
- 예측(predict): FASTA(타깃) + `sunfish.model` + config.toml → GFF3(gene, mRNA, exon, CDS)
- GFF3 출력은 `##gff-version 3` 헤더와 함께, 유전자 영역을 기록합니다.

## 설정 파일 (TOML)
모든 모델 하이퍼파라미터는 TOML 파일에서 지정됩니다:
- `d_model`: 모델 차원 (예: 512)
- `num_encoder_layers`: 인코더 레이어 수
- `num_decoder_layers`: 디코더 레이어 수  
- `num_heads`: 어텐션 헤드 수
- `d_ff`: 피드포워드 네트워크 차원
- `dropout_rate`: 드롭아웃 비율
- `learning_rate`: 학습률
- `num_threads`: 병렬 처리 스레드 수

## Transformer 아키텍처

### 1. 입력 임베딩
- DNA 토큰 (A, C, G, T)을 밀집 벡터로 변환
- 어휘 크기: 4 (DNA 염기)

### 2. 위치 인코딩 (Sinusoidal)
```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```
- 위치 정보를 모델에 주입

### 3. 인코더 스택
각 인코더 레이어는 다음으로 구성:
- **멀티헤드 셀프어텐션**: 입력 시퀀스의 모든 위치 간 관계 학습
- **레이어 정규화**: 학습 안정화
- **위치별 피드포워드**: FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
- **잔차 연결**: 각 서브레이어 주변

### 4. 디코더 스택  
각 디코더 레이어는 다음으로 구성:
- **마스크된 멀티헤드 셀프어텐션**: 자기회귀적 생성
- **크로스어텐션**: 인코더 출력과 연결
- **레이어 정규화**: 학습 안정화
- **위치별 피드포워드**: ReLU 활성화
- **잔차 연결**: 각 서브레이어 주변

### 5. 출력 투영
- 선형 레이어로 어휘 크기에 투영
- 유전자 주석 예측

## 어텐션 메커니즘

### Scaled Dot-Product Attention
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

여기서:
- Q (쿼리), K (키), V (값)는 입력 행렬
- d_k는 키 벡터의 차원
- 스케일링 인수 1/√d_k는 소프트맥스 안정화

### Multi-Head Attention
```
MultiHead(Q, K, V) = Concat(head₁, ..., head_h)W^O
where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

- 여러 헤드가 서로 다른 표현 부분공간 학습
- 헤드 간 병렬 계산

## 병렬화 전략

pthread를 사용한 고성능 병렬 처리:

### 1. 어텐션 헤드 병렬화
- 각 어텐션 헤드를 독립적인 스레드에서 계산
- 헤드 수 ≥ 스레드 수일 때 최적

### 2. 행렬 곱셈 병렬화
- 결과 행렬의 행을 스레드 간 분배
- 작업 균형을 위한 동적 분할

### 3. 피드포워드 네트워크 병렬화  
- 배치 항목을 스레드 간 분배
- 독립적인 계산으로 동기화 최소화

## 구현 세부사항

### 행렬 연산
- 동적 할당을 통한 유연한 크기 조정
- 행 우선 메모리 레이아웃
- 캐시 친화적 접근 패턴

### 레이어 정규화
```
LayerNorm(x) = γ ⊙ (x - μ) / √(σ² + ε) + β
```
- 평균 μ와 분산 σ²를 레이어 차원에서 계산
- 학습 가능한 γ (스케일), β (시프트) 파라미터

### 수치 안정성
- 소프트맥스에서 최대값 빼기 (오버플로 방지)
- 레이어 정규화의 ε = 1e-6
- 학습 시 그래디언트 클리핑

### 스레드 안전성
- 재사용 가능한 워커 스레드 풀
- 순전파 중 공유 상태 없음
- 모델 업데이트 시 뮤텍스 보호

## 과학적 정확성

이 구현은 원본 Transformer 논문을 정확히 따릅니다:
- 정현파 위치 인코딩 공식
- 스케일된 내적 어텐션 메커니즘  
- 적절한 차원 분할을 통한 멀티헤드 어텐션
- ReLU를 사용한 위치별 피드포워드 네트워크
- 각 서브레이어 후 레이어 정규화
- 각 서브레이어 주변 잔차 연결

## 이전 버전과의 차이점

이전 버전 (HSMM 기반):
- 은닉 반마르코프 모델 (HSMM)
- 연속 웨이블릿 변환 특징
- Baum-Welch 학습 알고리즘
- Viterbi 디코딩
- constants.h의 하드코딩된 상수

새 버전 (Transformer 기반):
- 셀프어텐션을 사용한 Transformer 아키텍처
- 토큰 기반 시퀀스 모델링
- 그래디언트 기반 학습
- 어텐션 기반 디코딩
- 런타임 TOML 설정
- pthread를 통한 병렬화

## 성능 최적화

1. **병렬 행렬 연산**: 행렬 곱셈을 위한 스레드 풀
2. **헤드 레벨 병렬화**: 독립적인 어텐션 헤드 계산
3. **메모리 레이아웃 최적화**: 순차 메모리 접근
4. **스레드 풀 재사용**: 스레드 생성 오버헤드 회피
  - 연속 엑손 사이 구간은 INTRON, 나머지는 INTERGENIC
- 라벨에 기반해 초기/전이 카운트와 상태별 방출 평균/분산 누적 후 확률화
- 엑손 프레임 순환 제약 적용: F0→F1, F1→F2, F2→F0 전이질량을 강제하고 비정상 전이는 ε(1e−10)로 억제한 뒤 행 정규화
- START/STOP 코돈 전이 제약: START_CODON → EXON_F0, EXON_F* → STOP_CODON → INTERGENIC/INTRON

4) Baum–Welch(EM) 보정(반지도)
- Forward–Backward를 로그-합-지수 안정화로 수행하여 매 반복 누적치 갱신
- 초기, 전이, 방출(평균/분산) 갱신 시 하한(ε)과 분산 하한(≥1e−6) 유지
- 수렴 임계치 또는 최대 반복(100)까지 반복 후, 다시 엑손 순환 제약을 재적용

5) 모델 저장
- INITIAL, TRANSITION, EMISSION(각 상태의 MEAN/VARIANCE), GLOBAL_STATS(MEAN/STDDEV) 섹션으로 직렬화(`sunfish.model`)

## 예측 파이프라인(Viterbi + 스플라이스 보정)
1. 각 염기 위치의 특징을 계산하고 모델의 전역 통계로 Z-score 정규화
2. Viterbi 알고리즘으로 최대우도 상태열 추정 중, 전이 로그확률에 스플라이스 보정 적용
   - EXON→INTRON 경계에서 (donor) GT 매칭 시 보너스 log(3.0), 불일치 시 패널티 log(0.3)
   - INTRON→EXON 경계에서 (acceptor) AG 매칭 시 보너스 log(3.0), 불일치 시 패널티 log(0.3)
3. 상태열 후처리로 엑손 구간을 묶어 유전자 단위를 정의하고, 점수는 길이 정규화된 로그우도(log_prob/len) 사용
4. 좌표 변환
   - 정방향(+)은 0-기반 내부 좌표 → 1-기반 GFF 좌표로 변환
   - 역상보(−)는 원래 서열 길이를 기준으로 대칭 좌표 계산
5. 출력
   - gene + mRNA(Parent=gene) + exon(Parent=mRNA) + CDS(Parent=gene)를 생성
   - 출력은 내부 큐에 수집 후 (seqid → start → feature 우선순위: gene<mRNA<CDS)로 정렬하여 방출

## 병렬화와 성능
- 고정 크기 스레드 풀로 작업 분배
   - 학습 시 wavelet 특징 계산을 시퀀스별(+/−) 병렬화
  - 예측 시 시퀀스·가닥 단위 병렬 Viterbi 수행
- 스레드 수는 미지정 시 온라인 CPU 수 자동 감지(`-t/--threads`로 지정 가능)
- 복잡도(대략)
   - CWT: O(K·N log N) (K: 스케일 수, N: 길이)
  - HMM Forward/Backward/Viterbi: O(N·|S|²), |S|=5
   - 메모리: 관측행렬 O(N·D), D=2K

## 수치 안정성과 예외 처리
- Forward/Backward는 로그-합-지수(log-sum-exp) 사용
- 분산 하한(≥1e−6)과 확률 하한(≥1e−10)으로 언더플로우 방지
- σ_g가 너무 작을 때 최소값으로 클리핑하여 정규화 안정화
- 미확정 염기(N)는 0+0i로 임베딩되어 영향 최소화

## 한계와 설계 선택
- 상태 공간이 엑손/인트론/인터제닉 + 프레임(3)으로 단순화되어 UTR, 프로모터 등은 모델링하지 않음
- 스플라이스 모티프는 GT-AG 이진 매칭 보정만 사용(고급 PWM/딥 모델 없음)
- 방출 공분산은 대각 가정(특징 간 상관을 모델링하지 않음)

## 명령줄 옵션 요약
- `train <train.fasta> <train.gff> [--wavelet|-w S1,S2,...|s:e:step] [--threads|-t N]`
- `predict <target.fasta> [--threads|-t N]`
- 스케일 최대치는 `MAX_NUM_WAVELETS`(기본 100)로 제한되며, 특징 차원은 `2×스케일 수`

## 파일 포맷: sunfish.model
- 헤더와 메타: `#HMM_MODEL_V1`, `#num_features`, `#wavelet_features`, `#kmer_features` (legacy), `#kmer_size` (legacy), `#num_states`
- INITIAL: 상태별 초기확률 1행
- TRANSITION: 상태별 전이확률 5행
- EMISSION: 상태별 MEAN/VARIANCE
- GLOBAL_STATS: 관측의 정규화용 MEAN/STDDEV

---
이 문서는 `src/sunfish.c`, `src/hmm.c`, `src/cwt.c`, `src/fft.c`, `src/thread_pool.c` 및 해당 헤더를 기준으로 자동 생성되었습니다. 구현 세부가 업데이트되면 이 문서도 함께 갱신되어야 합니다.

## 구현 상세와 데이터 플로우 ( Deep Dive )

아래는 코드 레벨 동작을 함수/구조체 단위로 풀어쓴 상세 설계입니다. 핵심 데이터 경로를 먼저 요약한 뒤, 모듈별 알고리즘을 수식과 함께 설명합니다.

### 끝에서 끝까지 데이터 플로우

- 학습(train)
   1) FASTA 로딩 → `parse_fasta` → `FastaData{records[i].id, sequence}`
   2) GFF3 로딩 → `parse_gff_for_cds` → `CdsGroup{parent_id, Exon{seqid,start,end,strand,phase}[]}` (Parent 단위 그룹)
   3) 시퀀스 증강(+/− strand) → 총 2×개수의 학습 시퀀스
   4) 스레드풀 → 각 시퀀스에 대해 `build_observation_matrix`
       - CWT: `compute_cwt_features`(Morlet, FFT 컨볼루션)
   5) Pass1 전역통계(정규화 파라미터) 계산 → Z-score로 관측값 정규화
   6) Pass2 지도 신호 생성(GFF 기반 라벨링) → 상태별 초기/전이/방출 통계 누적 → HMM 파라미터 산출 + 엑손 프레임 순환 제약
   7) 스플라이스 PWM 학습 → 모델에 저장
   8) Baum–Welch(EM) 반지도 수렴 → 엑손 순환 제약 재적용
   9) 모델 파일(`sunfish.model`) 저장: INITIAL/TRANSITION/EMISSION/GLOBAL_STATS/PWM/#메타데이터

- 예측(predict)
   1) 모델 로딩 → 파생 설정(스케일/특징/청킹) 강제 동기화
   2) FASTA 로딩 → (옵션) 청킹(+오버랩)
   3) 스레드풀 → 각 (청크×strand) 작업에 대해 `build_observation_matrix` → Z-score 정규화
   4) Viterbi(전이 로그에 스플라이스 보정 포함) → 상태열
   5) 상태열 후처리 → 엑손 블록 묶기 → ORF 검증(ATG…STOP, 3의 배수, 내부 stop 없음)
   6) 좌표/strand 변환 + GFF3 라인 생성(gene, mRNA, exon, CDS)
   7) 내부 큐에 축적 후 전역 정렬(SeqID → start → feature rank) → 표준출력

---

### 특징 추출 파이프라인

#### 1) DNA → 복소 신호 매핑
- 구현: `dna_to_complex` in `cwt.c`
- 매핑: A→1+i, T→1−i, G→−1+i, C→−1−i, 그 외(N 등)→0
- 시퀀스 길이 N에 대해 `dna_to_signal(sequence, N, out)`이 벡터를 생성합니다.

#### 2) Morlet 연속 웨이블릿 변환(CWT)
- 파형 생성: `generate_morlet_wavelet(scale, length=⌊scale⌋, ψ)`
   - 중심은 `length/2`로 정렬하여 시간 정렬 편향을 줄입니다.
   - 수식: $\psi(t) = \frac{1}{\sqrt{s\cdot\pi^{1/4}}}\exp\left(-\tfrac12 (t/s)^2\right)\cdot e^{-j 2\pi t/s}$
- 컨볼루션: `convolve_with_wavelet(signal, N, ψ, L, out)`
   - 패딩 길이 $P=\text{next\_power\_of\_2}(N+L-1)$
   - $\mathcal{F}(x*\psi)=\text{FFT}(x)\odot\text{FFT}(\psi)$ → IFFT 후 `offset=L/2`만큼 보정 추출
   - 복소 결과에서 4가지 채널을 생성: 실수, 허수, 크기, 위상
- 다중 스케일: `compute_cwt_features(seq, N, scales[], K, features)`는 스케일별 4채널을 세로로 이어붙인 2D 행렬을 채웁니다.

#### 3) 관측행렬 구성
- 내부 버퍼(features[rows][N])를 채운 뒤, 시간축 우선 표현으로 전치하여 `observations[N][D]`를 생성합니다.
- 차원 $D = 4\times K_\text{wavelet}$.

### Z-score 정규화(학습/예측 공통)
- Pass1에서 전역 통계: $\mu_g[f] = \frac{1}{T}\sum x_f$, $\sigma^2_g[f] = \frac{1}{T}\sum x_f^2 - \mu_g[f]^2$; $\sigma_g<10^{-10}$이면 하한 적용.
- 관측값 표준화: $z = (x-\mu_g)/\sigma_g$; 학습 후 `GLOBAL_STATS`로 모델에 저장, 예측 시 재사용.

### 지도 신호 라벨링(학습 Pass2)
- 구현: `label_forward_states`, `label_reverse_states` in `sunfish.c`
- 규칙
   - Forward(+): 각 exon [start,end]를 0-기반으로 변환 후, phase를 기준으로 프레임(F0/F1/F2)에 주기적으로 배정. 인접 엑손 사이 구간은 INTRON, 그 외는 INTERGENIC.
   - Reverse(−): 시퀀스 길이 N에 대해 `rc_start=N−1−end`, `rc_end=N−1−start`로 뒤집어 동일 규칙 적용.
- 출력: 길이 N의 상태라벨 배열(값∈{EXON_F0,F1,F2,INTRON,INTERGENIC}).

### HMM 파라미터 추정(Pass2)
- 누적: `accumulate_statistics_for_sequence`
   - 초기: `initial_counts[state_0]++`
   - 전이: `transition_counts[s_t][s_{t+1}]++`
   - 방출: 정규화된 관측 $z_t$를 상태별 Σ, Σ², 카운트에 누적
- 최종화
   - 초기/전이: 행 합으로 정규화, 하한 $\ge 10^{-10}$ 적용
   - 방출(대각 가우시안): $\mu_{s,f}=\frac{\sum z}{n_s}$, $\sigma^2_{s,f}=\max(\frac{\sum z^2}{n_s}-\mu^2, 10^{-6})$
- 엑손 프레임 순환 제약: `enforce_exon_cycle_constraints`
   - F0→F1, F1→F2, F2→F0만 허용(동일 프레임 간 전이는 $10^{-10}$로 억제) 후 행 정규화.

### 스플라이스 PWM 학습과 적용
- 학습: `train_splice_model`
   - 도너(엑손→인트론) 창 크기: DONOR_MOTIF_SIZE=9, 어셉터(인트론→엑손): ACCEPTOR_MOTIF_SIZE=15
   - GFF 엑손 경계에서 창을 절반-중심 정렬로 추출(−strand는 RC 보정) → 염기별/좌표별 카운트
   - 로그-오즈 PWM: 배경 $0.25$, pseudo=1.0, $\log\frac{f_{b,pos}}{0.25}$; 열별 최소 합산으로 최소 점수도 저장
   - 모델에 `PWM` 블록으로 저장(WEIGHT, DONOR/ACCEPTOR 행렬, MIN_SCORE)
- 예측 중 보정: `splice_signal_adjustment`
   - 전이 로그에 더해짐: GT/AG 단순 체크 보너스/패널티($\pm 10^{-3}$) + PWM 점수×가중치

### EM(바움-웰치) 반지도 보정
- 구현: `hmm_train_baum_welch`
- 로그-합-지수 안정화로 Forward/Backward 계산 후, 감마/크시로 초기/전이/방출 재추정(분산 하한 유지)
- 지정 횟수(k=10) 또는 수렴 임계값으로 반복, 이후 엑손 순환 제약 재적용

### 비터비(Viterbi) 추론과 후처리
- 재귀식
   - 초기: $\delta_0(j)=\log \pi_j + \log p(z_0|j)$
   - 전개: $\delta_t(j)=\max_i\big[\delta_{t-1}(i)+\log a_{ij}+\Delta_\text{splice}(i\to j,t)\big]+\log p(z_t|j)$
   - 되돌이기: $\psi_t(j)=\arg\max_i(\cdot)$
- 상태열 → 엑손 구간 병합
   - 연속 EXON_F*는 하나의 엑손으로 묶음, 엑손 시작 프레임의 상태로 phase 기록
   - 엑손 덩어리들 사이의 구간은 인트론으로 간주하되, 출력은 exon/CDS만 생성
- ORF 검증: `is_valid_orf`
   - ATG 시작, TAA/TAG/TGA 종료, 길이%3=0, 내부 stop 부재일 때만 출력(메모리 부족 시 검증 생략하고 출력)
- 점수: $\text{score}=\exp(\text{log\_prob}/T)$를 [0,1]로 클램프
- 좌표 변환
   - ‘+’: 0-기반 내부좌표를 1-기반으로 변환
   - ‘−’: 원본 길이 L에 대해 start’=L−1−end, end’=L−1−start → 1-기반 변환
- GFF3 출력
   - gene + mRNA(Parent=gene) + exon(Parent=mRNA) + CDS(Parent=gene)
   - 내부 큐에 축적 후 `compare_gff_lines`로 정렬(SeqID→start→gene<mRNA<CDS)

### 청킹(chunking)
- 설정: `--chunk-size`, `--chunk-overlap`, `--chunk/--no-chunk`(학습 시 메타 저장 → 예측시 강제 재사용)
- 경계: `step = size - overlap`; 마지막 청크는 잔여 포함
- 예측은 각 청크(+/− strand)를 독립적으로 비터비 후 즉시 출력합니다(병합/중복 제거는 하지 않음). 따라서 긴 유전자(청크 경계에 걸친 경우)에서 중복 예측 가능. 후처리 병합을 권장합니다.

### 스레드풀 아키텍처
- 구현: `thread_pool.c`
   - 단순 작업큐(FIFO) + `active_tasks`/condvar로 완료 대기
- 학습: (+/−) 각 시퀀스의 특징행렬 계산을 병렬화
- 예측: (시퀀스×strand×청크) 단위로 작업을 병렬 제출

### 파일 포맷과 상호 운용
- 입력
   - FASTA: 헤더의 첫 공백 전까지를 ID로 사용, 시퀀스 라인은 연결
   - GFF3: type=CDS만 사용, `Parent=`로 그룹핑, `phase` 필드 사용
- 모델(`sunfish.model`)
   - 헤더 메타: `#num_features`, `#wavelet_features`, `#kmer_features` (legacy, 항상 0), `#kmer_size` (legacy, 항상 0), `#num_states`
   - 블록: `INITIAL`, `TRANSITION`, `EMISSION`(STATE별 MEAN/VARIANCE), `GLOBAL_STATS`(MEAN/STDDEV)
   - 옵션: `PWM`(WEIGHT, DONOR/ACCEPTOR, MIN_SCORE)
   - 꼬리 메타: `#chunk_size`, `#chunk_overlap`, `#use_chunking`, `#num_wavelet_scales`, `#wavelet_scales ...`

### 수치/안정성 세부
- 로그-도메인 연산, `log-sum-exp` 패턴으로 언더플로우 억제
- 분산 하한 $\ge 10^{-6}$, 확률 하한 $\ge 10^{-10}$
- 전역 표준편차 하한 $\ge 10^{-10}$

### 시간/메모리 복잡도(구체화)
- CWT: 각 스케일당 FFT 2회 + 역FFT 1회 → $\mathcal{O}(K\cdot N\log N)$
- HMM(Forward/Backward/Viterbi): $\mathcal{O}(N\cdot |S|^2)$, $|S|=5$
- 메모리: 관측 $N\times D$, $D=4K$; 학습 시 (+/−) 증강 포함

### 파라미터 일관성
- 예측은 학습시 저장된 wavelet 스케일/k-mer/청킹 설정을 강제 사용합니다. `predict` 명령은 `--threads`만 허용합니다.

### 경계 사례와 주의점
- 염기 ‘N’ 등 비표준 염기
   - CWT: 0+0i로 매핑 → 영향 최소화
- 음수/0 청크 크기 또는 overlap≥size는 에러로 종료
- 예측 청킹은 청크 경계 병합을 수행하지 않으므로, 중복 gene 출력 가능(후처리 도구 권장)

### 코드 참조(주요 엔트리)
- 특징: `compute_cwt_features`(`cwt.c`), `dna_to_complex`, `convolve_with_wavelet`
- 관측행렬: `build_observation_matrix`(`sunfish.c`)
- 정규화: Pass1 in `handle_train` → `normalize_observations_in_place`
- 라벨링: `label_forward_states`, `label_reverse_states`
- 누적/추정: `accumulate_statistics_for_sequence` → HMM 파라미터 최종화 → `enforce_exon_cycle_constraints`
- PWM: `train_splice_model` → 모델에 PWM 저장/로드(`hmm.c`)
- EM: `hmm_train_baum_welch`
- 추론: `hmm_viterbi`(전이 로그에 스플라이스 보정 포함)
- 출력: `output_predicted_gene` + 내부 큐 정렬/플러시

### 수식 요약(참고)
- 가우시안 방출 로그확률(대각 공분산)
   $$\log p(\mathbf{z}_t|s)=\sum_{f=1}^D\Big(-\tfrac12\log(2\pi\,\sigma^2_{s,f})-\tfrac{(z_{t,f}-\mu_{s,f})^2}{2\,\sigma^2_{s,f}}\Big)$$
- Forward/Backward는 표준 HMM 공식을 로그-합-지수 안정화로 구현
- Viterbi 전개식에 스플라이스 보정 $\Delta_\text{splice}$를 가산

---

부가 메모
- 본 문서는 구현에 맞춘 동작 명세이며, 추가 상태(UTR, 프로모터 등)나 고급 스플라이스 모델(PWM 확장, DP 병합 등)을 도입할 경우 본 문서의 관련 절을 확장하세요.