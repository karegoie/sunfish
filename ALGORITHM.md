# Sunfish 알고리즘 요약

Sunfish는 DNA 서열로부터 유전자 영역을 예측하는 경량 HMM(은닉 마르코프 모델) 기반 주석기입니다. 핵심 아이디어는 DNA를 복소수 신호로 매핑한 뒤, Morlet 연속 웨이블릿 변환(CWT)으로 다중 스케일 특징을 추출하고, 이를 HMM의 가우시안 방출 모델에 입력하여 엑손/인트론/인터제닉 상태를 추정하는 것입니다. 예측 시에는 스플라이스 신호(GT-AG)에 대한 점수 보정을 도입해 정확도를 높입니다.

## 입력과 출력
- 학습(train): FASTA(게놈) + GFF3(정답 CDS) → `sunfish.model`
- 예측(predict): FASTA(타깃) + `sunfish.model` → GFF3(gene, mRNA, exon, CDS)
- GFF3 출력은 `##gff-version 3` 헤더와 함께, 엑손 블록을 묶어 gene/mRNA/ exon/CDS를 생성하며, 좌표는 1-기반 포함 범위로 기록됩니다.

## 특징 추출: Wavelet + k-mer
1. 복소수 임베딩
   - A → 1 + i, T → 1 − i, G → −1 + i, C → −1 − i
   - 알려지지 않은 염기는 0 + 0i 로 취급
2. Morlet 웨이블릿 생성
   - 스케일 s마다 길이 ≈ s의 커널을 생성하고 중앙 정렬
3. FFT 기반 컨볼루션
   - 신호와 웨이블릿을 제로패딩하여 FFT → 주파수영역 곱셈 → IFFT
   - 컨볼루션 지연은 wavelet_len/2 만큼 보정해 시점 정렬
4. k-mer 빈도 보강 (선택 사항)
   - `-k/--kmer`로 지정한 k > 0이면 각 위치에서 길이 k 윈도를 슬라이딩
   - 유효한 염기로 구성된 경우 해당 4ᵏ 슬롯에 원-핫(one-hot) 값을 1로 설정
   - 염기 중 N 등 미지정 문자가 포함되면 해당 위치의 k-mer 특징은 0으로 유지
5. 특징 벡터 구성
   - 각 웨이블릿 스케일의 결과를 실수/허수부로 분해(스케일당 2차원)
   - k-mer 원-핫 벡터를 뒤에 이어 붙여 최종 관측값을 구성
   - 최종 관측값 차원 D = 2 × (웨이블릿 스케일 수) + 4ᵏ (k=0이면 0)

기본 스케일은 3의 거듭제곱(3, 9, 27, …)이며, `-w/--wavelet`으로 다음 형태를 지원합니다.
- 쉼표 목록: `a,b,c`
- 범위: `start:end:step` (증가/감소 모두 지원)
- 단일 값: `s`

## HMM 구조
- 상태 집합 S = {EXON_F0, EXON_F1, EXON_F2, INTRON, INTERGENIC} (총 5개)
  - EXON_Fk는 코돈 프레임(k=0,1,2)을 구분
- 초기확률 π, 전이확률 A(5×5)
- 방출분포: 대각 공분산 가우시안 N(μ_s, diag(σ²_s))
- 전역 정규화 통계: 각 특징 f의 전체 평균/표준편차(μ_g[f], σ_g[f])를 모델에 저장하여 예측 시 Z-score 정규화에 사용

## 학습 파이프라인(지도 + EM 보정)
1) 전처리와 데이터 증강
- 모든 FASTA 시퀀스의 정방향(+)과 역상보(−)를 모두 사용(2× 증강)

2) Pass 1: 전역 통계(정규화) 추정
- 전체(증강 포함) 관측값에 대해 특징별 평균/분산을 계산 → μ_g, σ_g
- 이후 모든 관측값을 Z-score: (x − μ_g)/σ_g 로 정규화

3) Pass 2: 지도(supervised) 추정
- GFF3의 CDS를 Parent 단위로 그룹화하여 엑손 목록 정렬
- 상태 라벨링 규칙
  - 정방향(+): 각 엑손의 시작을 phase로 맞추고 거리 오프셋에 따라 EXON_F(phase+offset mod 3)로 라벨링
  - 역상보(−): 좌표를 역상보 프레임으로 변환해 동일한 방식 적용
  - 연속 엑손 사이 구간은 INTRON, 나머지는 INTERGENIC
- 라벨에 기반해 초기/전이 카운트와 상태별 방출 평균/분산 누적 후 확률화
- 엑손 프레임 순환 제약 적용: F0→F1, F1→F2, F2→F0 전이질량을 강제하고 비정상 전이는 ε(1e−10)로 억제한 뒤 행 정규화

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
   - 학습 시 wavelet+k-mer 특징 계산을 시퀀스별(+/−) 병렬화
  - 예측 시 시퀀스·가닥 단위 병렬 Viterbi 수행
- 스레드 수는 미지정 시 온라인 CPU 수 자동 감지(`-t/--threads`로 지정 가능)
- 복잡도(대략)
   - CWT: O(K·N log N) (K: 스케일 수, N: 길이)
  - HMM Forward/Backward/Viterbi: O(N·|S|²), |S|=5
   - 메모리: 관측행렬 O(N·D), D=2K + 4ᵏ

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
- `train <train.fasta> <train.gff> [--wavelet|-w S1,S2,...|s:e:step] [--kmer|-k K] [--threads|-t N]`
- `predict <target.fasta> [--wavelet|-w S1,S2,...|s:e:step] [--kmer|-k K] [--threads|-t N]`
- 스케일 최대치는 `MAX_NUM_WAVELETS`(기본 100)로 제한되며, 특징 차원은 `2×스케일 수 + 4ᵏ` (k=0이면 k-mer 특징 없음)

## 파일 포맷: sunfish.model
- 헤더와 메타: `#HMM_MODEL_V1`, `#num_features`, `#wavelet_features`, `#kmer_features`, `#kmer_size`, `#num_states`
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
       - k-mer: one-hot 인코딩
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

#### 3) k-mer 원-핫 보강
- 구현: `build_observation_matrix` in `sunfish.c`
- 파라미터 k>0일 때, 각 위치 t에서 길이 k 윈도우를 읽어 인덱스 `index = ((...((b0<<2)|b1)<<2)|...|bk-1)`로 매핑(A=0,C=1,G=2,T=3; 기타는 무효)
- 행렬의 `(wavelet_rows + index, t)`에 1.0을 설정합니다. 범위를 벗어나거나 무효 염기가 포함되면 0 유지.

#### 4) 관측행렬 구성
- 내부 버퍼(features[rows][N])를 채운 뒤, 시간축 우선 표현으로 전치하여 `observations[N][D]`를 생성합니다.
- 차원 $D = 4\times K_\text{wavelet} + 4^k$ (k=0이면 k-mer 없음).

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
   - 헤더 메타: `#num_features`, `#wavelet_features`, `#kmer_features`, `#kmer_size`, `#num_states`
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
- 메모리: 관측 $N\times D$, $D=4K+4^k$; 학습 시 (+/−) 증강 포함

### 파라미터 일관성
- 예측은 학습시 저장된 wavelet 스케일/k-mer/청킹 설정을 강제 사용합니다. `predict` 명령은 `--threads`만 허용합니다.

### 경계 사례와 주의점
- 염기 ‘N’ 등 비표준 염기
   - CWT: 0+0i로 매핑 → 영향 최소화
   - k-mer: 윈도우 내 1개라도 비표준이면 해당 위치의 k-mer 특징 0 유지
- 너무 큰 k-mer는 특징 차원 상한(`MAX_NUM_FEATURES=8192`)을 넘어 실패 → 학습시 검증됨
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