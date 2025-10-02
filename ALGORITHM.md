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