# NoC-aware AI Accelerator DSE

AI 가속기 아키텍처를 대상으로 **Design Space Exploration (DSE)**를 수행하여
성능을 개선한 프로젝트

---

## 개요

본 프로젝트는 AI accelerator에서 발생하는
**연산 및 통신 병목을 분석하고**,
core allocation과 placement를 최적화하는 것을 목표로 한다.

대상 workload:

* BERT 기반 연산

---

## 구조

### 🔹 시스템 구성

* Processing Element (PE) array
* On-chip Network (NoC)
* DRAM / SRAM 기반 메모리 구조
* Layer group 단위 실행 모델

--> 연산 + 통신 + 메모리를 포함한 시스템 레벨 구조

---

### 🔹 DSE 흐름

1. Hardware 분석

   * core / memory 구성 파악

2. Workload 분석

   * 각 compute node의

     * 데이터 이동량
     * MAC 연산량 계산

3. Core allocation

   * node 중요도 기반 core 수 할당
   * 제약:

     * 전체 core의 약 1/3 이하
     * 4의 배수 정렬

4. Placement

   * NoC 거리 기반 배치
   * 일부 노드에 random placement 적용

---

## 핵심 특징

### 1. Communication bottleneck 기반 설계

* matmul 연산이 compute가 아닌
  **memory / NoC communication에 의해 지배됨**

---

### 2. Adaptive resource allocation

* MAC + traffic 기반 score 계산
* 중요 노드에 더 많은 compute resource 할당

---

### 3. NoC-aware placement

* 통신 비용 최소화를 고려한 배치
* symmetric + locality 기반 전략

---

### 4. Iterative DSE

* 이전 iteration 결과를 활용하여
* 성능을 점진적으로 개선

---

## 결과

|     Model     | Throughput (TOPS) |
| ------------- | ----------------- |
| Layer group A | 30.70             |
| Layer group B | 38.09             |
| Layer group C | 41.20             |

---

## 참고

자세한 분석 및 실험 결과는 보고서를 참고

* TP_AS_submission.pdf
