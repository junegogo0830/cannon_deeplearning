# cannon_deeplearning

이미지 기반 공장 품질검사(PASS/FAIL) 딥러닝 과제 프로젝트.

> 현재 상태: **스캐폴딩 단계** — 폴더 구조, 설정, 함수 시그니처/docstring 만 있고 모델·판정 로직은 미구현(TODO)입니다.

## 1. 문제 정의

- **입력**: 기종별 제품 이미지
- **출력**: 15개 검사 스텝 각각에 대한 PASS / FAIL (+ 제품 단위 최종 판정)
- **판정 방식**: 스텝별 이상 스코어를 계산하고 **임계값(threshold)** 과 비교하여 PASS/FAIL 결정

### 제약 조건

| 항목 | 내용 |
|---|---|
| 클래스 불균형 | FAIL 데이터가 극히 적음 → 일반 분류(supervised) 학습이 어려움 |
| 연산 환경 | GPU 사용 불가, **CPU 전용** → 경량 접근 위주 |
| 판정 로직 | YOLO / OCR / 템플릿 매칭 등 기성 알고리즘의 결과를 **그대로 판정에 쓰지 않음**. 스코어링·threshold 는 직접 구성 |

## 2. 접근 방향: 이상탐지 (Anomaly Detection)

정상(PASS) 이미지만으로 "정상이 어떻게 생겼는지"를 학습하고, 정상에서 벗어난 정도를 **이상 스코어**로 수치화합니다.

```
이미지 ─► 전처리(ROI) ─► 특징 추출 ─► 이상탐지 모델(정상만 학습) ─► 이상 스코어
                                                                        │
                                       스텝별 threshold (직접 산출) ◄───┤
                                                                        ▼
                                               스텝 PASS/FAIL ─► 제품 최종 판정
```

- **학습**: PASS 이미지만 사용. FAIL 은 검증·평가·threshold 보정에만 사용
- **후보 모델** (CPU 경량 순): 가우시안/마할라노비스 → 패치 메모리뱅크 k-NN → 소형 Conv AutoEncoder
- **특징**: 수작업 특징(색/텍스처/엣지) 또는 경량 CNN 임베딩 (특징 추출기로만 사용)
- **Threshold**: 정상 검증 스코어 분포 기반(percentile, mean+k·std) + 소수 FAIL 로 보정
- **평가**: accuracy 대신 AUROC / AUPR / Precision·Recall·F1 (불균형 고려)

## 3. 폴더 구조

```
cannon(2)/
├── configs/            # 실험 설정 (config.yaml: 기종/스텝/ROI/threshold/경로)
├── data/
│   ├── raw/            # 원본 이미지 + labels.csv (git 제외)
│   └── processed/      # 전처리 결과 (git 제외)
├── src/
│   ├── data/           # 이미지·라벨 로딩, 전처리, 분할
│   ├── features/       # 수작업/패치/CNN 임베딩 특징 추출
│   ├── models/         # 이상탐지 모델 (gaussian, patch_knn, autoencoder)
│   ├── scoring/        # 스코어 집계·정규화, threshold 산출, PASS/FAIL 판정
│   ├── eval/           # 지표, 시각화, 리포트
│   └── utils/          # config / seed / io / logger
├── scripts/            # 실행 엔트리포인트 (train / evaluate / infer)
├── notebooks/          # 탐색·EDA 노트북
├── reports/            # 분석 리포트, 그래프
├── results/            # 학습 산출물 (모델, threshold, 스코어 CSV) (git 제외)
├── requirements.txt
└── README.md
```

## 4. 데이터 형식 (가정)

`data/raw/labels.csv`

| image_id | machine_type | step | label |
|---|---|---|---|
| img_0001 | MODEL_A | 1 | PASS |
| img_0001 | MODEL_A | 2 | FAIL |

> 실제 데이터 형식이 다르면 `src/data/dataset.py` 와 config 의 `paths` 를 조정합니다.

## 5. 실행법 (뼈대)

```bash
# 1) 환경 설정 (Python 3.10+ 권장)
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt   # torch 는 CPU 빌드로 설치됨

# 2) 데이터 배치
#    data/raw/ 에 기종별 이미지와 labels.csv 를 넣는다

# 3) 설정 확인
#    configs/config.yaml 의 기종/스텝/ROI/threshold 수정

# 4) 학습 (정상 데이터로 스텝별 모델 + threshold 산출)
python -m scripts.train --config configs/config.yaml --machine MODEL_A

# 5) 평가
python -m scripts.evaluate --config configs/config.yaml --machine MODEL_A --run <run_name>

# 6) 추론
python -m scripts.infer --config configs/config.yaml --machine MODEL_A --run <run_name> --image path/to/img.png
```

## 6. TODO / 로드맵

- [ ] 실제 데이터 확인 후 config 의 ROI·스텝 정의 채우기
- [ ] 데이터 로더·전처리 구현 (`src/data`)
- [ ] 베이스라인: 수작업 특징 + 가우시안 모델
- [ ] 스코어 정규화 및 threshold 산출 구현 (`src/scoring`)
- [ ] 평가 지표·시각화 구현 (`src/eval`)
- [ ] 패치 k-NN / AutoEncoder 로 확장 및 비교
- [ ] 결과 리포트 작성 (`reports/`)
