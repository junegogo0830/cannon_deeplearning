# cannon_deeplearning

이미지 기반 공장 품질검사(PASS/FAIL) 딥러닝 과제 프로젝트.

> 현재 상태: **EDA 완료, 모델·판정 로직은 미구현(TODO)**. 실 데이터 기반 데이터 특성 분석 결과는 [reports/eda_summary.md](reports/eda_summary.md)와 [notebooks/01_eda.ipynb](notebooks/01_eda.ipynb) 참고.

## 1. 문제 정의

- **입력**: 기종별 제품 이미지 (13개 기종, 핸디형 카메라로 촬영)
- **출력**: 기종별 검사 스텝(13~20개, 기종마다 다름) 각각에 대한 PASS / FAIL (+ 제품 단위 최종 판정)
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
- **정렬(registration)**: 핸디형 카메라 특성상 제품마다 이동·회전·배율 편차가 큼(EDA 실측: 이동 최대 ~120px, 회전 최대 ~5°, 배율 ±10%) → affine(4DOF) 정렬 + 이중 게이트(inlier 조건 + 변환값 타당성 bound) 필수. 자세한 근거는 [reports/eda_summary.md](reports/eda_summary.md) §5
- **모델** (메인): 패치 메모리뱅크 k-NN (PatchCore류, 직접 구현) — 비교용 베이스라인으로 가우시안/마할라노비스, 데이터 많은 기종 한정 실험으로 소형 Conv AutoEncoder
- **특징**: 수작업 특징(엣지 방향·그래디언트, 채도가중 Hue 등 조명/반사에 강건한 것 위주) 또는 경량 CNN 임베딩 (특징 추출기로만 사용)
- **Threshold**: 정상 검증 스코어 분포 기반(percentile, mean+k·std) + 소수 FAIL 로 보정 (FAIL 라벨이 있는 기종은 3개뿐, 총 9개 제품 — §3 참고)
- **평가**: accuracy 대신 AUROC / AUPR / Precision·Recall·F1 (불균형 고려). FAIL 라벨 없는 10개 기종은 합성 결함 주입으로 보완 (정량 검증 불가능함을 명시)

## 3. 폴더 구조

```
cannon(2)/
├── configs/            # 실험 설정 (config.yaml: 기종/스텝/정렬/threshold/경로)
├── data/
│   ├── raw/            # 원본 이미지, data/raw/<기종>/<제품>/step_N.jpg (git 제외, 용량 큼)
│   └── processed/
│       └── eda/        # EDA 결과 CSV (inventory, ng_events, alignment 등 — 용량 작아 git 포함)
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

## 4. 데이터 형식 (실측, EDA 완료)

```
data/raw/<기종코드>/<제품 시리얼>/step_N.jpg        # N = 0부터 시작, 기종마다 스텝 수 다름(13/15/18/20)
data/raw/<기종코드>/<제품 시리얼>/NG/<timestamp>_cell{c}_process{p}_step_{N}.jpg   # 불량이 확인된 스텝 (유일한 FAIL 라벨 소스)
```

- 라벨 CSV는 따로 없음. `NG/` 폴더의 존재·파일명이 FAIL 라벨의 전부이며, `scripts/build_inventory.py`가 이를 스캔해 `data/processed/eda/inventory.csv`(전체 목록)와 `ng_events.csv`(FAIL 목록)를 생성함
- 기종코드(예 `3029C003AA`)는 마케팅 모델명이 아니라 생산 주문(대오더) 코드. 제품 시리얼 앞 3글자가 기종코드와 1:1 대응
- 해상도는 전수 표본 확인 결과 1024×768로 완전히 동일
- 전체 3,214개 제품 중 FAIL 라벨이 있는 건 **9개(0.28%)**, 그마저도 13개 기종 중 3개 기종에만 존재

자세한 내용과 근거(정렬 난이도, 조명 변동성, 결함 사례 갤러리 등)는 [reports/eda_summary.md](reports/eda_summary.md) 참고.

## 5. 실행법 (뼈대)

```bash
# 1) 환경 설정 (Python 3.10+ 권장)
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt   # torch 는 CPU 빌드로 설치됨

# 2) 데이터 배치
#    data/raw/ 에 기종별 폴더(예 3029C003AA/2EQ16144/step_0.jpg ...)를 넣는다

# 3) 인벤토리/EDA 재현 (선택 — data/processed/eda/*.csv 는 이미 git에 포함되어 있음)
python scripts/build_inventory.py
python scripts/eda_scan.py

# 4) 설정 확인
#    configs/config.yaml 의 기종/정렬/threshold 확인·조정

# 5) 학습 (정상 데이터로 스텝별 모델 + threshold 산출)
python -m scripts.train --config configs/config.yaml --machine 3029C003AA

# 6) 평가
python -m scripts.evaluate --config configs/config.yaml --machine 3029C003AA --run <run_name>

# 7) 추론
python -m scripts.infer --config configs/config.yaml --machine 3029C003AA --run <run_name> --image path/to/img.jpg
```

## 6. TODO / 로드맵

- [x] EDA — 데이터 구조/불균형/정렬난이도/조명변동/결함 사례 파악 (`reports/eda_summary.md`, `notebooks/01_eda.ipynb`)
- [x] config 를 실제 13개 기종·스텝 수로 갱신
- [ ] 데이터 로더·전처리 구현 (`src/data`) — 정렬(affine + 이중 게이트) 포함
- [ ] 베이스라인: 수작업 특징 + 가우시안 모델
- [ ] 스코어 정규화 및 threshold 산출 구현 (`src/scoring`)
- [ ] 평가 지표·시각화 구현 (`src/eval`)
- [ ] 패치 k-NN / AutoEncoder 로 확장 및 비교
- [ ] 결과 리포트 작성 (`reports/`)
