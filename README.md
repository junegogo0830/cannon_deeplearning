# 산업 이미지 PASS/FAIL 판정 (Canon 화상검사)

핸디형 카메라로 찍은 제품 사진에서 **기종·스텝별 PASS/FAIL**을 판정하는 과제 프로젝트예요. 과제 조건은 FAIL 데이터가 극소이고, GPU 없이 CPU만 쓸 수 있다는 점이에요.

> 상태: 데이터 탐색과 이상탐지 파이프라인은 구현·검증했어요. 라벨 의미와 체크포인트-스텝 대응은 **조교님 확인 대기** 중이에요.

---

## 1. 데이터 (실측)

```
data/raw/<기종코드>/<제품시리얼>/step_N.jpg              # 정상 촬영 사진 (N = 0부터)
data/raw/<기종코드>/<제품시리얼>/NG/<timestamp>_..._step_N.jpg   # NG 폴더 사진 (12장)
```

| 항목 | 값 |
|---|---|
| 전체 파일 | 47,270 (jpg 47,267, 그 외 3) |
| 기종 | 13개 (`3029C003AA` … `3031C003AA`) |
| 제품 | 3,214 (구조 이상으로 빠진 2개 포함 시 3,216) |
| 스텝 수 | 기종마다 다름: 13 / 15 / 18 / 20 (`3029C004AA` 0~17, `3029C010AA` 0~19) |
| 해상도 | 전부 1024×768, RGB |
| 디코딩 실패 / 완전 중복 | 0 / 0 |
| NG | 12장, 9개 제품, 3개 기종(3029C003AA·3029C004AA·3029C009AA) |

- **라벨 CSV는 없어요.** 정상과 NG는 폴더 위치로만 구분돼요.
- **중첩 폴더 2개**(`2EQ16161/2EQ16160`, `2EQ16974/2EQ16973`, 각 15장)가 제품 폴더 안에 잘못 들어가 있어요. 현재 파이프라인은 이걸 읽지 않아요.

### 확인된 것과 아직 아닌 것

| 구분 | 내용 | 상태 |
|---|---|---|
| 일람표(`T595NP_층별일람표_2.xlsx`) | 기종별 검사 부품과 체크포인트 ①~⑮ | 원본 확인 |
| 체크포인트 N ↔ 사진 `step_N` | 3029C003AA의 ①③⑨와 3029C010AA의 ⑪에서 일치 확인 | 기종 일부만 확인 |
| step_0 | 바코드 명판 (13개 기종 공통, 체크포인트 없음) | 확인 |
| NG 폴더 = 불량 정답 | 미확정. 파일명 스텝이 사진과 다른 사례가 있음 (예: 2EQ16580 step_9 두 번째 사진 = step_10 내용) | **조교님 확인 필요** |
| 루트 사진 = 정상 | 가정. 수리·재촬영 여부 미확정 | **조교님 확인 필요** |
| 일람표의 16~19번 | 항목 없음 | 문서 범위 밖 |

---

## 2. 접근

1. **스텝별 이상탐지 (기본)**: 정상 사진만으로 기종·스텝마다 "평소 모습"을 배우고, 벗어나면 이상으로 봐요.
2. **부품별 검사 (계획)**: 일람표에 항목이 있고 참고 사진으로 위치를 확인한 부품은, 그 부품 영역만 잘라서 검사해요. (N = step 가정 하에)
3. 지도학습은 FAIL이 극소라 사용하지 않아요.

평가는 정상 오판율(FPR)과 실제 NG 검출 여부를 함께 봐요. 12장으로 불량 성능을 단정하지 않아요.

---

## 3. 주요 결과 요약

- **임계값 검증 표본 문제**: 검증 표본이 8~12장이면 임계값이 실제 정상 분포보다 한참 낮게 잡혀요. 검증 표본을 제품 단위로 150장 이상 따로 떼도록 고쳤어요(`min_val`). 이후 FPR이 0.8~2.7%로 안정됐어요.
- **정렬 정책**: 실패 시 제외하는 방식은 데이터를 최대 55%까지 버리고, 실제 FAIL 검출이 우연에 기댔어요. 원본을 그대로 쓰는 완화 방식이 더 안정적이었어요.
- **방법 비교 (3029C003AA, step 2·14)**: patch-kNN(AUROC 0.74~0.80) > PaDiM(0.71~0.75) > AutoEncoder(0.25~0.34, 무작위보다 낮음).
- **한계**: 뚜렷한 결함(라벨 누락)은 잡히지만, 미묘한 결함(이음새 간격, 라벨 문구)은 현재 방법으로 잡히지 않아요. 실제 FAIL 예시가 스텝당 1~2개뿐이라 튜닝과 검증이 어려워요.

---

## 4. 저장소 구조

```
src/                       # 파이프라인
  data/                    # 로딩, 정렬, 분할(제품 단위 누수 검사 포함)
  features/                # 핸드크래프트 특징, 고정 CNN 임베딩
  models/                  # gaussian, patch_knn, padim, autoencoder
  scoring/                 # 집계, 정규화, 임계값, PASS/FAIL 판정
  eval/                    # 지표, 합성 결함, 시각화
  pipeline.py              # 학습·추론 공통 흐름
scripts/
  build_inventory.py       # NG 라벨 목록 생성
  eda_full.py              # 전수 EDA (구조·무결성·밝기·선명도·NG 대조·정렬 통계)
  step_audit.py            # 기종×스텝 감사 (contact sheet, 기종 간 유사도)
  step_compare_table.py    # 기종×스텝 비교표
  train.py / evaluate.py / infer.py   # 학습·평가·추론 CLI
  compare_ad_methods.py    # 방법 비교 실험
configs/
  config.yaml              # handcrafted 특징 설정
  config_cnn.yaml          # CNN 임베딩 설정
reports/
  eda_summary.md           # 1차 EDA 요약
  eda_full.md              # 전수 EDA 보고서
  step_audit.md            # 기종×스텝 감사 (195행)
  progress_summary.md      # 팀 공유용 진행 요약
  eda/                     # 예시 이미지, 스텝 카탈로그
notebooks/01_eda.ipynb     # EDA 노트북
data/processed/            # 가벼운 결과 CSV (대용량은 .gitignore)
```

---

## 5. 실행

```bash
python -m pip install -r requirements.txt

# 1) 데이터 목록과 NG 라벨
python scripts/build_inventory.py

# 2) 전수 EDA (약 4분)
python scripts/eda_full.py

# 3) 기종×스텝 감사
python scripts/step_audit.py
python scripts/step_compare_table.py

# 4) 학습·평가 (예: 3029C003AA의 2, 9, 14번 스텝)
python -m scripts.train    --config configs/config.yaml     --machine 3029C003AA --steps 2 9 14 --run demo --min-val 150 --max-train 60
python -m scripts.evaluate --config configs/config.yaml     --machine 3029C003AA --steps 2 9 14 --run demo --n-synthetic 15

# 5) 방법 비교
python -m scripts.compare_ad_methods --machine 3029C003AA --steps 2 14
```

> Windows에서는 `import torch`를 `cv2`/`numpy`보다 먼저 import해야 DLL 오류가 나지 않아요. 진입 스크립트에 이미 반영돼 있어요.

결과(`results/`)와 원본 데이터(`data/raw/`)는 저장소에 올리지 않아요.

---

## 6. 열린 질문 (조교님께 확인)

1. NG 폴더의 사진은 **그 스텝의 불량**인가, **제품 전체의 불량**인가?
2. 체크포인트 N과 사진 파일 `step_N`이 같은 순서인가? (3029C003AA·3029C010AA에서는 같은 것으로 보여요)
3. 루트 사진은 모두 **정상**인가? 같은 제품의 수리·재촬영 사진이 있는가?
4. 일람표는 어느 기종을 기준으로 한 것인가? 16~19번 스텝의 검사 기준은 어디서 확인하는가?

---

## 7. 팀

- 팀원 저장소: [chaehwanjung/deeplearing_project](https://github.com/chaehwanjung/deeplearing_project) (데이터 점검, 스텝 분류 기준선, 보고서 틀)
- 이 저장소는 파이프라인과 EDA를 담당해요.
- 보고서 제출 기한: 10월 16일 (LMS, PDF).
- AI 사용: 분석 스크립트, 보고서 초안, 이미지 판독 보조에 AI(Claude)를 사용했어요. 보고서에 사용 범위를 명시해야 해요.
