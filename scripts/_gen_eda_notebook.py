"""notebooks/01_eda.ipynb 생성 스크립트 (1회성 빌더 — 노트북 내용을 코드로 관리하기 위함).

실행: python scripts/_gen_eda_notebook.py
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))

def code(text):
    cells.append(nbf.v4.new_code_cell(text))

md("""# EDA — 이미지 기반 공장 품질검사 데이터

목적: 모델/판정 로직을 설계하기 전에 실제 raw 데이터의 구조·불균형·촬영 변동성을 정량적으로 파악한다.

요약 리포트: `reports/eda_summary.md` (이 노트북의 코드로 재현한 표/수치를 근거로 작성됨)
""")

md("## 0. 준비")
code("""import os, re, random, time
import numpy as np
import pandas as pd
import cv2

pd.set_option('display.width', 200)
RAW_DIR = 'data/raw'
OUT_DIR = 'data/processed/eda'
os.makedirs(OUT_DIR, exist_ok=True)

def imread(path: str):
    \"\"\"한글/유니코드 경로에서도 안전하게 이미지를 읽는다.\"\"\"
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
""")

md("""## 1. 인벤토리 구축

`data/raw`를 전수 스캔해 (기종, 제품, 스텝) 목록과, NG 폴더 안의 실패 스텝 목록(유일한 FAIL 라벨 소스)을 만든다.
이미지 디코딩 없이 파일 시스템만 훑으므로 47,000여 개 파일도 빠르게 처리된다.
""")
code("""from importlib import reload
import sys
sys.path.insert(0, 'scripts')
import build_inventory
rows, ng_rows = build_inventory.scan()
print('root-level step images:', len(rows))
print('NG-flagged images:', len(ng_rows))

inv = pd.DataFrame(rows, columns=['machine_type','product_id','step','filename','size_bytes'])
inv['step'] = inv['step'].astype(int)
ng = pd.DataFrame(ng_rows, columns=['machine_type','product_id','step','filename','size_bytes','cell','process','timestamp'])
inv.to_csv(f'{OUT_DIR}/inventory.csv', index=False)
ng.to_csv(f'{OUT_DIR}/ng_events.csv', index=False)
inv.head()
""")

md("""## 2. 기종별 규모 — 제품 수, 스텝 수

과제에서 "15개 스텝"이라 했지만, 실제로는 **기종마다 스텝 수가 다르다** (13/15/18/20). 또한 기종코드(대오더)는
마케팅 모델명이 아니라 생산 주문 코드이며, `T595NP_층별일람표_2.xlsx`의 "화상검사Point" 시트에 기종별 실제
제품명·검사 체크포인트 매핑이 정리되어 있다 (별도 확인, 본 노트북에서는 다루지 않음).
""")
code("""summary = inv.groupby('machine_type').agg(
    n_products=('product_id','nunique'),
    n_images=('filename','count'),
    min_step=('step','min'),
    max_step=('step','max'),
).reset_index()

per_product_stepcount = inv.groupby(['machine_type','product_id'])['step'].count().reset_index(name='n_steps')
mode_per_machine = per_product_stepcount.groupby('machine_type')['n_steps'].agg(lambda s: s.mode().iloc[0]).reset_index(name='mode_n_steps')
deviation = per_product_stepcount.merge(mode_per_machine, on='machine_type')
deviation['is_off'] = deviation['n_steps'] != deviation['mode_n_steps']
off_counts = deviation.groupby('machine_type')['is_off'].sum().reset_index(name='n_products_off_mode')

summary = summary.merge(mode_per_machine, on='machine_type').merge(off_counts, on='machine_type')
summary = summary.sort_values('n_products', ascending=False)
summary.to_csv(f'{OUT_DIR}/machine_summary.csv', index=False)
summary
""")

md("""**확인**: `n_products_off_mode`가 전부 0 → 기종 내에서 제품마다 스텝 수가 완전히 일정함 (결측 스텝 없음).
스텝 수는 기종의 고정 속성으로 config에 반영하면 된다.
""")

md("""## 3. FAIL 라벨 현황 — 극심한 불균형

NG 폴더가 유일한 FAIL 라벨 소스다. 전체 3,214개 제품 중 몇 개가, 어느 기종에 FAIL로 표시돼 있는지 확인한다.
""")
code("""ng_summary = ng.groupby('machine_type').agg(
    n_fail_products=('product_id','nunique'),
    n_fail_step_instances=('filename','count'),
).reset_index()
total_products = inv.groupby('machine_type')['product_id'].nunique().reset_index(name='n_products')
ng_summary = total_products.merge(ng_summary, on='machine_type', how='left').fillna(0)
ng_summary['fail_rate_%'] = (ng_summary['n_fail_products'] / ng_summary['n_products'] * 100).round(3)
ng_summary.sort_values('n_fail_products', ascending=False)
""")

md("""→ FAIL 있는 기종은 13개 중 3개(`3029C003AA`, `3029C004AA`, `3029C009AA`)뿐이고, 그마저도 9개 제품(0.28%)이다.
나머지 10개 기종은 FAIL 라벨이 0건 → 그 기종들은 **정량 평가가 원천적으로 불가능**하다 (합성 결함 주입으로 보완 필요).
""")

md("""## 4. (기종, 스텝)별 표본 스캔 — 해상도 / 디코딩 / 정렬난이도 / 밝기 / 색상

`scripts/eda_scan.py`와 동일한 로직. (기종,스텝) 조합마다 정상 제품 8개를 무작위 표본으로 뽑아
① 디코딩 성공 여부, ② 해상도, ③ 밝기·색상(Hue/Sat) 평균, ④ 기준 이미지 대비 ORB+RANSAC affine
정렬 추정(이동/회전/배율, inlier 비율)을 계산한다. 195개 조합 × 8장 = 약 1,400장, 전체 90초 내외.
""")
code("""SAMPLES_PER_STEP = 8
SEED = 42
random.seed(SEED)

scan_rows = []
t0 = time.time()

for machine in sorted(inv.machine_type.unique()):
    sub = inv[inv.machine_type == machine]
    for step in sorted(sub.step.unique()):
        prods = sub[sub.step == step]['product_id'].unique().tolist()
        sample = random.sample(prods, min(SAMPLES_PER_STEP, len(prods)))

        orb = cv2.ORB_create(1500)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        ref_kp = ref_des = None

        for i, pid in enumerate(sample):
            path = f'data/raw/{machine}/{pid}/step_{step}.jpg'
            img = imread(path)
            row = {'machine_type': machine, 'step': step, 'product_id': pid, 'decode_ok': img is not None,
                   'width': None, 'height': None, 'mean_brightness': None, 'mean_hue': None, 'mean_sat': None,
                   'is_ref': i == 0, 'orb_matches': None, 'orb_inliers': None, 'orb_inlier_ratio': None,
                   'dx': None, 'dy': None, 'rot_deg': None, 'scale': None}
            if img is None:
                scan_rows.append(row); continue
            h, w = img.shape[:2]
            row['width'], row['height'] = w, h
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            row['mean_brightness'] = float(gray.mean())
            row['mean_hue'] = float(hsv[..., 0].mean())
            row['mean_sat'] = float(hsv[..., 1].mean())

            if i == 0:
                ref_kp, ref_des = orb.detectAndCompute(gray, None)
                scan_rows.append(row); continue

            kp, des = orb.detectAndCompute(gray, None)
            if des is None or ref_des is None or len(kp) < 8:
                scan_rows.append(row); continue
            matches = sorted(bf.match(ref_des, des), key=lambda m: m.distance)[:200]
            row['orb_matches'] = len(matches)
            if len(matches) >= 8:
                src = np.float32([ref_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                dst = np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                M, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=5.0)
                if M is not None:
                    row['orb_inliers'] = int(inliers.sum())
                    row['orb_inlier_ratio'] = float(inliers.sum() / len(inliers))
                    row['dx'], row['dy'] = float(M[0, 2]), float(M[1, 2])
                    row['scale'] = float(np.sqrt(M[0, 0] ** 2 + M[1, 0] ** 2))
                    row['rot_deg'] = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
            scan_rows.append(row)
    print(f'  done {machine} - elapsed {time.time()-t0:.1f}s')

step_scan = pd.DataFrame(scan_rows)
step_scan.to_csv(f'{OUT_DIR}/step_scan.csv', index=False)
print('saved', len(step_scan), 'rows')
""")

md("### 4-1. 해상도 / 디코딩 점검")
code("""print('decode failures:', int((~step_scan.decode_ok).sum()), '/', len(step_scan))
res = step_scan.dropna(subset=['width','height']).groupby('machine_type').apply(
    lambda g: sorted(set(zip(g.width.astype(int), g.height.astype(int))))
).reset_index(name='resolutions')
res
""")

md("### 4-2. 정렬(registration) 난이도")
code("""nonref = step_scan[step_scan.is_ref == False].copy()
nonref['implausible'] = (nonref.dx.abs() > 200) | (nonref.dy.abs() > 150) | (nonref.rot_deg.abs() > 20)

align = nonref.groupby(['machine_type','step']).agg(
    n=('orb_inlier_ratio','count'),
    mean_inlier_ratio=('orb_inlier_ratio','mean'),
    pct_implausible=('implausible','mean'),
    mean_abs_dx=('dx', lambda s: s.abs().mean()),
    mean_abs_dy=('dy', lambda s: s.abs().mean()),
    mean_abs_rot=('rot_deg', lambda s: s.abs().mean()),
).reset_index()
align.to_csv(f'{OUT_DIR}/alignment_by_step.csv', index=False)

print('overall mean inlier_ratio:', round(align.mean_inlier_ratio.mean(), 3))
print('combos with inlier_ratio < 0.3:', (align.mean_inlier_ratio < 0.3).sum(), '/', len(align))
print('combos with >=50% implausible affine estimates:', (align.pct_implausible >= 0.5).sum(), '/', len(align))
align.sort_values('mean_inlier_ratio').head(10)
""")

md("""**주의**: inlier 비율이 낮다고 해서 실제 handheld 편차가 큰 것은 아니다. 예를 들어 `3030C001AA` step9(전원
스위치 구간)는 표면이 평평·대칭적이라 ORB가 모호하게 매칭되어 "물리적으로 비현실적인" 변환이 자주 추정되지만,
실제 이미지 2장을 육안으로 비교하면 구도가 거의 동일하다 (아래 5번 셀에서 직접 확인 가능). 즉 자동 지표만
맹신하지 말고 육안 스팟체크를 병행해야 한다.
""")

md("### 4-3. 밝기 / 색상(Hue) 변동성")
code("""bh = step_scan.dropna(subset=['mean_brightness']).groupby(['machine_type','step']).agg(
    n=('mean_brightness','count'),
    brightness_mean=('mean_brightness','mean'),
    brightness_std=('mean_brightness','std'),
    hue_mean=('mean_hue','mean'),
    hue_std=('mean_hue','std'),
    sat_mean=('mean_sat','mean'),
).reset_index()
bh.to_csv(f'{OUT_DIR}/brightness_hue_by_step.csv', index=False)

print('brightness_std: mean=%.1f max=%.1f' % (bh.brightness_std.mean(), bh.brightness_std.max()))
print('hue_std: mean=%.1f max=%.1f' % (bh.hue_std.mean(), bh.hue_std.max()))
bh.sort_values('hue_std', ascending=False).head(10)
""")

md("""**Hue 표준편차 해석 주의**: 채도(`sat_mean`)가 낮은(흰색/회색 표면) 조합은 Hue 각도 자체가 RGB 미세잡음에도
크게 흔들려 표준편차가 커 보일 수 있다(측정 아티팩트). 채도가 충분히 높으면서 Hue 편차도 큰 조합만 "진짜 색상
서브그룹" 후보로 봐야 한다 — 아래 5번 셀에서 실제 사례(파란색 vs 회색 부품)를 육안 확인한다.
""")

md("""## 5. 육안 확인 — NG 사례 갤러리 & 색상 서브그룹

`reports/eda/figs/`에 저장된 예시 이미지. 결함 스펙트럼이 넓다는 것과(라벨 누락처럼 뚜렷한 것부터 이음새
미세 차이까지), 같은 (기종,스텝)의 "정상" 안에도 색상 서브그룹이 존재한다는 것을 육안으로 확인한다.
""")
md("""### Case 1 — 미세한 기구 정렬 차이 (3029C003AA, step14, 제품 2EQ16165)

| PASS | FAIL |
|---|---|
| ![](../reports/eda/figs/case1_pass_2EQ16165_step14.jpg) | ![](../reports/eda/figs/case1_fail_2EQ16165_step14.jpg) |
""")
md("""### Case 2 — 라벨 완전 누락 (3029C003AA, step2, 제품 2EQ17039)

| PASS | FAIL |
|---|---|
| ![](../reports/eda/figs/case2_pass_2EQ17039_step2.jpg) | ![](../reports/eda/figs/case2_fail_2EQ17039_step2.jpg) |

FAIL 쪽에는 "Canon" 로고 라벨 자체가 보이지 않는다 — 육안으로도 명백한 결함.
""")
md("""### Case 3 — 모델 라벨 문자/체크박스 불일치 (3029C009AA, step3, 제품 2EX36524)

| PASS | FAIL |
|---|---|
| ![](../reports/eda/figs/case3_pass_2EX36524_step3.jpg) | ![](../reports/eda/figs/case3_fail_2EX36524_step3.jpg) |
""")
md("""### 색상 서브그룹 — 같은 (기종,스텝)인데 부품 색이 다름 (3029C003AA, step7)

| 기준(회색/베이지) | 다른 정상 제품(파란색, 제품 2EQ16187) |
|---|---|
| ![](../reports/eda/figs/subgroup_reference_gray_step7.jpg) | ![](../reports/eda/figs/subgroup_blue_variant_step7.jpg) |

두 이미지 모두 라벨상 PASS다. 이 차이를 "정상 변동"으로 학습에 포함할지, 별도 서브그룹으로 다룰지는 모델
설계 단계에서 결정해야 한다.
""")

md("""## 6. 결론 (설계에 반영)

1. 정렬은 affine(4DOF) + **이중 게이트**(inlier 비율 + dx/dy/회전 물리적 타당성 bound). Homography는 쓰지 않음
   (매칭이 나쁠 때 과적합해서 더 나빠짐 — 별도 실험으로 확인, `reports/eda_summary.md` 참고).
2. 조명/반사에 강건한 특징(엣지 방향, 국소 대비 정규화, 채도 가중 Hue) 우선. 원본 밝기 직접 사용 지양.
3. 패치는 적당히 크게, 겹치게(overlap) — 정렬 잔차 흡수.
4. 뱅크 검색은 위치 무관(전체 정상 패치 대상 최근접 탐색).
5. 결함 스펙트럼이 넓음 → 최소 2개 스케일(미세/거시) 고려.
6. FAIL 있는 3개 기종은 실제 사례 leave-one-out, 나머지 10개는 합성 결함 주입 + 정상 FPR만.
7. 정렬 신뢰도 지표를 맹신하지 말 것 — 자동 지표 + 육안 스팟체크 병행.

전체 서술형 요약: `reports/eda_summary.md`
""")

nb['cells'] = cells
nb['metadata'] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.10"},
}

with open('notebooks/01_eda.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print('wrote notebooks/01_eda.ipynb with', len(cells), 'cells')
