"""전수 EDA (팀 보고서 구조 + 우리 근거 추가). 결과는 data/processed/eda_full/ 에 저장.

단계
  1. 파일 목록: 전체 파일을 os.walk로 훑고 경로 구조(정상/NG/중첩/비정상 깊이)를 분류
  2. 파일별 무결성·내용: SHA-256, 디코딩, 해상도/채널, EXIF 방향, 밝기·선명도·채도 (한 번 읽어서 전부 계산)
  3. 구조: 기종별 제품 수, 제품당 스텝 수, 기종 최빈 구성과 비교한 누락/추가 스텝
  4. 완전 중복: SHA-256 동일 묶음
  5. 밝기·선명도 IQR 후보: 기종·스텝별 (참고 20장 미만은 보류), k=1.5/2/3 민감도
  6. NG 대조: NG 사진이 같은 제품의 어느 루트 스텝과 가장 비슷한지 (파일명 스텝과 일치하는지 검증),
     같은 스텝 다른 제품과의 상관 기준선 포함
  7. 정렬 기하통계: 기종·스텝별 ORB+RANSAC 이동/회전/배율 (표본 15개 제품 기준)
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import random
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

RAW = Path("data/raw")
OUT = Path("data/processed/eda_full")
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42
STEP_RE = re.compile(r"^step_(\d+)\.jpg$", re.IGNORECASE)
NG_STEP_RE = re.compile(r"_step_(\d+)\.jpg$", re.IGNORECASE)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------- 1. 파일 목록 + 경로 구조 ----------
def stage1_inventory() -> pd.DataFrame:
    rows = []
    for root, _dirs, files in os.walk(RAW):
        for f in files:
            p = Path(root) / f
            parts = p.relative_to(RAW).parts  # (기종, ..., 파일)
            is_ng = any(part.upper() == "NG" for part in parts[:-1])
            machine = parts[0] if parts else ""
            # 제품 폴더 = 파일 바로 위 폴더 (NG면 그 위)
            parent_dirs = [x for x in parts[:-1] if x.upper() != "NG"]
            product = parent_dirs[-1] if len(parent_dirs) >= 2 else ""
            expected_len = 4 if is_ng else 3
            nested = len(parts) != expected_len
            ext = p.suffix.lower()
            step_m = STEP_RE.match(f) or NG_STEP_RE.search(f)
            rows.append({
                "path": str(p).replace("\\", "/"),
                "rel": "/".join(parts),
                "machine": machine,
                "product": product,
                "depth_parts": len(parts),
                "is_ng": is_ng,
                "nested_or_odd_depth": nested,
                "ext": ext,
                "is_jpg": ext == ".jpg",
                "file": f,
                "step_from_name": int(step_m.group(1)) if step_m else None,
                "is_step_name": bool(STEP_RE.match(f)),
            })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "inventory_all_files.csv", index=False, encoding="utf-8-sig")
    log(f"stage1: 전체 파일 {len(df)}개, jpg {int(df.is_jpg.sum())}, NG {int(df.is_ng.sum())}, 비정상 깊이 {int(df.nested_or_odd_depth.sum())}")
    return df


# ---------- 2. 파일별 무결성·내용 ----------
def analyze_file(path: str) -> dict:
    out = {"path": path, "sha256": None, "decode_ok": False, "h": None, "w": None, "channels": None,
           "pil_mode": None, "exif_orientation": None, "mean_gray": np.nan, "sharpness": np.nan, "mean_sat": np.nan,
           "nbytes": None}
    try:
        data = Path(path).read_bytes()
    except OSError:
        return out
    out["nbytes"] = len(data)
    out["sha256"] = hashlib.sha256(data).hexdigest()
    try:
        pil = Image.open(io.BytesIO(data))
        out["pil_mode"] = pil.mode
        ex = pil.getexif()
        out["exif_orientation"] = ex.get(274) if ex else None
    except Exception:
        pass
    arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        return out
    out["decode_ok"] = True
    out["h"], out["w"], out["channels"] = arr.shape
    gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(arr, cv2.COLOR_BGR2HSV)
    scale = 256 / max(gray.shape)
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    out["mean_gray"] = float(gray.mean())
    out["sharpness"] = float(cv2.Laplacian(small, cv2.CV_64F).var())
    out["mean_sat"] = float(hsv[..., 1].mean())
    return out


def stage2_content(inv: pd.DataFrame) -> pd.DataFrame:
    paths = inv.loc[inv.is_jpg, "path"].tolist()
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=6) as ex:
        res = list(ex.map(analyze_file, paths, chunksize=64))
    df = pd.DataFrame(res)
    df.to_csv(OUT / "image_stats.csv", index=False, encoding="utf-8-sig")
    log(f"stage2: {len(df)}장 분석 {time.time()-t0:.0f}s, 디코딩 실패 {int((~df.decode_ok).sum())}")
    return df


# ---------- 3. 구조 ----------
def stage3_structure(inv: pd.DataFrame, stats: pd.DataFrame) -> dict:
    m = inv.merge(stats[["path", "sha256", "h", "w", "channels", "pil_mode", "exif_orientation"]], on="path", how="left")
    jpg = m[m.is_jpg]
    root = jpg[~jpg.is_ng & ~jpg.nested_or_odd_depth & jpg.is_step_name]
    # 제품 단위 (기종, 제품 폴더명). 중첩 폴더 안의 제품도 별도 제품으로 센다
    prod_key = jpg.assign(prod_key=jpg.machine + "/" + jpg["rel"].str.split("/").str[:-1].str.join("/").str.replace("/NG", "", regex=False))
    prod_key = prod_key[~prod_key.prod_key.str.contains("/NG")]
    per_prod_steps = root.assign(prod_key=root.machine + "/" + root["rel"].str.split("/").str[:-1].str.join("/")) \
        .groupby(["machine", "prod_key"])["step_from_name"].agg(["count", "min", "max"]).reset_index()
    machine_summary = []
    for mach, g in per_prod_steps.groupby("machine"):
        mode = int(g["count"].mode().iloc[0])
        machine_summary.append({
            "machine": mach,
            "products_total": int(g.shape[0]),
            "products_nested": int(g.prod_key.str.count("/").gt(2).sum()),
            "steps_mode": mode,
            "products_off_mode": int((g["count"] != mode).sum()),
            "step_min": int(g["min"].min()), "step_max": int(g["max"].max()),
        })
    ms = pd.DataFrame(machine_summary)
    ms.to_csv(OUT / "structure_by_machine.csv", index=False, encoding="utf-8-sig")
    # 구조 이상 목록
    odd = jpg[jpg.nested_or_odd_depth | (~jpg.is_step_name & ~jpg.is_ng)]
    odd[["rel", "depth_parts", "is_ng", "step_from_name"]].to_csv(OUT / "structure_anomalies.csv", index=False, encoding="utf-8-sig")
    non_jpg = inv[~inv.is_jpg][["rel", "ext"]]
    non_jpg.to_csv(OUT / "non_jpg_files.csv", index=False, encoding="utf-8-sig")
    log(f"stage3: 구조 이상 {len(odd)}장, 비jpg {len(non_jpg)}개, 기종 {len(ms)}개")
    return {
        "n_files_total": int(len(inv)), "n_jpg": int(inv.is_jpg.sum()),
        "n_root_step_images": int(len(root)), "n_ng_images": int(jpg.is_ng.sum()),
        "n_structure_anomaly_images": int(len(odd)), "n_non_jpg": int(len(non_jpg)),
    }


# ---------- 4. 무결성·중복 ----------
def stage4_integrity(stats: pd.DataFrame, inv: pd.DataFrame) -> dict:
    dup = stats.groupby("sha256").filter(lambda g: len(g) > 1).sort_values("sha256")
    dup[["path", "sha256"]].to_csv(OUT / "exact_duplicates.csv", index=False, encoding="utf-8-sig")
    res = {
        "decode_fail": int((~stats.decode_ok).sum()),
        "exact_duplicate_groups": int(dup.sha256.nunique()),
        "exact_duplicate_files": int(len(dup)),
        "resolution_counts": {f"{int(h)}x{int(w)}": int(c) for (h, w), c in
                              stats.groupby(["h", "w"]).size().items()},
        "channel_counts": {str(int(c)): int(n) for c, n in stats.channels.value_counts().items()},
        "pil_mode_counts": stats.pil_mode.value_counts().to_dict(),
        "exif_orientation_counts": {str(k): int(v) for k, v in stats.exif_orientation.value_counts(dropna=False).items()},
    }
    log(f"stage4: 디코딩 실패 {res['decode_fail']}, 완전 중복 묶음 {res['exact_duplicate_groups']}")
    return res


# ---------- 5. IQR 후보 + 민감도 ----------
def stage5_iqr(stats: pd.DataFrame, inv: pd.DataFrame) -> dict:
    m = stats.merge(inv[["path", "machine", "is_ng", "nested_or_odd_depth", "is_step_name", "step_from_name"]], on="path")
    ref = m[(~m.is_ng) & (~m.nested_or_odd_depth) & m.is_step_name & m.decode_ok].copy()
    rows, cand = [], []
    for (mach, step), g in ref.groupby(["machine", "step_from_name"]):
        n = len(g)
        if n < 20:
            rows.append({"machine": mach, "step": step, "n": n, "status": "보류(20장 미만)"})
            continue
        rec = {"machine": mach, "step": step, "n": n, "status": "평가"}
        for col, short in [("mean_gray", "bright"), ("sharpness", "sharp")]:
            q1, q3 = g[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            for k in [1.5, 2.0, 3.0]:
                lo, hi = q1 - k * iqr, q3 + k * iqr
                if short == "bright":
                    flags = (g[col] < lo) | (g[col] > hi)
                else:
                    flags = g[col] < lo
                rec[f"{short}_k{k}"] = int(flags.sum())
                if k == 1.5:
                    rec[f"{short}_lo_k1.5"] = lo
                    rec[f"{short}_hi_k1.5"] = hi if short == "bright" else np.nan
                    gg = g[flags].assign(reason=short)
                    cand.append(gg)
        rows.append(rec)
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "iqr_summary_by_group.csv", index=False, encoding="utf-8-sig")
    if cand:
        pd.concat(cand)[["path", "machine", "step_from_name", "mean_gray", "sharpness", "reason"]] \
            .to_csv(OUT / "iqr_candidates_k1.5.csv", index=False, encoding="utf-8-sig")
    evaluated = summary[summary.status == "평가"]
    res = {
        "groups_total": int(len(summary)), "groups_evaluated": int(len(evaluated)),
        "groups_pending": int((summary.status != "평가").sum()),
        "bright_candidates_k1.5": int(evaluated["bright_k1.5"].sum()),
        "sharp_candidates_k1.5": int(evaluated["sharp_k1.5"].sum()),
        "bright_candidates_k2": int(evaluated["bright_k2.0"].sum()),
        "sharp_candidates_k2": int(evaluated["sharp_k2.0"].sum()),
        "bright_candidates_k3": int(evaluated["bright_k3.0"].sum()),
        "sharp_candidates_k3": int(evaluated["sharp_k3.0"].sum()),
        "groups_sharp_lower_bound_le0": int((evaluated["sharp_lo_k1.5"] <= 0).sum()),
    }
    log(f"stage5: 평가 그룹 {res['groups_evaluated']}, 보류 {res['groups_pending']}")
    return res


# ---------- 6. NG 대조: 파일명 스텝이 사진 내용과 맞는가 ----------
def _norm_small(path: str) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(Path(path).read_bytes(), np.uint8), cv2.IMREAD_GRAYSCALE)
    small = cv2.resize(img, (128, 96), interpolation=cv2.INTER_AREA).astype(np.float32)
    small -= small.mean()
    return small / (small.std() + 1e-6)


def stage6_ng_check(inv: pd.DataFrame) -> dict:
    rng = random.Random(SEED)
    root = inv[inv.is_jpg & ~inv.is_ng & ~inv.nested_or_odd_depth & inv.is_step_name].copy()
    root["prod_key"] = root.machine + "/" + root["rel"].str.split("/").str[:-1].str.join("/")
    ng = inv[inv.is_jpg & inv.is_ng].copy()
    ng["prod_key"] = ng.machine + "/" + ng["rel"].str.split("/").str[:-2].str.join("/")
    rows = []
    for _, r in ng.iterrows():
        same = root[root.prod_key == r.prod_key]
        q = _norm_small(r.path)
        corrs = {}
        for _, s in same.iterrows():
            corrs[int(s.step_from_name)] = float((q * _norm_small(s.path)).mean())
        best_step = max(corrs, key=corrs.get) if corrs else None
        fname_step = int(r.step_from_name)
        # 기준선: 같은 기종·같은 스텝의 다른 제품 사진과의 상관 (무작위 30장)
        others = root[(root.machine == r.machine) & (root.step_from_name == fname_step) & (root.prod_key != r.prod_key)]
        base = others.sample(min(30, len(others)), random_state=SEED)
        base_corr = [float((q * _norm_small(p)).mean()) for p in base.path]
        rows.append({
            "ng_path": r.path, "machine": r.machine, "product": r.prod_key.split("/")[-1],
            "filename_step": fname_step,
            "corr_with_filename_step": corrs.get(fname_step),
            "best_root_step": best_step,
            "corr_best": corrs.get(best_step) if best_step is not None else None,
            "filename_step_is_best": best_step == fname_step,
            "baseline_other_products_same_step_mean": float(np.mean(base_corr)) if base_corr else None,
            "baseline_other_products_same_step_p95": float(np.percentile(base_corr, 95)) if base_corr else None,
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "ng_step_consistency.csv", index=False, encoding="utf-8-sig")
    res = {
        "ng_images": int(len(df)),
        "filename_step_is_best_match": int(df.filename_step_is_best.sum()) if len(df) else 0,
    }
    log(f"stage6: NG {res['ng_images']}장 중 파일명 스텝이 가장 비슷한 루트 스텝인 경우 {res['filename_step_is_best_match']}장")
    return res


# ---------- 7. 정렬 기하통계 (표본 15 제품, 기종·스텝별) ----------
def _orb_pair(ref_gray, img_gray, orb, bf):
    kp1, d1 = orb.detectAndCompute(ref_gray, None)
    kp2, d2 = orb.detectAndCompute(img_gray, None)
    if d1 is None or d2 is None or len(kp1) < 8 or len(kp2) < 8:
        return None
    ms = sorted(bf.match(d1, d2), key=lambda m: m.distance)[:200]
    if len(ms) < 8:
        return None
    src = np.float32([kp1[m.queryIdx].pt for m in ms]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in ms]).reshape(-1, 1, 2)
    M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=5.0)
    if M is None:
        return None
    return {"dx": float(M[0, 2]), "dy": float(M[1, 2]),
            "rot_deg": float(np.degrees(np.arctan2(M[1, 0], M[0, 0]))),
            "scale": float(np.sqrt(M[0, 0] ** 2 + M[1, 0] ** 2)),
            "inlier_ratio": float(inl.sum() / len(inl)), "matches": len(ms)}


def stage7_geometry(inv: pd.DataFrame) -> pd.DataFrame:
    rng = random.Random(SEED)
    root = inv[inv.is_jpg & ~inv.is_ng & ~inv.nested_or_odd_depth & inv.is_step_name].copy()
    orb = cv2.ORB_create(1500)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    rows = []
    t0 = time.time()
    for (mach, step), g in root.groupby(["machine", "step_from_name"]):
        paths = g.path.tolist()
        if len(paths) < 3:
            continue
        sample = rng.sample(paths, min(15, len(paths)))
        ref = cv2.imread(sample[0], cv2.IMREAD_GRAYSCALE) if False else cv2.imdecode(
            np.frombuffer(Path(sample[0]).read_bytes(), np.uint8), cv2.IMREAD_GRAYSCALE)
        for p in sample[1:]:
            img = cv2.imdecode(np.frombuffer(Path(p).read_bytes(), np.uint8), cv2.IMREAD_GRAYSCALE)
            r = _orb_pair(ref, img, orb, bf)
            rows.append({"machine": mach, "step": int(step), "path": p, **(r or {"failed": True})})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "geometry_pairs.csv", index=False, encoding="utf-8-sig")
    if len(df):
        df["implausible"] = (df.dx.abs() > 200) | (df.dy.abs() > 150) | (df.rot_deg.abs() > 20)
        agg = df.groupby(["machine", "step"]).agg(
            n=("path", "count"),
            failed=("failed", lambda s: int(s.fillna(False).sum())),
            inlier_mean=("inlier_ratio", "mean"),
            dx_absmean=("dx", lambda s: s.abs().mean()),
            dy_absmean=("dy", lambda s: s.abs().mean()),
            rot_absmean=("rot_deg", lambda s: s.abs().mean()),
            implausible_rate=("implausible", "mean"),
        ).reset_index()
        agg.to_csv(OUT / "geometry_by_group.csv", index=False, encoding="utf-8-sig")
    log(f"stage7: 기하 쌍 {len(df)}개 {time.time()-t0:.0f}s")
    return df


def main() -> None:
    t_all = time.time()
    inv = stage1_inventory()
    stats = stage2_content(inv)
    summary = {}
    summary["structure"] = stage3_structure(inv, stats)
    summary["integrity"] = stage4_integrity(stats, inv)
    summary["iqr"] = stage5_iqr(stats, inv)
    summary["ng_check"] = stage6_ng_check(inv)
    stage7_geometry(inv)
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    log(f"done. total {time.time()-t_all:.0f}s")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
