"""통합 검사 파이프라인(src/inspection/)을 눈으로 보는 간단한 데스크톱 GUI.

CLI(`scripts/infer_product.py`)가 내던 결과를 그대로 재사용한다 — 판정 로직을 GUI용으로
새로 만들지 않고, `inspect_product()` 함수를 직접 호출해서 표로 보여주고 선택한 스텝의
사진을 같이 띄운다. 표준 라이브러리 tkinter + 이미 의존성에 있는 Pillow만 쓴다(새 패키지
설치 불필요).

실행: python -m scripts.gui_inspect
"""
from __future__ import annotations

import torch  # noqa: F401,E402  (Windows DLL 순서)

import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import cv2
from PIL import Image, ImageTk

from scripts.infer_product import inspect_product
from src.data.dataset import read_image
from src.inspection.artifacts import ArtifactCache
from src.inspection.registry import InspectionRegistry

RAW_DIR = Path("data/raw")
STATUS_COLOR = {
    "PASS": "#d9f2d9", "FAIL": "#f8d0d0", "REVIEW": "#fdf0c0", "NOT_APPLICABLE": "#e4e4e4",
}

# 바로 눌러볼 수 있는 예시 — report.md에서 실제로 검증된 사례들 (빈 제품 칸으로 헤매지 않도록)
EXAMPLES = [
    ("3029C003AA", "2EQ17039", "Canon 각인 누락 (모듈 A가 FAIL로 잡음)"),
    ("3029C009AA", "2EX36613", "명판 전체 누락 (모듈 A)"),
    ("3029C009AA", "2EX36661", "명판 내용 오부착 2206N→2006N (모듈 B)"),
    ("3029C009AA", "2EX36524", "명판 글자 누락 (모듈 B)"),
    ("3029C003AA", "2EQ16580", "step9 전원스위치 불량 2건 (범용 PaDiM)"),
]


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


class InspectGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title("Canon 화상검사 — 통합 파이프라인 뷰어")
        root.geometry("1180x700")

        self.registry = InspectionRegistry()
        self.cache: ArtifactCache | None = None
        self.rows_data: dict[str, dict] = {}
        self._tkimg = None  # PhotoImage 참조 유지(가비지컬렉션 방지)

        self._build_widgets()

    # ---------- 레이아웃 ----------
    def _build_widgets(self) -> None:
        top = ttk.Frame(self.root, padding=8)
        top.pack(side="top", fill="x")

        ttk.Label(top, text="기종").pack(side="left")
        self.machine_var = tk.StringVar(value=self.registry.machines()[0])
        machine_box = ttk.Combobox(
            top, textvariable=self.machine_var, values=self.registry.machines(), width=14, state="readonly"
        )
        machine_box.pack(side="left", padx=(4, 12))
        machine_box.bind("<<ComboboxSelected>>", self._on_machine_change)

        ttk.Label(top, text="제품 시리얼").pack(side="left")
        self.product_var = tk.StringVar()
        self.product_box = ttk.Combobox(top, textvariable=self.product_var, width=16)
        self.product_box.pack(side="left", padx=(4, 12))

        self.ng_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(top, text="NG 폴더 포함", variable=self.ng_var).pack(side="left", padx=(0, 12))

        self.run_btn = ttk.Button(top, text="검사 실행", command=self.on_run)
        self.run_btn.pack(side="left")

        ttk.Label(top, text="예시:").pack(side="left", padx=(16, 4))
        example_var = tk.StringVar()
        example_box = ttk.Combobox(
            top, textvariable=example_var, width=46, state="readonly",
            values=[f"{m}/{p} — {desc}" for m, p, desc in EXAMPLES],
        )
        example_box.pack(side="left")
        example_box.bind("<<ComboboxSelected>>", lambda e: self._apply_example(example_box.current()))

        self.status_var = tk.StringVar(value="대기 중")
        ttk.Label(self.root, textvariable=self.status_var, padding=(8, 2), foreground="#555").pack(
            side="top", anchor="w"
        )

        self.summary_var = tk.StringVar(value="")
        ttk.Label(
            self.root, textvariable=self.summary_var, padding=(8, 4), font=("", 11, "bold")
        ).pack(side="top", anchor="w")

        body = ttk.Frame(self.root)
        body.pack(side="top", fill="both", expand=True, padx=8, pady=(0, 8))

        # 좌: 스텝별 결과 표
        cols = ("source", "step", "status", "module", "score", "threshold", "reason")
        headers = {"source": "출처", "step": "스텝", "status": "상태", "module": "모듈",
                   "score": "점수", "threshold": "임계값", "reason": "사유"}
        self.tree = ttk.Treeview(body, columns=cols, show="headings", height=22)
        for c in cols:
            self.tree.heading(c, text=headers[c])
            self.tree.column(c, width=90 if c not in ("reason",) else 260, anchor="w")
        self.tree.pack(side="left", fill="both", expand=True)
        self.tree.bind("<<TreeviewSelect>>", self.on_select_row)
        for status, color in STATUS_COLOR.items():
            self.tree.tag_configure(status, background=color)

        scrollbar = ttk.Scrollbar(body, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscroll=scrollbar.set)
        scrollbar.pack(side="left", fill="y")

        # 우: 선택한 스텝 이미지 + 상세
        right = ttk.Frame(body, padding=(10, 0, 0, 0))
        right.pack(side="left", fill="both")
        self.img_label = ttk.Label(right, text="(스텝을 선택하면 사진이 표시됩니다)", relief="groove", anchor="center")
        self.img_label.pack(side="top")
        self.img_label.configure(width=60)
        self.detail_text = tk.Text(right, width=50, height=18, wrap="word", state="disabled")
        self.detail_text.pack(side="top", pady=(8, 0), fill="both", expand=True)

        self._refresh_products()

    # ---------- 이벤트 ----------
    def _on_machine_change(self, _event=None) -> None:
        self._refresh_products()

    def _refresh_products(self) -> None:
        machine_dir = RAW_DIR / self.machine_var.get()
        try:
            products = sorted(p.name for p in machine_dir.iterdir() if p.is_dir())
        except FileNotFoundError:
            products = []
        self.product_box["values"] = products
        if products and not self.product_var.get():
            self.product_var.set(products[0])

    def _apply_example(self, idx: int) -> None:
        if idx < 0:
            return
        machine, product, _desc = EXAMPLES[idx]
        self.machine_var.set(machine)
        self._refresh_products()
        self.product_var.set(product)
        self.on_run()

    def on_run(self) -> None:
        machine = self.machine_var.get()
        product = self.product_var.get().strip()
        if not product:
            messagebox.showwarning("입력 필요", "제품 시리얼을 선택하거나 입력하세요.")
            return
        self.run_btn.config(state="disabled")
        self.status_var.set(f"{machine}/{product} 검사 중... (처음 실행은 모델 로딩 때문에 몇 초 걸려요)")
        self.tree.delete(*self.tree.get_children())
        self.rows_data = {}
        threading.Thread(target=self._run_worker, args=(machine, product, self.ng_var.get()), daemon=True).start()

    def _run_worker(self, machine: str, product: str, include_ng: bool) -> None:
        try:
            if self.cache is None:
                self.cache = ArtifactCache()
            step_decisions, ng_decisions, product_summary = inspect_product(
                machine, product, self.registry, self.cache, include_ng=include_ng
            )
        except Exception as e:  # noqa: BLE001 — GUI에서는 어떤 예외든 사용자에게 보여줘야 함
            self.root.after(0, lambda: self._on_error(e))
            return
        self.root.after(0, lambda: self._on_done(step_decisions, ng_decisions, product_summary))

    def _on_error(self, e: Exception) -> None:
        self.status_var.set("오류")
        self.run_btn.config(state="normal")
        messagebox.showerror("검사 실패", str(e))

    def _on_done(self, step_decisions: dict, ng_decisions: list, product_summary: dict) -> None:
        for step in sorted(step_decisions):
            d = step_decisions[step]
            self._insert_row("root", d)
        for d in ng_decisions:
            self._insert_row("NG", d)

        skipped = product_summary.get("skipped_steps_no_image") or []
        skip_note = f" (사진 없는 스텝 {skipped} 제외)" if skipped else ""
        self.summary_var.set(
            f"최종 판정: {product_summary['product_status']}  "
            f"(PASS {product_summary['n_pass']} / FAIL {product_summary['n_fail']} / REVIEW {product_summary['n_review']})"
            f"{skip_note}"
        )
        self.status_var.set("완료")
        self.run_btn.config(state="normal")

    def _insert_row(self, source: str, d: dict) -> None:
        iid = self.tree.insert(
            "", "end",
            values=(source, d["step"], d["status"], d["representative_module"] or "-",
                    _fmt(d["representative_score"]), _fmt(d["representative_threshold"]), d["reason"] or ""),
            tags=(d["status"],),
        )
        self.rows_data[iid] = d

    def on_select_row(self, _event=None) -> None:
        sel = self.tree.selection()
        if not sel:
            return
        d = self.rows_data.get(sel[0])
        if d is None:
            return

        try:
            img = read_image(d["image_path"])
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            pil_img.thumbnail((420, 420))
            self._tkimg = ImageTk.PhotoImage(pil_img)
            self.img_label.configure(image=self._tkimg, text="")
        except Exception as e:  # noqa: BLE001
            self.img_label.configure(image="", text=f"(이미지를 못 불러옴: {e})")

        lines = [
            f"기종/스텝: {d['machine']} / step{d['step']}",
            f"부품명: {d.get('part_name') or '체크포인트 미확인'} (checkpoint={d.get('checkpoint')})",
            f"상태: {d['status']}  (대표 모듈={d['representative_module']})",
            f"점수/임계값: {_fmt(d['representative_score'])} / {_fmt(d['representative_threshold'])}",
            f"판정 근거: {d['decision_source']}",
            f"사유: {d['reason'] or '-'}",
            f"실행된 모듈: {d['modules_executed']}",
            f"mapping_verified={d['mapping_verified']}  module_validated={d['module_validated']}",
            "",
            "모듈별 상세:",
        ]
        for mod, r in d["module_results"].items():
            lines.append(f"  [{mod}] {r['status']}  score={_fmt(r.get('raw_score'))}  "
                         f"threshold={_fmt(r.get('threshold'))}  elapsed={r.get('elapsed_s')}s")
        self.detail_text.configure(state="normal")
        self.detail_text.delete("1.0", "end")
        self.detail_text.insert("1.0", "\n".join(lines))
        self.detail_text.configure(state="disabled")


def main() -> None:
    root = tk.Tk()
    InspectGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
