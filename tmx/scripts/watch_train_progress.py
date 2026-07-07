#!/usr/bin/env python3
"""实时跟踪 AffectGPT 训练日志：epoch 进度、iter loss、每 epoch 汇总。"""

from __future__ import annotations

import argparse
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

EPOCH_START = re.compile(
    r"Start training epoch (\d+), (\d+) iters per inner epoch\."
)
EPOCH_STATS = re.compile(
    r"Averaged stats: lr: ([\d.eE+-]+)\s+loss: ([\d.eE+-]+)"
)
ITER_LINE = re.compile(
    r"Train: data epoch: \[(\d+)\]\s+\[(\d+)/(\d+)\].*?"
    r"loss: ([\d.eE+-]+).*?max mem: (\d+)"
)
EPOCH_TOTAL = re.compile(
    r"Train: data epoch: \[(\d+)\] Total time: .*?\(([\d.]+) s / it\)"
)
CKPT = re.compile(
    r"Saving checkpoint at epoch (\d+) to .*checkpoint_\d+_loss_([\d.]+)\.pth"
)

DEFAULT_LOG = Path(__file__).resolve().parents[1] / "logs" / "train_human_full.log"
MAX_EPOCH = 60
ITERS_PER_EPOCH = 500


@dataclass
class MonitorState:
    max_epoch: int = MAX_EPOCH
    iters_per_epoch: int = ITERS_PER_EPOCH
    current_epoch: int = 0
    current_iter: int = 0
    current_loss: float | None = None
    current_lr: float | None = None
    max_mem_mb: int = 0
    sec_per_iter: float | None = None
    epoch_times: list[float] = field(default_factory=list)
    epoch_losses: list[tuple[int, float, float]] = field(default_factory=list)
    epoch_start_ts: float | None = None
    training_started: bool = False

    def eta_total_sec(self) -> float | None:
        if not self.epoch_times or self.current_epoch <= 0:
            if self.sec_per_iter:
                remaining = (self.max_epoch * self.iters_per_epoch) - (
                    (self.current_epoch - 1) * self.iters_per_epoch + self.current_iter
                )
                return max(0.0, remaining * self.sec_per_iter)
            return None
        avg_epoch = sum(self.epoch_times) / len(self.epoch_times)
        done = len(self.epoch_losses)
        return max(0.0, (self.max_epoch - done) * avg_epoch)

    def fmt_eta(self, sec: float | None) -> str:
        if sec is None:
            return "--:--:--"
        sec = int(sec)
        h, rem = divmod(sec, 3600)
        m, s = divmod(rem, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"


def _bar(done: int, total: int, width: int = 30) -> str:
    if total <= 0:
        return "[" + " " * width + "]"
    filled = int(width * done / total)
    filled = min(width, max(0, filled))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def _print_header(log_path: Path, st: MonitorState) -> None:
    print("=" * 72)
    print("MER2026 Human-OV 训练监控")
    print(f"日志: {log_path}")
    print(
        f"目标: {st.max_epoch} epochs × {st.iters_per_epoch} iters | "
        f"batch_size_train=3"
    )
    print("Ctrl+C 退出监控（训练继续在后台运行）")
    print("=" * 72)


def _print_status(st: MonitorState, *, newline: bool = False) -> None:
    if st.current_epoch <= 0:
        msg = "等待训练启动（加载模型 / 数据集）..."
    else:
        done_iters = (st.current_epoch - 1) * st.iters_per_epoch + st.current_iter
        total_iters = st.max_epoch * st.iters_per_epoch
        bar = _bar(st.current_iter, st.iters_per_epoch)
        loss_s = f"{st.current_loss:.4f}" if st.current_loss is not None else "  --  "
        mem_s = f"{st.max_mem_mb}MB" if st.max_mem_mb else "--"
        eta_e = "--:--"
        if st.sec_per_iter and st.current_iter < st.iters_per_epoch:
            eta_e = st.fmt_eta(
                (st.iters_per_epoch - st.current_iter) * st.sec_per_iter
            )[-5:]
        msg = (
            f"[Epoch {st.current_epoch:2d}/{st.max_epoch}] "
            f"iter {st.current_iter:3d}/{st.iters_per_epoch} {bar} "
            f"loss={loss_s} mem={mem_s} "
            f"| 总进度 {done_iters}/{total_iters} "
            f"ETA={st.fmt_eta(st.eta_total_sec())}"
        )
    end = "\n" if newline else "\r"
    sys.stdout.write(msg + " " * 8 + end)
    sys.stdout.flush()


def _print_epoch_done(st: MonitorState, epoch: int, lr: float, loss: float) -> None:
    elapsed = time.time() - st.epoch_start_ts if st.epoch_start_ts else 0.0
    if elapsed > 0:
        st.epoch_times.append(elapsed)
    st.epoch_losses.append((epoch, loss, elapsed))
    print()  # newline after \r status
    print("-" * 72)
    print(
        f">>> Epoch {epoch}/{st.max_epoch} 完成 | "
        f"avg_loss={loss:.4f} | lr={lr:.8f} | "
        f"用时={elapsed/60:.1f}min | "
        f"预计剩余={st.fmt_eta(st.eta_total_sec())}"
    )
    print(f"    {'Epoch':>5}  {'Avg Loss':>10}  {'Time(min)':>10}")
    for ep, ls, t in st.epoch_losses[-10:]:
        print(f"    {ep:5d}  {ls:10.4f}  {t/60:10.1f}")
    if len(st.epoch_losses) > 10:
        print(f"    ... 共 {len(st.epoch_losses)} 个 epoch 已完成")
    print("-" * 72)


def process_line(line: str, st: MonitorState) -> None:
    line = line.rstrip("\n")

    m = EPOCH_START.search(line)
    if m:
        st.training_started = True
        st.current_epoch = int(m.group(1))
        st.current_iter = 0
        st.iters_per_epoch = int(m.group(2))
        st.epoch_start_ts = time.time()
        _print_status(st, newline=True)
        return

    m = ITER_LINE.search(line)
    if m:
        st.current_epoch = int(m.group(1))
        st.current_iter = int(m.group(2))
        st.iters_per_epoch = int(m.group(3))
        st.current_loss = float(m.group(4))
        st.max_mem_mb = int(m.group(5))
        _print_status(st)
        return

    m = EPOCH_TOTAL.search(line)
    if m:
        st.sec_per_iter = float(m.group(2))
        return

    m = EPOCH_STATS.search(line)
    if m:
        lr = float(m.group(1))
        loss = float(m.group(2))
        _print_epoch_done(st, st.current_epoch, lr, loss)
        return

    if "Training time" in line and st.current_epoch >= st.max_epoch:
        print()
        print("=" * 72)
        print("训练已全部完成!")
        if st.epoch_losses:
            print(f"最终 loss: {st.epoch_losses[-1][1]:.4f}")
        print("=" * 72)


def tail_follow(path: Path, st: MonitorState) -> None:
    while not path.exists():
        print(f"等待日志文件: {path}")
        time.sleep(1)

    with path.open("r", encoding="utf-8", errors="replace") as f:
        # 回放已有内容
        for line in f:
            process_line(line, st)
        _print_status(st, newline=True)

        while True:
            line = f.readline()
            if line:
                process_line(line, st)
            else:
                time.sleep(0.5)


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch AffectGPT training progress")
    parser.add_argument(
        "--log",
        type=Path,
        default=DEFAULT_LOG,
        help="Training log file path",
    )
    parser.add_argument("--max-epoch", type=int, default=MAX_EPOCH)
    args = parser.parse_args()

    st = MonitorState(max_epoch=args.max_epoch)
    _print_header(args.log, st)
    try:
        tail_follow(args.log, st)
    except KeyboardInterrupt:
        print("\n监控已停止。")


if __name__ == "__main__":
    main()
