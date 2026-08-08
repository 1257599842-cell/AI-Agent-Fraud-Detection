"""Q-B：沙盘翻档动图 —— 脚本驱动、可复现，不做任何手工剪辑。

金额从 $20 连续拖到 $500（p 固定 0.30、gang=0），**真实经过两次翻转**：
  $74.33  加验证 → 挂起
  $405.00 挂起  → 拒绝
每一帧都由页面自己的 `costs()` 重算，argmin 高亮随之移动——
**动图里的每个数都是页面算的，不是画上去的。**

- 帧由 WebKit 逐帧截取（与 shoot_demo.py 同引擎，owner 现场用 Safari）
- GIF 由 Pillow 合成（已在环境内，随 matplotlib 而来，**不引新依赖**）
- 金额走**对数**插值：低金额段变化才看得清；线性插值会让 $20→$74 一闪而过
- 两个翻转点各**停留**若干帧，让观众看清档位切换的瞬间
- 无声、循环、只框沙盘区域

产出 reports/demo/shots/sandbox_flip.gif —— 目标 < 3 MB、3–6 秒。

用法：python -m src.serving.shoot_sandbox_gif
"""

import io
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / "reports" / "demo" / "index.html"
OUT = ROOT / "reports" / "demo" / "shots" / "sandbox_flip.gif"

P_FIXED = 0.30
A_START, A_END = 20.0, 500.0
# **不硬编码临界金额**：上一版写死 $74.33（二分到 $0.01 的显示值），
# 而真实翻转点略低，导致「翻转前一瞬」那帧其实已经越过去了——断言当场抓到。
# 改为运行时用同一套 argmin5 二分算出，精度 1e-9。
N_MOVE = 46          # 移动帧
HOLD_ENDS = 6        # 首尾各停留帧数
HOLD_FLIP = 5        # 每个翻转点停留帧数
FRAME_MS = 70        # 77 帧 × 70ms ≈ 5.4s，落在 3–6s（80ms 会到 6.2s，超）
W, H = 1280, 720


def flip_points(P_FIXED):
    """用后端 argmin5 二分求出 [A_START, A_END] 内所有档位切换点，精度 1e-9。

    **P_FIXED 必须是页面实际生效的 p，不是我请求的 p**：p 滑块同样 step=0.01（对数），
    请求 log10(0.30) 会被吸附到 −0.52 → 实际 p=0.302。用 0.300 去算临界点会差 0.27%，
    刚好够让「翻转前一瞬」那帧越界（断言抓到过一次）。
    """
    import numpy as np
    from src.agent.disposition import BASE
    from src.model.stepup import STEPUP as S, argmin5

    def act(a):
        return argmin5(np.array([P_FIXED]), np.array([float(a)]),
                       np.array([0.0]), 76.02, BASE, S)[0]

    pts, prev, prev_a = [], act(A_START), A_START
    for a in np.geomspace(A_START, A_END, 3000)[1:]:
        cur = act(a)
        if cur != prev:
            lo, hi = prev_a, a
            for _ in range(80):
                m = (lo + hi) / 2
                if act(m) == prev:
                    lo = m
                else:
                    hi = m
            pts.append(((lo + hi) / 2, f"{prev}→{cur}"))
            prev = cur
        prev_a = a
    return pts


def amounts(FLIPS):
    """在**滑块自己的网格**上取帧。

    滑块 step=0.01 是**对数刻度**上的 0.01 ≈ 2.33% 的乘法网格。
    上一版按 0.2% 相对偏移取点，根本落不到网格上，会被吸附到别处——
    结果「翻转前一瞬」那帧其实已经越过临界了（断言抓到）。
    走网格既准确，也更诚实：动图就是用户真拖滑块能看到的样子。

    返回 (log10 网格值, 该帧重复次数)。
    """
    STEP = 0.01
    lo = round(math.log10(A_START) / STEP)
    hi = round(math.log10(A_END) / STEP)
    stride = max(1, (hi - lo) // N_MOVE)
    keys = list(range(lo, hi + 1, stride))
    if keys[-1] != hi:
        keys.append(hi)
    # 每个翻转点两侧最近的网格点必须入帧并停留——否则动图里翻转是"跳"过去的
    holds = {lo: HOLD_ENDS, hi: HOLD_ENDS}
    for a, _ in FLIPS:
        k = math.log10(a) / STEP
        for kk in (math.floor(k), math.ceil(k)):
            if kk not in keys:
                keys.append(kk)
            holds[kk] = HOLD_FLIP
    keys = sorted(set(keys))
    return [(k * STEP, holds.get(k, 1)) for k in keys]


def main():
    from PIL import Image
    from playwright.sync_api import sync_playwright

    frames, labels, errors = [], [], []
    with sync_playwright() as pw:
        b = pw.webkit.launch()
        pg = b.new_page(viewport={"width": W, "height": H}, device_scale_factor=1)
        pg.on("pageerror", lambda e: errors.append(str(e)))
        pg.goto(PAGE.as_uri())
        pg.wait_for_timeout(400)
        # 隐式拼接时**只有带 f 前缀的那段**会解转义，后段的 }} 会原样留下多一个花括号。
        # 统一用单个 f-string，别拼。
        set_slider = ("el=>{{el.value='{:.6f}';"
                      "el.dispatchEvent(new Event('input'))}}")
        pg.eval_on_selector("#p", set_slider.format(math.log10(P_FIXED)))
        pg.wait_for_timeout(150)
        p_eff = pg.eval_on_selector("#p", "el=>Math.pow(10, parseFloat(el.value))")
        FLIPS = flip_points(p_eff)
        print(f"请求 p={P_FIXED} → 滑块吸附后**实际生效 p={p_eff:.6f}**")
        print("按实际 p 由后端二分求得的档位切换点（精度 1e-9）：")
        for a, name in FLIPS:
            print(f"  ${a:.6f}  {name}")
        seq = amounts(FLIPS)
        card = pg.query_selector("section:nth-of-type(2) .grid2")

        for lg, rep in seq:
            pg.eval_on_selector("#a", set_slider.format(lg))
            pg.wait_for_timeout(20)
            # **读回页面实际取值**，不用我请求的值——滑块会吸附，二者可能不同
            amt, win = pg.evaluate(
                "()=>[Math.pow(10, parseFloat(document.getElementById('a').value)),"
                "document.querySelector('.rowc.win .nm').textContent.trim()]")
            labels.append((amt, win))
            png = card.screenshot()
            im = Image.open(io.BytesIO(png)).convert("RGB")
            frames.extend([im] * rep)
        b.close()

    if errors:
        print("❌ 运行时报错：", errors[:3])
        sys.exit(1)

    # 翻转是否真的发生在预期金额：不靠肉眼看动图，靠断言
    flips_seen = [(labels[i - 1], labels[i]) for i in range(1, len(labels))
                  if labels[i][1] != labels[i - 1][1]]
    print(f"金额 ${A_START:.0f} → ${A_END:.0f}（p 实际 {labels and p_eff:.6f}），"
          f"共 {len(labels)} 个采样点")
    print(f"实际发生 {len(flips_seen)} 次翻转：")
    for (a0, w0), (a1, w1) in flips_seen:
        print(f"  ${a0:>7.2f} {w0} → ${a1:>7.2f} {w1}")
    assert len(flips_seen) == len(FLIPS), \
        f"预期 {len(FLIPS)} 次翻转，实际 {len(flips_seen)} 次"
    for ((a0, _), (a1, _)), (expect, name) in zip(flips_seen, FLIPS):
        assert a0 <= expect <= a1, \
            f"{name} 临界 ${expect:.4f} 未落在观测区间 [${a0:.4f}, ${a1:.4f}]"
        # 滑块 step=0.01（对数）≈ 2.33% 的乘法网格 → 一步之内夹住已是页面能表达的最细
        assert a1 / a0 < 1.024, \
            f"{name} 观测区间跨了不止一个滑块刻度（{a0:.2f}→{a1:.2f}），翻转看不清"

    # 调色板量化到 128 色：沙盘只有纯色块与文字，视觉无损而体积减半
    qs = [f.quantize(colors=128, method=Image.Quantize.MEDIANCUT) for f in frames]
    qs[0].save(OUT, save_all=True, append_images=qs[1:], loop=0,
               duration=FRAME_MS, optimize=True, disposal=2)
    mb = OUT.stat().st_size / 1024 / 1024
    secs = len(frames) * FRAME_MS / 1000
    print(f"\n帧数 {len(frames)}　时长 {secs:.1f}s　尺寸 {frames[0].size[0]}×{frames[0].size[1]}"
          f"　体积 {mb:.2f} MB")
    ok = mb < 3 and 3 <= secs <= 6
    print(f"{'✅' if ok else '❌'} 约束：< 3 MB（{mb:.2f}）、3–6 秒（{secs:.1f}）")
    print(f"✅ → {OUT.relative_to(ROOT)}")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
