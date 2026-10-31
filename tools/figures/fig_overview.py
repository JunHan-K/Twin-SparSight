"""Fig. 1 (training and refinement) and Fig. 2 (inference) of the paper, drawn from real VISION data.

  python tools/figures/fig1_data.py --step train|replay|infer     # once: real crops / maps -> fig1_data/*.npz
  python tools/figures/fig_overview.py        # -> fig1_training.pdf, fig2_inference.pdf

Same data and drawing helpers as fig1_framework.py (the former one-figure overview); the layout follows a
two-figure structure with a title and a one-line role per panel and a track-analysis box next to the inference.
Fig. 1  (a) Twin-View SparSight Training: whole image and native crop -> one MiT-B0 segmenter; stage heads s1,2 (small
        defects) and s3,4 (all defects) feed the gated logit fusion of Eq. (1); size-aware main loss.
        (b) Sparse Defect Replay: (1) native false positives counted per image, (2) replay batch of small-defect crops and
        random background crops of the confused image (not crops at the false-positive locations).
Fig. 2  Global Sight Track -> p_L, candidates and the binary support M (dashed); Native Sight Track on a candidate crop;
        precision mode merges by Eq. (4), recall mode (all tiles) by the maximum. Right: track analysis numbers of
        outputs/vision_ours/analysis/track_disagreement_test.json.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402
from scipy import ndimage  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sds.utils import resolve  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fig1_framework import (ALL, DATA, GT, INK, LOW, MUTED, NAT, NETC, NETF, REP, SIZE,  # noqa: E402
                                          blend, centre, clipped_box, contour, heat, img, net, overlay, rbox,
                                          seg, wblend)

LAB, SMALL, TITLE, SUB = 7.9, 7.4, 9.0, 7.6  # printed at full text width: >= 7 pt
TINT = {"train": ("#eef4fb", "#9db9dd"), "replay": ("#fdf3e8", "#e9b98a"), "infer": ("#f4f0fb", "#b9a5df"),
        "why": ("#faf8fd", "#b9a5df")}
TRACK = "outputs/vision_ours/analysis/track_disagreement_test.json"


def panel(ax, x, y, w, h, title, sub, tint):
    fc, ec = TINT[tint]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.07", fc=fc, ec=ec, lw=0.8,
                                zorder=0))
    ax.text(x + 0.08, y + h - 0.06, title, ha="left", va="top", fontsize=TITLE, weight="bold", color=INK)
    if sub:
        ax.text(x + 0.08, y + h - 0.25, sub, ha="left", va="top", fontsize=SUB, style="italic", color=MUTED)


def canvas(w, y0, y1):
    fig = plt.figure(figsize=(w, y1 - y0))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(y0, y1)
    ax.axis("off")
    return fig, ax


def draw_training(T, R, out):
    """(a) and (b); coordinates as in fig1_framework.py, panels 0.22 in taller for the role line."""
    W, Y0, TOP = 7.16, 1.89, 3.93
    fig, ax = canvas(W, Y0, TOP)
    # ------------------------------------------------------------------ (a) Twin-View SparSight Training
    panel(ax, 0.03, 1.92, 4.38, 1.99, "(a) Twin-View SparSight Training",
          "One shared segmenter learns global context and native detail.", "train")
    wx, wy, ww = 0.12, 2.66, 0.88
    wh = img(ax, overlay(T["whole"], T["whole_g"] > 0, GT, 0.9), wx, wy, ww, ec=LOW, lw=1.0)
    ax.text(wx, wy + wh + 0.04, "global view ($1024^2$)", fontsize=LAB, color=LOW, va="bottom")
    nb = T["native_box"]
    sx = ww / nb[3]
    bx, by, bs = wx + nb[1] * sx, wy + wh - (nb[0] + 384) * sx, 384 * sx
    ax.add_patch(Rectangle((bx, by), bs, bs, fc="none", ec=NAT, lw=0.9, zorder=6))
    cs_n, cy_n = 0.50, 1.99
    cx_n = bx + bs / 2 - cs_n / 2
    img(ax, T["native"], cx_n, cy_n, cs_n, cs_n, ec=NAT, lw=1.0)
    contour(ax, T["native_g"], cx_n, cy_n, cs_n, cs_n, c=GT, lw=0.6)
    seg(ax, [(bx + bs / 2, by), (bx + bs / 2, cy_n + cs_n + 0.01)], c=NAT, lw=0.8)
    ax.text(bx + bs / 2 + 0.05, (wy + cy_n + cs_n) / 2, "every 3rd batch", fontsize=SMALL, color=NAT, va="center")
    ax.text(cx_n + cs_n + 0.05, cy_n + 0.02, "native view\n($384^2$ crop)", fontsize=LAB, color=NAT, va="bottom",
            linespacing=1.0)
    nx, ny = 1.55, 2.98
    xe = net(ax, nx, ny, 0.95, label=False)
    ax.text((nx + xe) / 2, ny + 0.38, "shared MiT-B0 segmenter", ha="center", va="bottom", fontsize=LAB, color=NETC)
    seg(ax, [(wx + ww, wy + wh / 2), (nx - 0.03, wy + wh / 2)], c=LOW)
    seg(ax, [(cx_n + cs_n, cy_n + cs_n * 0.8), (1.43, cy_n + cs_n * 0.8), (1.43, ny - 0.12), (nx - 0.03, ny - 0.12)],
        c=NAT)
    hb_x, hb_w = xe + 0.08, 0.36
    rbox(ax, hb_x, ny - 0.15, hb_w, 0.30, fc=NETF, ec=NETC)
    ax.text(hb_x + hb_w / 2, ny, "main\nhead", ha="center", va="center", fontsize=SMALL, color=NETC, zorder=5,
            linespacing=0.95)
    seg(ax, [(xe, ny), (hb_x - 0.01, ny)], c=NETC)
    fb_x, fb_w = hb_x + hb_w + 0.12, 0.52
    rbox(ax, fb_x, ny - 0.15, fb_w, 0.30, fc="white", ec=NETC, lw=0.9)
    ax.text(fb_x + fb_w / 2, ny, "gated logit\nfusion (1)", ha="center", va="center", fontsize=SMALL,
            color=NETC, zorder=5, linespacing=0.95)
    seg(ax, [(hb_x + hb_w, ny), (fb_x - 0.01, ny)], c=NETC)
    ax.text(hb_x + hb_w + 0.065, ny + 0.03, "$z$", ha="center", va="bottom", fontsize=SMALL, color=NETC)
    tx, ty, ts = 3.70, 2.44, 0.66
    img(ax, wblend(T["native"], T["native_w"], T["native_g"]), tx, ty, ts, ts, ec=SIZE, lw=1.0)
    seg(ax, [(fb_x + fb_w, ny), (tx - 0.02, ny)], c=SIZE)
    ax.text(tx + ts, ty + ts + 0.04, "BCE$\\cdot w(a)$ + Dice", ha="right", va="bottom", fontsize=SMALL, color=SIZE)
    ax.text(tx + ts, ty - 0.04, "size-aware: $w(a)\\leq\\!5\\times$", ha="right", va="top", fontsize=SMALL,
            color=SIZE)
    sy, st = 1.98, 0.34
    sxs = [nx + 0.17 * 0.95 * i + 0.055 * 0.95 for i in range(4)]
    tb = 0.27
    cA, cB = (sxs[0] + sxs[1]) / 2, (sxs[2] + sxs[3]) / 2
    faded = np.repeat((255 - (255 - T["native"].astype(np.float32).mean(2, keepdims=True)) * 0.3), 3, 2)
    for cc, m, c in ((cA, T["native_small"], SIZE), (cB, T["native_g"], ALL)):
        img(ax, overlay(faded, ndimage.binary_dilation(m > 0, iterations=3), c, 1.0), cc - tb / 2, sy + 0.02, tb, tb,
            ec=c, lw=0.8)
    for i, x in enumerate(sxs):
        cc = cA if i < 2 else cB
        seg(ax, [(x, 2.55), (cc + (x - cc) * 0.2, sy + 0.02 + tb + 0.01)], c=SIZE if i < 2 else ALL, lw=0.7)
    lx = cB + tb / 2 + 0.06
    ax.text(lx, sy + st, "stage heads", fontsize=SMALL, color=MUTED, va="top")
    t12 = ax.text(lx, sy + 0.17, "$s_{1,2}$: small defects", fontsize=SMALL, color=SIZE, va="center")
    t34 = ax.text(lx, sy + 0.06, "$s_{3,4}$: all defects", fontsize=SMALL, color=ALL, va="center")
    rnd = fig.canvas.get_renderer()
    pe, pg = fb_x + fb_w * 0.2, fb_x + fb_w * 0.8
    for t, px_, c, lab in ((t12, pe, SIZE, "evidence"), (t34, pg, ALL, "gate")):
        bb = t.get_window_extent(renderer=rnd).transformed(ax.transData.inverted())
        y_ = (bb.y0 + bb.y1) / 2
        seg(ax, [(bb.x1 + 0.03, y_), (px_, y_), (px_, ny - 0.16)], c=c, lw=0.8)
        ax.text(px_ - 0.03, ny - 0.30, lab, fontsize=SMALL, color=c, ha="right", va="center")

    # ------------------------------------------------------------------ (b) Sparse Defect Replay
    panel(ax, 4.48, 1.92, 2.65, 1.99, "(b) Sparse Defect Replay", "Revisit images where native inspection is confused.",
          "replay")
    rw = R["whole"]
    rx, ry, rwid = 4.57, 2.52, 1.00
    rh = img(ax, rw, rx, ry, rwid)
    s_ = rwid / rw.shape[1]
    lab_, _ = ndimage.label(R["whole_g"] > 0)
    small_ids = {lab_[min(int(round(y)), lab_.shape[0] - 1), min(int(round(x)), lab_.shape[1] - 1)]
                 for x, y in R["pos_xy"]} - {0}
    for j, sl in enumerate(ndimage.find_objects(lab_)):
        y0, y1, x0, x1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
        w_, h_ = max((x1 - x0) * s_, 0.05), max((y1 - y0) * s_, 0.05)
        cxp, cyp = rx + (x0 + x1) / 2 * s_, ry + rh - (y0 + y1) / 2 * s_
        ax.add_patch(Rectangle((cxp - w_ / 2 - 0.01, cyp - h_ / 2 - 0.01), w_ + 0.02, h_ + 0.02, fc="none",
                               ec=SIZE if j + 1 in small_ids else GT, lw=0.8, zorder=6))
    for fx, fy, _ in R["fp_xy"]:
        ax.add_patch(Rectangle((rx + fx * s_ - 0.03, ry + rh - fy * s_ - 0.03), 0.06, 0.06, fc="none", ec=REP, lw=0.7,
                               zorder=6))
    ax.text(rx, ry + rh + 0.04, "1. native false positives", fontsize=LAB, color=REP, va="bottom")
    ax.text(rx + rwid / 2, ry - 0.03, f"counted per image ({len(R['fp_xy'])} here)", fontsize=SMALL, color=REP,
            ha="center", va="top")
    cs, gp, gx0 = 0.30, 0.03, 5.86
    bgy, psy = 3.00, 2.50
    for i, c in enumerate(R["bg_crops"]):
        img(ax, c, gx0 + i * (cs + gp), bgy, cs, cs, ec=REP, lw=0.7)
    for i, c in enumerate(R["pos_crops"]):
        img(ax, c, gx0 + i * (cs + gp), psy, cs, cs, ec=SIZE, lw=0.7)
    pad = 0.035
    bg_x1 = gx0 + 3 * cs + 2 * gp + pad
    ps_x1 = gx0 + 2 * cs + gp + pad
    rbox(ax, gx0 - pad, bgy - pad, bg_x1 - gx0 + pad, cs + 2 * pad, fc="none", ec=REP, lw=0.9, r=0.03, z=5)
    rbox(ax, gx0 - pad, psy - pad, ps_x1 - gx0 + pad, cs + 2 * pad, fc="none", ec=SIZE, lw=0.9, r=0.03, z=5)
    ax.text(bg_x1, bgy + cs + pad + 0.03, "2. random background crops", fontsize=SMALL, color=REP, va="bottom",
            ha="right")
    ax.text(ps_x1 + 0.05, psy + cs / 2, "small-\ndefect\ncrops", fontsize=SMALL, color=SIZE, va="center",
            linespacing=1.0)
    seg(ax, [(rx + rwid, bgy + cs / 2), (gx0 - pad - 0.01, bgy + cs / 2)], c=REP)
    seg(ax, [(rx + rwid, psy + cs / 2), (gx0 - pad - 0.01, psy + cs / 2)], c=SIZE)
    rbox(ax, 4.57, 1.98, 2.49, 0.36, fc="white", ec=INK, lw=0.8)
    ax.text(5.815, 2.25, "replay batch: half small defects, half background", ha="center", va="center",
            fontsize=SMALL, color=INK)
    ax.text(5.815, 2.08, "more background from images with more FPs", ha="center", va="center", fontsize=SMALL,
            color=INK)
    seg(ax, [((gx0 - pad + ps_x1) / 2, psy - pad), ((gx0 - pad + ps_x1) / 2, 2.35)], c=SIZE)
    seg(ax, [(bg_x1, bgy + cs / 2), (7.03, bgy + cs / 2), (7.03, 2.35)], c=REP)
    save(fig, out, "fig1_training")


def draw_inference(I, track, out):
    W, H = 7.16, 1.86
    fig, ax = canvas(W, 0, H)
    PX1 = 5.50
    panel(ax, 0.03, 0.03, PX1 - 0.03, H - 0.06, "Twin-Track SparSight Inference",
          "Global context guides where native detail is inspected and constrains how it is used.", "infer")
    im_ = I["img"]
    iH, iW = im_.shape[:2]
    zs, zy0 = 0.56, 0.62
    iw = ih = zs
    iy = zy0
    ix = 0.13
    ly = iy + ih / 2
    img(ax, im_, ix, iy, iw, ih)
    contour(ax, I["g"] > 0, ix, iy, iw, ih, c=GT, lw=0.5)
    ax.text(ix, iy + ih + 0.04, "input image", fontsize=LAB, va="bottom")
    gx0n = ix + iw + 0.20
    xe = net(ax, gx0n, ly, 0.5, NETC, label=False, stages=False)
    seg(ax, [(ix + iw, ly), (gx0n - 0.02, ly)], c=LOW)
    ax.text((gx0n + xe) / 2, iy - 0.05, "Global Sight\nTrack (L)", fontsize=SMALL, color=LOW, ha="center", va="top",
            linespacing=1.0)
    # global prediction: probability map, candidates (thin), example window (bold), binary support M (dashed)
    px, pw = xe + 0.18, iw
    k_ = 5
    Lb = I["L"][: I["L"].shape[0] // k_ * k_, : I["L"].shape[1] // k_ * k_]
    Lb = Lb.reshape(Lb.shape[0] // k_, k_, Lb.shape[1] // k_, k_).max((1, 3))
    imb = im_[: Lb.shape[0] * k_: k_, : Lb.shape[1] * k_: k_]
    img(ax, blend(imb, Lb), px, iy, pw, ih, ec=LOW, lw=0.8, interp="nearest")
    Mm = ndimage.binary_dilation(I["M"] > 0, iterations=4)  # display: thin support regions stay visible
    ys, xs = np.mgrid[0:Mm.shape[0], 0:Mm.shape[1]]
    ax.contour(px + (xs + 0.5) * pw / Mm.shape[1], iy + ih - (ys + 0.5) * ih / Mm.shape[0], Mm.astype(float),
               levels=[0.5], colors="#00a6e0", linewidths=0.9, linestyles="--", zorder=6)
    seg(ax, [(xe, ly), (px - 0.02, ly)], c=LOW)
    ax.text(px + pw / 2, iy + ih + 0.04, "$p_L$, support $M$", fontsize=LAB, color=LOW, ha="center", va="bottom")
    b = 384 * 900 / I["full_hw"][1] * pw / iW
    for cx_, cy_, _ in I["cand"]:
        clipped_box(ax, px + cx_ * pw / iW - b / 2, iy + ih - cy_ * pw / iW - b / 2, b, b, px, iy, pw, ih, ec=NAT,
                    lw=0.6, zorder=7)
    ax.text(px + pw / 2, iy - 0.04, "boxes: candidates", fontsize=SMALL, color=NAT, ha="center", va="top")
    zb = I["zoom_box"]
    zg = I["zoom_g"] > 0
    gx_, gy_ = centre(I["zoom_g"])
    Z = 128
    wy0, wx0 = (int(np.clip(round(v - Z / 2), 0, 384 - Z)) for v in (gy_, gx_))

    def crop_panel(a, x, ec, lab, col, y=zy0, sz=zs, below=False):
        img(ax, a[wy0:wy0 + Z, wx0:wx0 + Z], x, y, sz, sz, ec=ec, lw=1.0)
        contour(ax, zg[wy0:wy0 + Z, wx0:wx0 + Z], x, y, sz, sz, c=GT, lw=0.6 if sz > 0.4 else 0.3)
        if lab:
            ax.text(x + sz / 2, (y - 0.04) if below else (y + sz + 0.04), lab, ha="center",
                    va="top" if below else "bottom", fontsize=LAB, color=col)

    zx0 = px + pw + 0.30
    crop_panel(I["zoom_img"], zx0, NAT, "native crop", NAT)
    sc_ = zb[2] / 384
    wx_, wy_, ws_ = zb[0] + wx0 * sc_, zb[1] + wy0 * sc_, Z * sc_
    s_ = max(ws_ * pw / iW, 0.07)
    cxw, cyw = px + (wx_ + ws_ / 2) * pw / iW, iy + ih - (wy_ + ws_ / 2) * pw / iW
    ax.add_patch(Rectangle((cxw - s_ / 2, cyw - s_ / 2), s_, s_, fc="none", ec=NAT, lw=1.6, zorder=8))
    for yb_, yz_ in ((cyw + s_ / 2, zy0 + zs), (cyw - s_ / 2, zy0)):
        ax.plot([cxw + s_ / 2, zx0], [yb_, yz_], color=NAT, lw=0.7, zorder=8)
    n2 = zx0 + zs + 0.18
    xe2 = net(ax, n2, zy0 + zs / 2, 0.5, NETC, label=False, stages=False)
    seg(ax, [(zx0 + zs, zy0 + zs / 2), (n2 - 0.02, zy0 + zs / 2)], c=NAT)
    ax.text((n2 + xe2) / 2, iy - 0.05, "Native Sight\nTrack (S)", fontsize=SMALL, color=NAT, ha="center", va="top",
            linespacing=1.0)
    ax.text((n2 + xe2) / 2, iy + ih + 0.04, "same network", fontsize=SMALL, color=NETC, ha="center", va="bottom")
    cz = (zs - 0.04) / 2
    ox = xe2 + 0.18
    ctr = zy0 + zs / 2
    yt, yb = ctr + 0.02, ctr - 0.02 - cz
    crop_panel(heat(I["zoom_L"]), ox, LOW, "", LOW, y=yt, sz=cz)
    crop_panel(heat(I["zoom_S"]), ox, NAT, "", NAT, y=yb, sz=cz)
    ax.text(ox + cz + 0.03, yt + cz - 0.02, "$p_L$", fontsize=SMALL, color=LOW, va="top")
    ax.text(ox + cz + 0.03, yb + 0.02, "$p_S$", fontsize=SMALL, color=NAT, va="bottom")
    seg(ax, [(xe2, ctr), (ox - 0.06, ctr), (ox - 0.06, yb + cz / 2), (ox - 0.02, yb + cz / 2)], c=NAT)
    yr = iy + ih + 0.24
    rx_ = px + pw - 0.05
    seg(ax, [(rx_, iy + ih), (rx_, yr), (ox + cz / 2, yr), (ox + cz / 2, yt + cz + 0.01)], c=LOW)
    mb_x, mb_w, mb_h = ox + cz + 0.22, 0.46, 0.36
    rbox(ax, mb_x, ctr - mb_h / 2, mb_w, mb_h, fc="white", ec=INK, lw=0.9)
    ax.text(mb_x + mb_w / 2, ctr, "merge\nEq. (4)", ha="center", va="center", fontsize=SMALL, color=INK,
            linespacing=1.0)
    seg(ax, [(ox + cz + 0.16, yt + cz / 2), (mb_x - 0.01, yt + cz / 2)], c=LOW, head=False)
    seg(ax, [(ox + cz + 0.16, yb + cz / 2), (mb_x - 0.01, yb + cz / 2)], c=NAT, head=False)
    fx_ = mb_x + mb_w + 0.14
    seg(ax, [(mb_x + mb_w, ctr), (fx_ - 0.02, ctr)], c=INK)
    crop_panel(heat(I["zoom_P"]), fx_, INK, "final $p$", INK)
    # operating points
    my, mt = 0.11, 0.27
    t = 384 * 900 / I["full_hw"][1] * mt / iW
    img(ax, im_, ix, my, mt, mt)
    for cx_, cy_, _ in I["cand"]:
        clipped_box(ax, ix + cx_ * mt / iW - t / 2, my + mt - cy_ * mt / iW - t / 2, t, t, ix, my, mt, mt, ec=NAT,
                    lw=0.7, zorder=7)
    ax.text(ix + mt + 0.06, my + mt / 2, "precision mode: candidate crops, merged by Eq. (4)", fontsize=SMALL,
            color=NAT, va="center")
    rx2 = 2.92
    img(ax, im_, rx2, my, mt, mt)
    k = 1
    while k * t < mt:
        ax.plot([rx2 + k * t] * 2, [my, my + mt], color=NAT, lw=0.4, zorder=6)
        ax.plot([rx2, rx2 + mt], [my + mt - k * t] * 2, color=NAT, lw=0.4, zorder=6)
        k += 1
    ax.text(rx2 + mt + 0.06, my + mt / 2, "recall mode: all native tiles, max", fontsize=SMALL, color=NAT,
            va="center")

    # ------------------------------------------------------------------ track analysis
    qx, qw = PX1 + 0.06, W - PX1 - 0.09
    panel(ax, qx, 0.03, qw, H - 0.06, "Why global support?", "", "why")
    g, p = track["gt_components"]["small"], track["pred_components"]
    small = 100 * g["S_only"] / sum(g.values())
    s_only = p["S_only"]["true"] + p["S_only"]["false"]
    fp = 100 * p["S_only"]["false"] / s_only
    tin, tout = track["s_only_true"], track["s_only_false"]
    pin = 100 * tin["inside_M"] / (tin["inside_M"] + tout["inside_M"])
    pout = 100 * tin["outside_M"] / (tin["outside_M"] + tout["outside_M"])
    rows = [(f"{small:.0f}%", "of small defects found\nonly at native resolution"),
            (f"{fp:.0f}%", "of native-only predictions\nare false positives"),
            (f"{pin:.0f}% vs {pout:.0f}%", "native-only predictions correct\ninside vs. outside $M$")]
    y = H - 0.36
    for big, small_t in rows:
        ax.text(qx + 0.09, y, big, fontsize=11.5, weight="bold", color=NAT, va="top")
        ax.text(qx + 0.09, y - 0.20, small_t, fontsize=7.0, color=INK, va="top", linespacing=1.0)
        y -= 0.50
    save(fig, out, "fig2_inference")
    print(f"track: small S-only {small:.1f}%, native-only FP {fp:.1f}%, correct inside {pin:.1f}% / outside {pout:.1f}%")


def save(fig, out, name):
    out = resolve(out)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", dpi=300)
    print("saved", out / f"{name}.pdf")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/paper/figures")
    a = ap.parse_args()
    plt.rcParams.update({"font.size": 7.2})
    T = np.load(resolve(DATA) / "train.npz")
    R = np.load(resolve(DATA) / "replay.npz")
    I = np.load(resolve(DATA) / "infer.npz")
    draw_training(T, R, a.out)
    draw_inference(I, json.load(open(resolve(TRACK))), a.out)
