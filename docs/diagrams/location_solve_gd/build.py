"""The GD location-solve diagram as an editable draw.io file, with each stage's plot.

    PYTHONPATH=<retina-geolocator> python docs/diagrams/location_solve_gd/build.py

Records one run (circle-vehicle, seed 101, receiver Doppler on), solves it with
GD and the receiver pose from the KF (step 3c), draws one small plot per stage
into ``plots/`` and writes ``location_solve_gd.drawio`` beside this file: boxes,
text and arrows as native draw.io shapes, each plot embedded as an image (also
kept in ``plots/`` so it can be swapped).  The delay-Doppler map is a
schematic (speckle around the measured detection), not a real cross-correlation.
"""

import base64
import html
import math
import os
import struct
import sys
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path[:0] = [REPO, os.path.join(REPO, "tools")]

import gnss_pose  # noqa: E402
import offline_solve as osv  # noqa: E402

from retina_simulation import experiment, gnss  # noqa: E402
from retina_simulation.motion import motion_from  # noqa: E402

CASE, SEED, REAL = "circle-vehicle", 101, 0
PLOTS = os.path.join(HERE, "plots")
TOWER = {0: ("tower A", "#2a78d6"), 1: ("tower B", "#e8622c"), 2: ("tower C", "#1baf7a")}
DPI = 200


def _fig(w=3.2, h=2.1):
    fig, ax = plt.subplots(figsize=(w, h), dpi=DPI)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=6, length=2)
    return fig, ax


def _save(fig, name):
    path = os.path.join(PLOTS, name)
    fig.tight_layout(pad=0.3)
    fig.savefig(path, dpi=DPI, transparent=True)
    plt.close(fig)
    return path


def _title(ax, text):
    ax.set_title(text, fontsize=6.5, color="#444")


def data():
    exp = experiment.load(os.path.join(REPO, "experiments", "receiver_gnss.json"))
    case = next(c for c in exp["cases"] if c["id"] == CASE)
    run = next(r for r in exp["runs"] if r["seed"] == SEED)
    d = tempfile.mkdtemp(prefix="locdiag-")
    experiment.record_run(exp, run, d, motion_from(case["motion"]), rx_doppler=True)
    poses = gnss_pose.poses_for(d, exp, case, SEED, REAL, "kf")
    rows = osv.solve_run(d, exp.get("solver"), "geometric_doppler", pose_estimate=poses)
    return exp, case, d, poses, rows


def plot_gnss(exp, case):
    model = gnss.model_from(exp["gnss"])
    motion = motion_from(case["motion"])
    fixes = gnss.fixes(model, motion, exp["duration_s"], gnss_pose.gnss_key(SEED), REAL)
    err = np.array([(f["east_m"] - motion.state(f["t_s"])[0], f["north_m"] - motion.state(f["t_s"])[1]) for f in fixes])
    k = 15
    drift = np.array([err[max(0, i - k) : i + k + 1].mean(axis=0) for i in range(len(err))])
    fig, ax = _fig(2.6, 2.1)
    ax.scatter(err[:, 0], err[:, 1], s=2, color="#e8a33c", alpha=0.8, label="GNSS fixes")
    ax.plot(drift[:, 0], drift[:, 1], color="#c27a12", linewidth=0.9, label="slow drift")
    ax.plot([0], [0], "k+", markersize=8, markeredgewidth=1.4, label="true position")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("east", fontsize=6.5)
    ax.set_ylabel("north", fontsize=6.5)
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(fontsize=5.5, frameon=False, loc="upper right", bbox_to_anchor=(1.35, 1.0))
    _title(ax, "GNSS error over time")
    return _save(fig, "1_gnss.png")


def plot_kf(exp, case):
    motion = motion_from(case["motion"])
    ests = gnss_pose.estimates(exp, case, SEED, REAL, "kf")
    t = np.array([e["t_s"] for e in ests if e and 40 <= e["t_s"] <= 70])
    v = np.array([e["vel_east_ms"] for e in ests if e and 40 <= e["t_s"] <= 70])
    sd = np.array([e["sd_vel_east_ms"] for e in ests if e and 40 <= e["t_s"] <= 70])
    tt = np.linspace(40, 70, 200)
    fig, ax = _fig(3.0, 2.0)
    ax.fill_between(t, v - 2 * sd, v + 2 * sd, color="#2a78d6", alpha=0.2, linewidth=0, label="KF uncertainty (±2σ)")
    ax.plot(t, v, color="#2a78d6", linewidth=1.1, label="KF estimate")
    ax.plot(tt, [motion.state(x)[2] for x in tt], color="k", linewidth=1.1, label="truth")
    ax.set_xlabel("time (s)", fontsize=6.5)
    ax.set_ylabel("east velocity (m/s)", fontsize=6.5)
    ax.legend(fontsize=5.5, frameon=False, loc="lower left")
    _title(ax, "receiver velocity (zoomed)")
    return _save(fig, "2_kf.png")


def solve_tick(d, poses, rows):
    """The tick shown in the middle stages: a published solve about 30-40 m from the truth."""
    ok = [r for r in rows if r.get("published") and r.get("t_s", 0) >= 60]
    row = min(ok, key=lambda r: abs(r["err_horiz_m"] - 35.0))
    t_ms = row["t_ms"]
    with open(os.path.join(d, "nodes.json"), encoding="utf-8") as f:
        import json

        nodes = json.load(f)
    moved = osv._at_pose(nodes, poses[t_ms])
    geo = osv.Geometry(moved)
    meas = []
    for fr in osv._read_ndjson(os.path.join(d, "frames.ndjson")):
        if fr["t_ms"] == t_ms:
            for det in fr["detections"]:
                if det.get("adsb") or True:
                    meas.append({"node_id": fr["node_id"], "delay_us": det["delay_us"], "doppler_hz": det["doppler_hz"],
                                 "object_id": det["object_id"]})  # fmt: skip
    truth = next(t for t in osv._read_ndjson(os.path.join(d, "truth.ndjson")) if t["t_ms"] == t_ms)["aircraft"][0]
    meas = [m for m in meas if m["object_id"] == truth["id"]]
    ids = sorted(n["node_id"] for n in nodes)
    meas.sort(key=lambda m: ids.index(m["node_id"]))
    guess = geo.initial_guess(meas, osv.DEFAULT_SOLVER["alt_layers_km"])
    corr = {nid: row.get(f"rxdop_corr_hz_{osv._short(nid)}", 0.0) for nid in ids}
    return row, geo, meas, truth, guess, corr, ids


def _delay_field(geo, nid, e, n, alt):
    rx, tx, base, _ = geo.nodes[nid]
    d_tx = np.sqrt((e - tx[0]) ** 2 + (n - tx[1]) ** 2 + (alt - tx[2]) ** 2)
    d_rx = np.sqrt((e - rx[0]) ** 2 + (n - rx[1]) ** 2 + (alt - rx[2]) ** 2)
    return (d_tx + d_rx - base) / osv._C_KM_US


def plot_propose(geo, meas, guess, ids):
    ge, gn, _ = geo.enu(guess["lat"], guess["lon"], guess["alt_km"])
    axis = np.linspace(-45, 45, 361)
    e, n = np.meshgrid(axis, axis)
    cost = sum((_delay_field(geo, m["node_id"], e, n, guess["alt_km"]) - m["delay_us"]) ** 2 for m in meas)
    fig, ax = _fig(2.6, 2.3)
    ax.imshow(np.log10(cost + 1), extent=(-45, 45, -45, 45), origin="lower", cmap="Blues_r", alpha=0.55)
    for i, m in enumerate(meas):
        ax.contour(e, n, _delay_field(geo, m["node_id"], e, n, guess["alt_km"]), levels=[m["delay_us"]],
                   colors=[TOWER[ids.index(m["node_id"])][1]], linewidths=0.9)  # fmt: skip
    rx = geo.nodes[ids[0]][0]
    ax.plot(rx[0], rx[1], "ks", markersize=3)
    ax.annotate("RX", (rx[0], rx[1]), xytext=(3, 2), textcoords="offset points", fontsize=5.5)
    ax.plot(ge, gn, "o", color="#2a78d6", markersize=4, markeredgecolor="white")
    ax.annotate("first guess\n(x, y, alt layer, v = 0)", (ge, gn), xytext=(-25, -22), textcoords="offset points",
                fontsize=5.5, color="#2a3fd6", arrowprops={"arrowstyle": "-", "color": "#2a3fd6", "lw": 0.6})  # fmt: skip
    ax.set_xlim(-45, 45)
    ax.set_ylim(-45, 45)
    ax.set_xlabel("east (km)", fontsize=6.5)
    ax.set_ylabel("north (km)", fontsize=6.5)
    return _save(fig, "3_propose.png"), (ge, gn)


def _dd_axes(ax):
    ax.axhline(0, color="#ccc", linewidth=0.6, zorder=0)
    ax.set_xlim(0, 200)
    ax.set_ylim(-300, 300)
    ax.set_xlabel("delay (µs)", fontsize=6.5)
    ax.set_ylabel("Doppler (Hz)", fontsize=6.5)


def plot_convert(geo, meas, guess, ids):
    ge, gn, _ = geo.enu(guess["lat"], guess["lon"], guess["alt_km"])
    pred = {m["node_id"]: float(_delay_field(geo, m["node_id"], ge, gn, guess["alt_km"])) for m in meas}
    fig, ax = _fig(2.8, 2.1)
    _dd_axes(ax)
    for nid, dl in pred.items():
        name, c = TOWER[ids.index(nid)]
        ax.plot(dl, 0, "o", markerfacecolor="white", markeredgecolor=c, markersize=5, markeredgewidth=1.2)
        up = sorted(pred.values()).index(dl) % 2 == 0
        ax.annotate(name, (dl, 0), xytext=(0, 7 if up else -11), textcoords="offset points", fontsize=6, color=c,
                    fontweight="bold", ha="center")  # fmt: skip
    _title(ax, "predicted from the proposed state")
    return _save(fig, "4_convert.png"), pred


def plot_lm(meas, pred, corr, ids):
    fig, ax = _fig(2.8, 2.1)
    _dd_axes(ax)
    for m in meas:
        _, c = TOWER[ids.index(m["node_id"])]
        x0, x1, y1 = pred[m["node_id"]], m["delay_us"], m["doppler_hz"] - corr[m["node_id"]]
        ax.annotate("", (x1, y1), (x0, 0), arrowprops={"arrowstyle": "-|>", "color": c, "lw": 0.8, "shrinkA": 3,
                                                        "shrinkB": 3, "mutation_scale": 6})  # fmt: skip
        ax.plot(x0, 0, "o", markerfacecolor="white", markeredgecolor=c, markersize=5, markeredgewidth=1.2)
        ax.plot(x1, y1, "o", color=c, markersize=5)
    ax.plot([], [], "o", markerfacecolor="white", markeredgecolor="#555", markersize=5, label="predicted (first guess)")
    ax.plot([], [], "o", color="#555", markersize=5, label="measured")
    ax.legend(fontsize=5.5, frameon=False, loc="upper right")
    _title(ax, "each step moves the predictions onto the measurements")
    return _save(fig, "5_lm.png")


def plot_detected(geo, meas, row, truth, ids):
    ee, en, _ = geo.enu(row["est_lat"], row["est_lon"], row["est_alt_m"] / 1000)
    te, tn, _ = geo.enu(truth["lat"], truth["lon"], truth["alt_m"] / 1000)
    ax_e = np.linspace(te - 2, te + 2, 201)
    ax_n = np.linspace(tn - 2, tn + 2, 201)
    e, n = np.meshgrid(ax_e, ax_n)
    fig, ax = _fig(2.4, 2.3)
    for m in meas:
        ax.contour(e, n, _delay_field(geo, m["node_id"], e, n, row["est_alt_m"] / 1000), levels=[m["delay_us"]],
                   colors=[TOWER[ids.index(m["node_id"])][1]], linewidths=0.9)  # fmt: skip
    ax.plot(te, tn, "kx", markersize=6, markeredgewidth=1.5, label="ADS-B truth")
    ax.plot(ee, en, "o", color="#8a2be2", markersize=5, label="solve")
    ve, vn = row["est_vel_east_ms"], row["est_vel_north_ms"]
    sp = math.hypot(ve, vn)
    s = 0.8 / sp
    ax.annotate("", (ee + ve * s, en + vn * s), (ee, en),
                arrowprops={"arrowstyle": "-|>", "color": "#8a2be2", "lw": 1, "mutation_scale": 7})  # fmt: skip
    ax.annotate(f"{sp:.0f} m/s", (ee + ve * s, en + vn * s), xytext=(3, -2), textcoords="offset points", fontsize=5.5,
                color="#8a2be2")  # fmt: skip
    ax.set_xlim(te - 2, te + 2)
    ax.set_ylim(tn - 2, tn + 2)
    ax.set_aspect("equal")
    ax.set_xlabel("east (km)", fontsize=6.5)
    ax.set_ylabel("north (km)", fontsize=6.5)
    ax.legend(fontsize=5.5, frameon=False, loc="lower right")
    _title(ax, f"{row['err_horiz_m']:.0f} m from the truth (zoomed, ±2 km)")
    return _save(fig, "6_detected.png")


def _cv_track(t, z, q=0.5, r=60.0):
    x = np.array([z[0][0], z[0][1], 0.0, 0.0])
    P = np.diag([r * r, r * r, 400.0, 400.0])
    out = [x[:2].copy()]
    H = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        G = np.array([[dt * dt / 2, 0], [0, dt * dt / 2], [dt, 0], [0, dt]])
        x = F @ x
        P = F @ P @ F.T + q * q * G @ G.T
        S = H @ P @ H.T + r * r * np.eye(2)
        K = P @ H.T @ np.linalg.inv(S)
        x = x + K @ (np.array(z[i]) - H @ x)
        P = (np.eye(4) - K @ H) @ P
        out.append(x[:2].copy())
    return np.array(out)


def plot_tracker(geo, rows):
    pub = [r for r in rows if r.get("published")]
    tr = np.array([geo.enu(r["true_lat"], r["true_lon"], 0)[:2] for r in rows if r.get("true_lat") is not None])
    det = [geo.enu(r["est_lat"], r["est_lon"], 0)[:2] for r in pub]
    trk = _cv_track([r["t_s"] for r in pub], [(e * 1000, n * 1000) for e, n in det]) / 1000
    det = np.array(det)
    fig, ax = _fig(2.4, 2.3)
    ax.plot(tr[:, 0], tr[:, 1], color="#888", linewidth=1.0, label="ADS-B path")
    ax.plot(det[:, 0], det[:, 1], ".", color="#e05aa0", markersize=1.5, label="detected objects")
    ax.plot(trk[:, 0], trk[:, 1], color="#d62728", linewidth=1.1, label="track")
    pad = 5
    ax.set_xlim(det[:, 0].min() - pad * 2, det[:, 0].max() + pad * 2)
    ax.set_ylim(det[:, 1].min() - pad, det[:, 1].max() + pad)
    ax.set_xlabel("east (km)", fontsize=6.5)
    ax.set_ylabel("north (km)", fontsize=6.5)
    ax.legend(fontsize=5.5, frameon=False, loc="upper right")
    _title(ax, f"{(pub[-1]['t_s'] - pub[0]['t_s']) / 60:.0f} minutes: one track id")
    return _save(fig, "7_tracker.png")


def plot_adsb(rows):
    pub = [r for r in rows if r.get("published") and r.get("err_horiz_m") is not None]
    t = [r["t_s"] for r in pub]
    err = np.array([r["err_horiz_m"] for r in pub])
    rmse = math.sqrt(float(np.mean(err**2)))
    fig, ax = _fig(3.4, 1.4)
    ax.plot(t, err, color="#1f8f2e", linewidth=0.7)
    ax.axhline(rmse, color="#c0392b", linewidth=0.8, linestyle=(0, (4, 2)))
    ax.text(8, ax.get_ylim()[1] * 0.98, f"radar-only position error, RMSE {rmse:.0f} m", fontsize=5.5, va="top")
    ax.set_xlabel("time (s)", fontsize=6.5)
    ax.set_ylabel("error (m)", fontsize=6.5)
    return _save(fig, "8_adsb.png")


def plot_dd(meas, corr, ids):
    m = meas[0]
    rng = np.random.default_rng(3)
    dl, dp = m["delay_us"], m["doppler_hz"]
    x = np.linspace(dl - 15, dl + 15, 240)
    y = np.linspace(dp - 50, dp + 50, 160)
    X, Y = np.meshgrid(x, y)
    from scipy.ndimage import gaussian_filter

    z = gaussian_filter(rng.gamma(3, 1 / 3, X.shape), 2.0)
    z = z + 1.6 * np.exp(-(((X - dl) / 0.6) ** 2 + ((Y - dp) / 3.0) ** 2))
    fig, ax = _fig(3.0, 1.9)
    ax.imshow(z, extent=(x[0], x[-1], y[0], y[-1]), origin="lower", aspect="auto", cmap="viridis")
    ax.annotate("aircraft", (dl, dp), xytext=(6, 6), textcoords="offset points", fontsize=6, color="white",
                fontweight="bold", arrowprops={"arrowstyle": "-", "color": "white", "lw": 0.6})  # fmt: skip
    ax.set_xlabel("delay (µs)", fontsize=6.5)
    ax.set_ylabel("Doppler (Hz)", fontsize=6.5)
    _title(ax, f"{TOWER[ids.index(m['node_id'])][0]} (zoomed)")
    return _save(fig, "0_delay_doppler.png")


# ---------------------------------------------------------------- draw.io

BLUE, RED, PURPLE, GREEN = "#2020e0", "#e02020", "#8a2be2", "#1f9f2e"
S = 2.0  # draw.io units per pixel of the original 1063 x 511 picture


class Doc:
    def __init__(self):
        self.cells = []
        self.n = 2

    def _id(self):
        self.n += 1
        return f"c{self.n}"

    def vertex(self, x, y, w, h, style, value=""):
        i = self._id()
        self.cells.append(
            f'<mxCell id="{i}" value="{html.escape(value)}" style="{style}" vertex="1" parent="1">'
            f'<mxGeometry x="{x * S:.0f}" y="{y * S:.0f}" width="{w * S:.0f}" height="{h * S:.0f}" as="geometry"/></mxCell>'
        )
        return i

    def image(self, path, x, y, w, max_h=None):
        """An embedded PNG at (x, y), ``w`` wide (narrower if needed to stay within ``max_h``), centred on x + w/2."""
        with open(path, "rb") as f:
            raw = f.read()
        pw, ph = struct.unpack(">II", raw[16:24])
        h = w * ph / pw
        if max_h is not None and h > max_h:
            x, w, h = x + (w - max_h * pw / ph) / 2, max_h * pw / ph, max_h
        b64 = base64.b64encode(raw).decode()
        style = f"shape=image;verticalLabelPosition=bottom;verticalAlign=top;imageAspect=0;aspect=fixed;image=data:image/png,{b64};"
        return self.vertex(x, y, w, h, style)

    def edge(self, src, dst, style, points=(), value="", exit=None, entry=None):
        i = self._id()
        st = style
        if exit:
            st += f"exitX={exit[0]};exitY={exit[1]};exitDx=0;exitDy=0;"
        if entry:
            st += f"entryX={entry[0]};entryY={entry[1]};entryDx=0;entryDy=0;"
        pts = "".join(f'<mxPoint x="{px * S:.0f}" y="{py * S:.0f}"/>' for px, py in points)
        arr = f'<Array as="points">{pts}</Array>' if points else ""
        self.cells.append(
            f'<mxCell id="{i}" value="{html.escape(value)}" style="{st}" edge="1" parent="1" source="{src}" target="{dst}">'
            f'<mxGeometry relative="1" as="geometry">{arr}</mxGeometry></mxCell>'
        )
        return i

    def xml(self):
        body = "".join(self.cells)
        return (
            '<mxfile host="app.diagrams.net"><diagram name="GD location solve" id="locsolve">'
            f'<mxGraphModel dx="1400" dy="900" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" '
            f'fold="1" page="1" pageScale="1" pageWidth="{1080 * S:.0f}" pageHeight="{520 * S:.0f}" math="0" shadow="0">'
            f'<root><mxCell id="0"/><mxCell id="1" parent="0"/>{body}</root></mxGraphModel></diagram></mxfile>'
        )


def _box_style(colour, dashed=False):
    return (
        f"rounded=1;arcSize=6;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor={colour};strokeWidth=2.5;"
        f"verticalAlign=top;align=center;spacingTop=6;fontFamily=Helvetica;{'dashed=1;dashPattern=6 4;' if dashed else ''}"
    )


def _label(title, sub, colour):
    sub_html = "<br>".join(html.escape(s) for s in sub)
    return (
        f'<b><font style="font-size:15px" color="{colour}">{html.escape(title)}</font></b><br>'
        f'<font style="font-size:12px" color="{colour}">{sub_html}</font>'
    )


def build_drawio(plots, path):
    doc = Doc()
    doc.vertex(
        3,
        3,
        1057,
        505,
        "rounded=1;arcSize=1;whiteSpace=wrap;html=1;fillColor=none;strokeColor=#2020e0;strokeWidth=1.5;",
    )

    def box(x, y, w, h, colour, title, sub, img, img_w, dashed=False, img_dy=40):
        b = doc.vertex(x, y, w, h, _box_style(colour, dashed), _label(title, sub, colour))
        if img:
            doc.image(img, x + (w - img_w) / 2, y + img_dy, img_w, max_h=h - img_dy - 5)
        return b

    # delay-Doppler stack: two frames behind the front one, and "x3"
    for k in (2, 1):
        doc.vertex(418 + 6 * k, 13 + 4 * k, 219, 138, _box_style(BLUE))
    gnss_b = box(8, 13, 170, 139, BLUE, "GNSS Receiver",
                 ["Receiver position (x, y, z) fixes;", "the error drifts slowly around the truth"],
                 plots["gnss"], 130, img_dy=44)  # fmt: skip
    kf_b = box(228, 13, 170, 139, RED, "Receiver Kalman Filter",
               ["Estimates receiver position + velocity", "(state: position, velocity, GNSS drift)"],
               plots["kf"], 150, img_dy=44)  # fmt: skip
    dd_b = box(418, 13, 219, 138, BLUE, "delay-Doppler Data",
               ["Cross-correlation between surveillance", "signal and reference signal"], plots["dd"], 185, img_dy=44)  # fmt: skip
    doc.vertex(620, 138, 18, 12, "text;html=1;align=center;verticalAlign=middle;fontStyle=1;fontSize=14;", "x3")
    prop_b = box(8, 183, 170, 174, BLUE, "Propose Target State",
                 ["position, velocity", "(start: grid search over the 3 ellipses)"], plots["propose"], 135, img_dy=50)  # fmt: skip
    conv_b = box(228, 183, 170, 174, BLUE, "Convert to delay-Doppler",
                 ["Project state using the receiver's position;", "remove the receiver's own Doppler (GD)"],
                 plots["convert"], 150, img_dy=48)  # fmt: skip
    lm_b = box(448, 183, 170, 174, RED, "Levenberg-Marquardt Solver",
               ["Least squares fit of delay-Doppler", "residuals (predicted vs measured)"], plots["lm"], 150, img_dy=48)  # fmt: skip
    det_b = box(668, 183, 170, 174, PURPLE, "Detected Object",
                ["Position/velocity maximizing likelihood", "of delay-Doppler data"], plots["detected"], 125, img_dy=46)  # fmt: skip
    trk_b = box(888, 183, 168, 174, RED, "Kalman Tracker",
                ["Differentiates detected targets", "through temporal tracking"], plots["tracker"], 125, img_dy=46)  # fmt: skip
    adsb_b = box(645, 387, 213, 111, GREEN, "ADS-B", ["ground truth comparison"], plots["adsb"], 180, dashed=True,
                 img_dy=40)  # fmt: skip

    arrow = (
        "edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;endArrow=block;endFill=1;strokeWidth=1.5;strokeColor=#222222;"
    )
    doc.edge(gnss_b, kf_b, arrow, exit=(1, 0.5), entry=(0, 0.5))
    doc.edge(kf_b, conv_b, arrow, exit=(0.5, 1), entry=(0.5, 0))
    doc.vertex(318, 156, 70, 22, "text;html=1;align=left;verticalAlign=middle;fontStyle=2;fontColor=#e02020;fontSize=11;whiteSpace=wrap;",
               "receiver position + velocity")  # fmt: skip
    doc.edge(dd_b, lm_b, arrow, exit=(0.5, 1), entry=(0.5, 0))
    doc.edge(prop_b, conv_b, arrow, exit=(1, 0.5), entry=(0, 0.5))
    doc.edge(conv_b, lm_b, arrow, exit=(1, 0.5), entry=(0, 0.5))
    doc.edge(lm_b, det_b, arrow, exit=(1, 0.5), entry=(0, 0.5))
    doc.edge(det_b, trk_b, arrow, exit=(1, 0.5), entry=(0, 0.5))
    doc.edge(trk_b, det_b, arrow, points=[(972, 168), (752, 168)], exit=(0.5, 0), entry=(0.5, 0))
    doc.edge(det_b, adsb_b, arrow, exit=(0.5, 1), entry=(0.5, 0))
    dashed = ("edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;endArrow=block;endFill=1;strokeWidth=1.2;"
              "strokeColor=#e02020;dashed=1;dashPattern=4 3;")  # fmt: skip
    doc.edge(lm_b, prop_b, dashed, points=[(533, 377), (93, 377)], exit=(0.5, 1), entry=(0.5, 1))
    doc.vertex(200, 379, 260, 14, "text;html=1;align=center;verticalAlign=middle;fontStyle=2;fontColor=#e02020;fontSize=11;",
               "update the state and repeat until the residuals stop shrinking")  # fmt: skip
    with open(path, "w", encoding="utf-8") as f:
        f.write(doc.xml())


def main():
    os.makedirs(PLOTS, exist_ok=True)
    exp, case, d, poses, rows = data()
    row, geo, meas, truth, guess, corr, ids = solve_tick(d, poses, rows)
    print(f"tick t = {row['t_s']} s, error {row['err_horiz_m']} m")
    plots = {"gnss": plot_gnss(exp, case), "kf": plot_kf(exp, case), "dd": plot_dd(meas, corr, ids)}
    plots["propose"], _ = plot_propose(geo, meas, guess, ids)
    plots["convert"], pred = plot_convert(geo, meas, guess, ids)
    plots["lm"] = plot_lm(meas, pred, corr, ids)
    plots["detected"] = plot_detected(geo, meas, row, truth, ids)
    plots["tracker"] = plot_tracker(geo, rows)
    plots["adsb"] = plot_adsb(rows)
    out = os.path.join(HERE, "location_solve_gd.drawio")
    build_drawio(plots, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
