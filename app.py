"""IntelliTraffic AI - Streamlit dashboard.   Run:  streamlit run app.py"""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import pandas as pd
import plotly.express as px
import streamlit as st

from app import analytics as an
from app.config import ROOT, load_config
from app.detection import COCO_VEHICLES, VehicleDetector
from app.ocr import PlateReader
from app.pipeline import process_video

OUT_DIR = ROOT / "outputs"
VIDEO_DIR = ROOT / "videos"

st.set_page_config(page_title="IntelliTraffic AI", page_icon="🚦", layout="wide")


# ----------------------------------------------------------------- cached resources
@st.cache_resource(show_spinner="Loading YOLOv8 model (first run downloads weights)...")
def get_detector(weights, imgsz, conf, iou, device, classes, tracker):
    cfg = load_config(overrides={"model": {"weights": weights, "imgsz": imgsz, "conf": conf, "iou": iou,
                                            "device": device, "classes": list(classes), "tracker": tracker}})
    return VehicleDetector(cfg.model)


@st.cache_resource(show_spinner="Loading OCR model...")
def get_reader(languages, min_crop_px, min_conf, gpu):
    cfg = load_config(overrides={"ocr": {"languages": list(languages), "min_crop_px": min_crop_px,
                                          "min_confidence": min_conf}})
    return PlateReader(cfg.ocr, gpu=gpu)


def parse_points(text: str):
    """'x,y' per line -> [[x, y], ...]. Raises ValueError on bad input."""
    pts = []
    for line in text.strip().splitlines():
        if line.strip():
            x, y = (float(v) for v in line.replace(";", ",").split(",")[:2])
            pts.append([x, y])
    return pts


def load_existing() -> dict | None:
    rec_path = OUT_DIR / "vehicle_records.csv"
    if not rec_path.exists():
        return None
    rec = an.load_records(rec_path)

    def read(name):
        p = OUT_DIR / name
        return pd.read_csv(p) if p.exists() and p.stat().st_size > 0 else pd.DataFrame()

    video = OUT_DIR / "final_output.mp4"
    heat = OUT_DIR / "trajectory_heatmap.png"
    return {"records": rec, "violations": rec[rec["overspeed"]], "trajectories": read("trajectories.csv"),
            "speed_timeline": read("speed_timeline.csv"), "video": video if video.exists() else None,
            "heatmap": heat if heat.exists() else None, "metadata": {}, "output_dir": OUT_DIR}


# ----------------------------------------------------------------- sidebar
st.sidebar.title("IntelliTraffic AI")
uploaded = st.sidebar.file_uploader("Upload traffic video", type=["mp4", "avi", "mov", "mkv"])

with st.sidebar.expander("Detection", expanded=True):
    weights = st.selectbox("YOLOv8 model", ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"], help="n = fastest, m = most accurate")
    classes = st.multiselect("Vehicle classes", list(COCO_VEHICLES), default=list(COCO_VEHICLES))
    conf = st.slider("Confidence", 0.10, 0.90, 0.35, 0.05)
    imgsz = st.select_slider("Image size", [480, 640, 960], value=640)
    tracker = st.selectbox("Tracker", ["bytetrack.yaml", "botsort.yaml"])

with st.sidebar.expander("Speed & violations", expanded=True):
    limit = st.number_input("Speed limit (km/h)", 10, 200, 60)
    mode = st.radio("Calibration", ["simple", "homography"], horizontal=True)
    mpp, img_pts, world_pts = 0.05, "", ""
    if mode == "simple":
        mpp = st.number_input("Meters per pixel", 0.001, 5.0, 0.05, 0.005, format="%.3f",
                              help="Real length of the road divided by its length in pixels at the vehicle's position.")
    else:
        img_pts = st.text_area("4 image points (pixels, 'x,y' per line)", "420,300\n860,300\n1100,700\n180,700")
        world_pts = st.text_area("Same 4 points in meters ('x,y' per line)", "0,0\n7,0\n7,30\n0,30")

with st.sidebar.expander("License plates (OCR)"):
    ocr_on = st.checkbox("Enable OCR", True)
    ocr_every = st.slider("Read every N frames", 3, 30, 10)

max_frames = st.sidebar.number_input("Max frames (0 = all)", 0, 100000, 0,
                                     help="Limit frames to keep hosted demos fast.")
run = st.sidebar.button("▶ Run analysis", type="primary")

# ----------------------------------------------------------------- layout
st.title("IntelliTraffic AI")
st.caption("YOLOv8 + ByteTrack + EasyOCR + OpenCV traffic analytics")
tabs = st.tabs(["Live Processing", "Overview", "Violations", "Search", "Trajectories", "Video", "Reports"])
t_live, t_over, t_viol, t_search, t_traj, t_video, t_rep = tabs

if run:
    with t_live:
        if uploaded is None:
            st.warning("Upload a video in the sidebar first.")
        elif not classes:
            st.warning("Select at least one vehicle class.")
        else:
            try:
                VIDEO_DIR.mkdir(exist_ok=True)
                src = VIDEO_DIR / f"uploaded_video{Path(uploaded.name).suffix.lower()}"
                src.write_bytes(uploaded.getbuffer())
                overrides = {
                    "model": {"weights": f"models/{weights}", "imgsz": imgsz, "conf": conf, "classes": classes,
                              "tracker": tracker},
                    "speed": {"mode": mode, "meters_per_pixel": mpp, "limit_kmh": float(limit)},
                    "ocr": {"enabled": ocr_on, "every_n_frames": ocr_every},
                    "video": {"max_frames": int(max_frames) or None},
                }
                if mode == "homography":
                    overrides["speed"]["image_points"] = parse_points(img_pts)
                    overrides["speed"]["world_points"] = parse_points(world_pts)
                cfg = load_config(overrides=overrides)

                detector = get_detector(cfg.model.weights, imgsz, conf, cfg.model.iou, cfg.model.device,
                                        tuple(classes), tracker)
                reader = get_reader(tuple(cfg.ocr.languages), cfg.ocr.min_crop_px, cfg.ocr.min_confidence,
                                    detector.device.startswith("cuda")) if ocr_on else None

                bar, status, preview = st.progress(0.0), st.empty(), st.empty()
                t0 = time.time()

                def on_progress(i, total, frame):
                    pct = min(1.0, (i + 1) / total) if total and total > 0 else 0.0
                    bar.progress(pct)
                    el = time.time() - t0
                    eta = f"{el / max(pct, 1e-3) - el:.0f}s left" if pct > 0.02 else "estimating..."
                    status.write(f"Frame {i + 1}/{total if total > 0 else '?'} · {(i + 1) / max(el, 1e-3):.1f} fps · {eta}")
                    preview.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                res = process_video(src, cfg, OUT_DIR, detector=detector, ocr_reader=reader, progress_cb=on_progress)
                bar.progress(1.0)
                status.success(f"Done in {res['metadata']['processing_time_s']}s · "
                               f"{len(res['records'])} vehicles · {len(res['violations'])} violations")
                st.session_state["result"] = res
                st.session_state["limit"] = float(limit)
            except ValueError as exc:
                st.error(f"Could not process: {exc}")
            except Exception as exc:  # show a friendly message, keep the app alive
                st.error(f"Processing failed: {exc}")
                st.exception(exc)

result = st.session_state.get("result")
from_disk = False
if result is None:
    result = load_existing()
    from_disk = result is not None
limit_used = st.session_state.get("limit", float(limit))

if result is None:
    for t in (t_over, t_viol, t_search, t_traj, t_video, t_rep):
        with t:
            st.info("No results yet. Upload a video and click **Run analysis** in the sidebar.")
    with t_live:
        st.info("Upload a traffic video and click **Run analysis**. Progress and a live preview appear here.")
else:
    rec: pd.DataFrame = result["records"]
    with t_live:
        if from_disk and not run:
            st.info("Showing the most recent saved results from the outputs folder.")
        elif result.get("metadata"):
            m = result["metadata"]
            st.caption(f"Last run: {m['frames_processed']} frames in {m['processing_time_s']}s "
                       f"({m['processing_fps']} fps) on {m['device']}")

    with t_over:
        k = an.kpis(rec)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total vehicles", k["total"])
        c2.metric("Average speed", f"{k['avg_speed']:.1f} km/h")
        c3.metric("Max speed", f"{k['max_speed']:.1f} km/h")
        c4.metric("Violations", k["violations"])
        if rec.empty:
            st.warning("No vehicles were tracked long enough. Try a lower confidence or a longer clip.")
        else:
            a, b = st.columns(2)
            a.plotly_chart(px.pie(an.type_distribution(rec), names="class", values="count", title="Vehicle types"))
            b.plotly_chart(px.histogram(rec.dropna(subset=["avg_speed_kmh"]), x="avg_speed_kmh", nbins=20,
                                        title="Average speed distribution (km/h)"))
            tl = result.get("speed_timeline")
            if tl is not None and not tl.empty:
                ids = st.multiselect("Vehicles to plot (empty = fleet mean)", sorted(tl["vehicle_id"].unique()))
                if ids:
                    fig = px.line(tl[tl["vehicle_id"].isin(ids)], x="time_s", y="speed_kmh", color="vehicle_id",
                                  title="Speed over time")
                else:
                    mean = tl.assign(t=tl["time_s"].round()).groupby("t", as_index=False)["speed_kmh"].mean()
                    fig = px.line(mean, x="t", y="speed_kmh", title="Mean speed over time")
                fig.add_hline(y=limit_used, line_dash="dash", line_color="red", annotation_text="limit")
                st.plotly_chart(fig)

    with t_viol:
        v = rec[rec["overspeed"]]
        if v.empty:
            st.success("No overspeeding vehicles detected.")
        else:
            st.dataframe(v.drop(columns=["snapshot_path"]))
            cols = st.columns(4)
            for i, row in enumerate(v.itertuples()):
                p = Path(str(row.snapshot_path)) if row.snapshot_path else None
                if p and p.exists():
                    cols[i % 4].image(str(p), caption=f"#{row.vehicle_id} · {row.max_speed_kmh:.0f} km/h · {row.plate or 'no plate'}")

    with t_search:
        c1, c2 = st.columns([2, 1])
        q = c1.text_input("Search plate number (partial, case-insensitive)")
        cls = c2.multiselect("Vehicle type", sorted(rec["class"].dropna().unique()))
        hi = float(rec["max_speed_kmh"].max()) if rec["max_speed_kmh"].notna().any() else 100.0
        rng = st.slider("Max speed range (km/h)", 0.0, max(hi, 1.0), (0.0, max(hi, 1.0)))
        only = st.checkbox("Only violations")
        hits = an.filter_records(rec, q, cls, rng, only)
        st.write(f"{len(hits)} vehicle(s)")
        st.dataframe(hits.drop(columns=["snapshot_path"]))

    with t_traj:
        heat = result.get("heatmap")
        if heat and Path(heat).exists():
            st.image(str(heat), caption="Trajectory density heatmap")
        tr = result.get("trajectories")
        if tr is not None and not tr.empty:
            fig = px.line(tr, x="x", y="y", color="vehicle_id", line_group="vehicle_id", title="Vehicle trajectories")
            fig.update_yaxes(autorange="reversed", scaleanchor="x")
            st.plotly_chart(fig)
        else:
            st.info("No trajectories recorded.")

    with t_video:
        vp = result.get("video")
        if vp and Path(vp).exists():
            st.video(Path(vp).read_bytes())
        else:
            st.info("No processed video found.")

    with t_rep:
        st.download_button("⬇ vehicle_records.csv", rec.to_csv(index=False), "vehicle_records.csv", "text/csv")
        st.download_button("⬇ violations.csv", rec[rec["overspeed"]].to_csv(index=False), "violations.csv", "text/csv")
        tr = result.get("trajectories")
        if tr is not None and not tr.empty:
            st.download_button("⬇ trajectories.csv", tr.to_csv(index=False), "trajectories.csv", "text/csv")
        st.download_button("⬇ Summary report (HTML, print to PDF)", an.summary_html(rec, limit_used),
                           "intellitraffic_report.html", "text/html")
        vp = result.get("video")
        if vp and Path(vp).exists():
            st.download_button("⬇ Processed video (MP4)", Path(vp).read_bytes(), "final_output.mp4", "video/mp4")
