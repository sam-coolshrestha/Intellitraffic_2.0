# Research toolkit

**Research question:** How accurately can a low-cost, single-camera YOLOv8 + ByteTrack + EasyOCR pipeline
estimate vehicle speed, read plates and detect overspeeding, and how do model size, resolution, calibration
method and lighting affect performance?

All scripts run from the repo root (`python -m research.<script>`) and write CSVs + PNG/PDF figures to
`research/results/`. Every pipeline run also stores its full config, versions and timings in `run_metadata.json`.

| Table / figure | Command | Data you must supply |
|---|---|---|
| Detection P / R / mAP | `python -m research.evaluate_detection --data data/vehicles.yaml --weights yolov8n.pt yolov8s.pt --imgsz 480 640` | YOLO-format labelled dataset (e.g. UA-DETRAC converted) |
| Tracking MOTA / IDF1 / ID-switches | `python -m research.evaluate_tracking --video clip.mp4 --gt gt.txt` | MOT-format ground truth `frame,id,x,y,w,h,...` |
| Speed MAE / RMSE / MAPE + Bland-Altman | run `main.py`, then `python -m research.evaluate_speed --gt gt_speeds.csv --limit 60 --tag simple` | `gt_speeds.csv`: `vehicle_id,gt_speed_kmh` (GPS/radar/timed known distance) |
| Overspeed precision / recall / F1 | same command as above (also reports TP/FP/FN/TN) | same |
| Plate exact-match + CER by condition | `python -m research.evaluate_ocr --gt gt_plates.csv --tag preproc` | `gt_plates.csv`: `vehicle_id,gt_plate[,condition]` |
| Ablation (model × resolution × OCR) | `python -m research.ablation --video clip.mp4 --gt gt_speeds.csv --weights yolov8n.pt yolov8s.pt --imgsz 480 640 960 --ocr on off` | clip + optional speed ground truth |
| Latency / FPS / RAM, CPU vs GPU | `python -m research.benchmark --video clip.mp4 --devices cpu cuda:0` | any clip |

To compare calibration methods, run `main.py` twice with different configs (`--mpp` vs a homography YAML),
then run `evaluate_speed` with `--tag simple` and `--tag homography`; rows append to `speed_metrics.csv`.

## Threats to validity
Camera angle and calibration error dominate speed error; occlusion causes ID switches; night scenes, motion
blur and low plate resolution hurt OCR; a single camera cannot resolve depth without a road-plane assumption;
ground-truth sets are small, so report confidence intervals and avoid over-claiming. Random seeds are not
involved in inference, but record library versions (stored in run metadata).

The paper skeleton is in `paper_outline.md`.
