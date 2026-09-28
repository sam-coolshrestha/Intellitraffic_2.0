# Paper outline (IEEE style)

**Title:** Low-Cost Single-Camera Vehicle Speed Estimation, Plate Recognition and Overspeed Detection with YOLOv8, ByteTrack and EasyOCR

1. **Abstract** - problem, pipeline, headline numbers (speed MAE, plate exact-match, overspeed F1, FPS).
2. **Introduction** - motivation (road safety, cost of radar/ANPR), contributions.
3. **Related Work** - detection (YOLO family), tracking (ByteTrack, BoT-SORT), speed from monocular video, ANPR.
4. **Method** - pipeline diagram (README mermaid), calibration (simple vs homography), smoothing, plate voting, violation rule.
5. **Experiments** - datasets, ground truth collection, metrics, hardware. Reference `research/results/*.csv`.
6. **Results** - Table 1 detection (`detection_metrics.csv`); Table 2 tracking (`tracking_metrics.csv`); Table 3 speed (`speed_metrics.csv`), Fig. 1 Bland-Altman (`bland_altman_*.pdf`); Table 4 OCR (`ocr_metrics.csv`); Table 5 ablation (`ablation.csv`), Fig. 2 FPS vs image size (`ablation_fps.pdf`); Table 6 latency (`benchmark.csv`).
7. **Discussion** - what drives error (calibration, resolution, lighting), accuracy/speed trade-offs.
8. **Limitations** - single camera, road-plane assumption, occlusion, night, small ground truth.
9. **Future Work** - lane analysis, accident detection, multi-camera, learned plate detector.
10. **References**
