# Data-quality record 

Partitions covered: train (80 trips), student_test (10 trips). Evaluator trips not used.
All numbers below come from `results/eda/` (re-run `python scripts/eda.py`). Update after any re-run.

## Issues found
| # | Issue | Evidence file | Decision |
|---|---|---|---|
| 1 | Trips are in different cities worldwide (lon -100..137, lat -35..62), so one UTM zone (EPSG:32632) is wrong for most | `trip_centroids.png`, `trip_summary.csv` | Project per trip (local metres around the trip centroid, or per-trip UTM) |
| 2 | Not exactly 1 Hz: ~96.5% of train steps and ~98.6% of student_test steps are exactly 1 s | `sampling_intervals.csv` | Use real time gaps in features; never assume 1 s |
| 3 | Very large time gaps (train trips 00000069, 00000078, 00000090: up to ~153,000 s) | `trip_summary.csv` | Split HMM decoding at large gaps |
| 4 | Route breaks: `to_node(arc i) != from_node(arc i+1)` in some reference routes | `trip_summary.csv` (`route_breaks`) | Treat labels as reference, not perfect truth; alignment must tolerate breaks |
| 5 | Repeated arcs in some routes | `route_repeated_arcs` | Alignment must allow revisits; do not use set(route) |
| 6 | Network has many weakly connected components (median ~19 per trip) | `network_connectivity.png` | Candidates/transitions must handle unreachable arcs |
| 7 | Position jumps over 100 m between consecutive points in some trips | `jumps_gt100m` | Candidate radius sweep must cover these |
| 8 | Route is far shorter than the track in arcs (e.g. trip 45: 640 points, 59 arcs) | `trip_summary.csv` | Many points share one arc, so labels come from alignment |

## Checked and clean
No NaNs in tracks, no out-of-range coordinates, no non-increasing timestamps, no isolated nodes,
no out-of-range arc or route ids (see columns `nan_track`, `bad_coords`, `dt_nonmono`,
`isolated_nodes`, `route_ids_out_of_range`).

## Assumptions
- Longitude is column 1 and latitude column 2 in every file.
- `.route` ids are 0-based row indices into `.arcs`.
- Arc direction is from_node -> to_node (directed).
