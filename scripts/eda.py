"""Phase 1 EDA (Harsimran). Run from the repo root:
    python scripts/eda.py
Reads data/raw/{train,student_test}; writes CSVs and PNGs to results/eda/.
Every table/plot is labelled with its partition.
"""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pathsnap.eda_io import iter_trips, to_local_metres

OUT = Path("results/eda"); OUT.mkdir(parents=True, exist_ok=True)


def components(n, arcs, strong):
    g = coo_matrix((np.ones(len(arcs)), (arcs[:, 0], arcs[:, 1])), shape=(n, n))
    k, lab = connected_components(g, directed=True, connection="strong" if strong else "weak")
    return k, lab


def analyse(t):
    tr, nodes, arcs, route = t.track, t.nodes, t.arcs, t.route
    lon, lat, ts = tr[:, 0], tr[:, 1], tr[:, 2]
    dt = np.diff(ts)
    x, y = to_local_metres(lon, lat, lon.mean(), lat.mean())
    step = np.hypot(np.diff(x), np.diff(y))
    speed = step / np.where(dt > 0, dt, np.nan)

    row = dict(partition=t.partition, trip_id=t.trip_id,
               n_points=len(tr), n_nodes=len(nodes), n_arcs=len(arcs), route_len=len(route))
    # --- data quality: missing / invalid
    row["nan_track"] = int(np.isnan(tr).sum())
    row["bad_coords"] = int(((np.abs(lon) > 180) | (np.abs(lat) > 90)).sum())
    row["dup_points"] = int((step == 0).sum())
    # --- time
    row["duration_s"] = ts[-1] - ts[0]
    row["dt_nonmono"] = int((dt <= 0).sum())
    row["dt_eq1_pct"] = 100 * float((dt == 1).mean())
    row["gaps_gt5s"] = int((dt > 5).sum())
    row["gaps_gt30s"] = int((dt > 30).sum())
    row["max_gap_s"] = float(dt.max())
    row["max_speed_mps"] = float(np.nanmax(speed))
    row["jumps_gt100m"] = int((step > 100).sum())
    row["track_extent_km"] = float(max(np.ptp(x), np.ptp(y)) / 1000)
    row["centroid_lon"], row["centroid_lat"] = float(lon.mean()), float(lat.mean())
    # --- network
    n = len(nodes)
    row["arc_node_out_of_range"] = int(((arcs < 0) | (arcs >= n)).sum())
    row["nodes_nan"] = int(np.isnan(nodes).sum())
    deg = np.bincount(arcs.ravel(), minlength=n)
    row["isolated_nodes"] = int((deg == 0).sum())
    row["self_loops"] = int((arcs[:, 0] == arcs[:, 1]).sum())
    row["weak_components"] = int(components(n, arcs, False)[0])
    row["strong_components"] = int(components(n, arcs, True)[0])
    # --- route
    bad_ids = int(((route < 0) | (route >= len(arcs))).sum())
    row["route_ids_out_of_range"] = bad_ids
    if bad_ids == 0 and len(route) > 1:
        a = arcs[route]
        row["route_breaks"] = int((a[:-1, 1] != a[1:, 0]).sum())  # to_node(i) != from_node(i+1)
    else:
        row["route_breaks"] = -1
    row["route_repeated_arcs"] = int(len(route) - len(set(route.tolist())))
    return row, dt


def main():
    rows, all_dt = [], {"train": [], "student_test": []}
    for t in iter_trips():
        r, dt = analyse(t)
        rows.append(r); all_dt[t.partition].append(dt)
        print(t.partition, t.trip_id, r["n_points"], r["route_breaks"])
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "trip_summary.csv", index=False)

    # inventory per partition
    inv = df.groupby("partition").agg(
        trips=("trip_id", "count"),
        points_total=("n_points", "sum"), points_min=("n_points", "min"),
        points_median=("n_points", "median"), points_max=("n_points", "max"),
        nodes_median=("n_nodes", "median"), nodes_max=("n_nodes", "max"),
        arcs_median=("n_arcs", "median"), arcs_max=("n_arcs", "max"),
        route_len_median=("route_len", "median"), route_len_max=("route_len", "max"))
    inv.to_csv(OUT / "inventory_by_partition.csv")

    # sampling interval distribution
    srows = []
    for p, lst in all_dt.items():
        d = np.concatenate(lst)
        srows.append(dict(partition=p, steps=len(d), pct_exactly_1s=100 * (d == 1).mean(),
                          pct_gt_1s=100 * (d > 1).mean(), pct_gt_5s=100 * (d > 5).mean(),
                          pct_gt_30s=100 * (d > 30).mean(), pct_nonpositive=100 * (d <= 0).mean(),
                          max_gap_s=d.max()))
    pd.DataFrame(srows).to_csv(OUT / "sampling_intervals.csv", index=False)

    # flag anomalies for docs/data_quality.md
    flags = df[(df.gaps_gt30s > 0) | (df.route_breaks != 0) | (df.nan_track > 0) | (df.bad_coords > 0)
               | (df.dt_nonmono > 0) | (df.jumps_gt100m > 0) | (df.weak_components > 1)
               | (df.route_ids_out_of_range > 0)]
    flags.to_csv(OUT / "flagged_trips.csv", index=False)
    json.dump({"n_flagged": int(len(flags)), "created": time.strftime("%Y-%m-%d %H:%M")},
              open(OUT / "run_info.json", "w"))

    # ---------------- plots ----------------
    col = {"train": "tab:blue", "student_test": "tab:orange"}
    # 1. interval histogram (log y)
    fig, ax = plt.subplots(figsize=(7, 4))
    for p, lst in all_dt.items():
        d = np.concatenate(lst); d = d[(d > 0) & (d <= 30)]
        ax.hist(d, bins=np.arange(0.5, 31.5, 1), alpha=.6, label=p, color=col[p])
    ax.set_yscale("log"); ax.set_xlabel("time between points (s), clipped at 30")
    ax.set_ylabel("count (log)"); ax.set_title("GPS sampling interval (train + student_test)"); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "sampling_interval_hist.png", dpi=150); plt.close(fig)

    # 2. points per trip + network size
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    for p in col:
        s = df[df.partition == p]
        axs[0].hist(s.n_points, bins=20, alpha=.6, label=p, color=col[p])
        axs[1].scatter(s.n_arcs, s.route_len, s=14, alpha=.7, label=p, color=col[p])
        axs[2].scatter(s.n_points, s.track_extent_km, s=14, alpha=.7, label=p, color=col[p])
    axs[0].set(xlabel="GPS points per trip", ylabel="trips", title="Trip length")
    axs[1].set(xscale="log", yscale="log", xlabel="arcs in network", ylabel="arcs in reference route", title="Network size vs route")
    axs[2].set(xlabel="GPS points", ylabel="track extent (km)", title="Track extent")
    axs[0].legend(); fig.tight_layout(); fig.savefig(OUT / "trip_size_overview.png", dpi=150); plt.close(fig)

    # 3. where trips are in the world
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for p in col:
        s = df[df.partition == p]
        ax.scatter(s.centroid_lon, s.centroid_lat, s=18, label=p, color=col[p], alpha=.8)
    ax.set(xlabel="longitude", ylabel="latitude", title="Trip centroids (train + student_test)")
    ax.grid(alpha=.3); ax.legend(); fig.tight_layout(); fig.savefig(OUT / "trip_centroids.png", dpi=150); plt.close(fig)

    # 4. network components + route breaks
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    axs[0].hist(df.weak_components, bins=30, color="tab:green")
    axs[0].set(xlabel="weakly connected components per trip", ylabel="trips", title="Network connectivity")
    axs[1].hist(df.route_breaks, bins=30, color="tab:red")
    axs[1].set(xlabel="route breaks per trip", ylabel="trips", title="Reference-route breaks")
    fig.tight_layout(); fig.savefig(OUT / "network_connectivity.png", dpi=150); plt.close(fig)

    # 5. trajectory grid: first 3 trips per partition, GPS (blue) over reference route (green) in metres
    sel = []
    for t in iter_trips():
        if sum(s.partition == t.partition for s in sel) < 3:
            sel.append(t)
    fig, axs = plt.subplots(2, 3, figsize=(15, 9))
    for ax, t in zip(axs.ravel(), sel):
        lon0, lat0 = t.track[:, 0].mean(), t.track[:, 1].mean()
        if len(t.route) and t.route.max() < len(t.arcs):
            for a in t.route:
                f, to = t.arcs[a]
                xs, ys = to_local_metres(t.nodes[[f, to], 0], t.nodes[[f, to], 1], lon0, lat0)
                ax.plot(xs, ys, color="green", lw=2, zorder=1)
        x, y = to_local_metres(t.track[:, 0], t.track[:, 1], lon0, lat0)
        ax.scatter(x, y, s=3, color="tab:blue", zorder=2)
        ax.set_title(f"{t.partition} / {t.trip_id}"); ax.set_aspect("equal"); ax.set_xlabel("east (m)"); ax.set_ylabel("north (m)")
    fig.suptitle("GPS (blue) over reference route (green)"); fig.tight_layout()
    fig.savefig(OUT / "trajectory_examples.png", dpi=150); plt.close(fig)
    print("Done ->", OUT)


if __name__ == "__main__":
    main()
