"""Road-network figure for the report: one training trip and one student-test trip.

Draws every arc of the trip's network inside a 2.5 km square centred on the middle GPS point in grey, the
reference route in green and the GPS points in blue, in local metres. Writes results/eda/road_network_examples.png.

    python -m scripts.road_network_plot
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

from pathsnap.geo import local_origin, to_metric
from pathsnap.io import load_trip

RAW = Path("data/raw/Student folder")
TRIPS = [("train", "00000001"), ("student_test", "00000007")]
WINDOW_M = 2500
INK, MUTED = "#1f2328", "#59636e"
NETWORK, ROUTE, GPS = "#b8bec6", "#1a7f37", "#0969da"


def panel(ax, partition, trip_id):
    trip = load_trip(RAW / partition / trip_id)
    origin = local_origin(trip.track.lon.to_numpy(), trip.track.lat.to_numpy())
    gx, gy = to_metric(trip.track.lon.to_numpy(), trip.track.lat.to_numpy(), origin)
    nx, ny = to_metric(trip.nodes.lon.to_numpy(), trip.nodes.lat.to_numpy(), origin)
    a, b = trip.arcs.from_node.to_numpy(), trip.arcs.to_node.to_numpy()
    cx, cy = gx[len(gx) // 2], gy[len(gy) // 2]
    h = WINDOW_M / 2
    x0, x1, y0, y1 = cx - h, cx + h, cy - h, cy + h
    inside = ((nx[a] >= x0) & (nx[a] <= x1) & (ny[a] >= y0) & (ny[a] <= y1)) | \
             ((nx[b] >= x0) & (nx[b] <= x1) & (ny[b] >= y0) & (ny[b] <= y1))
    segs = np.stack([np.c_[nx[a], ny[a]], np.c_[nx[b], ny[b]]], axis=1)
    ax.add_collection(LineCollection(segs[inside], colors=NETWORK, linewidths=0.6, zorder=1))
    r = trip.route
    ax.add_collection(LineCollection(segs[r], colors=ROUTE, linewidths=2.2, zorder=2))
    ax.scatter(gx, gy, s=5, color=GPS, linewidths=0, zorder=3)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal")
    ax.set_title(f"{partition} / {trip_id} (2.5 km window, {inside.sum():,} of {len(segs):,} arcs)", color=INK, fontsize=10)
    ax.set_xlabel("east (m)", color=MUTED); ax.set_ylabel("north (m)", color=MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    for s in ax.spines.values():
        s.set_color("#d0d7de")


def main():
    fig, axs = plt.subplots(1, 2, figsize=(13, 6))
    for ax, (p, t) in zip(axs, TRIPS):
        panel(ax, p, t)
    handles = [Line2D([], [], color=NETWORK, lw=1.5, label="road network (arcs)"),
               Line2D([], [], color=ROUTE, lw=2.5, label="reference route"),
               Line2D([], [], color=GPS, marker="o", ls="", ms=4, label="GPS points")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle("Road network near the GPS track (train and student test)", color=INK)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    out = Path("results/eda/road_network_examples.png")
    fig.savefig(out, dpi=150); plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
