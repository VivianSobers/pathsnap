"""Folium maps (Phase 7, Harsimran).

build_trip_map(trip, pred_arcs=None) -> folium.Map with layers:
  GPS observations (blue), road network near the track (grey),
  reference route (green), predicted route (red, optional), legend, layer control.
"""

import numpy as np
import folium

MAX_NET_ARCS = 6000  # keep HTML small for big cities
NET_BUFFER_DEG = 0.003  # ~300 m around the GPS bounding box


def _arc_latlon(trip, arc_id):
    f, t = trip.arcs[arc_id]
    # folium wants (lat, lon); files store lon, lat
    return [(trip.nodes[f, 1], trip.nodes[f, 0]), (trip.nodes[t, 1], trip.nodes[t, 0])]


def _legend(has_pred):
    items = [
        ("#1f77b4", "Raw GPS observations"),
        ("#999999", "Road network (nearby arcs)"),
        ("#2ca02c", "Reference route"),
    ]
    if has_pred:
        items.append(("#d62728", "Predicted route"))
    rows = "".join(
        f'<div><span style="background:{c};width:14px;height:4px;display:inline-block;'
        f'margin-right:6px"></span>{n}</div>'
        for c, n in items
    )
    return folium.Element(
        '<div style="position:fixed;bottom:30px;left:30px;z-index:9999;background:white;'
        f'padding:8px 10px;border:1px solid #888;border-radius:4px;font-size:12px">{rows}</div>'
    )


def build_trip_map(trip, pred_arcs=None, title=None):
    lon, lat = trip.track[:, 0], trip.track[:, 1]
    m = folium.Map(
        location=[float(lat.mean()), float(lon.mean())],
        zoom_start=15,
        tiles="OpenStreetMap",
    )

    # road network inside the padded GPS bounding box
    nd = trip.nodes
    in_box = (
        (nd[:, 0] > lon.min() - NET_BUFFER_DEG)
        & (nd[:, 0] < lon.max() + NET_BUFFER_DEG)
        & (nd[:, 1] > lat.min() - NET_BUFFER_DEG)
        & (nd[:, 1] < lat.max() + NET_BUFFER_DEG)
    )
    arc_ok = np.where(in_box[trip.arcs[:, 0]] & in_box[trip.arcs[:, 1]])[0][
        :MAX_NET_ARCS
    ]
    net = folium.FeatureGroup(name="Road network", show=True)
    for a in arc_ok:
        folium.PolyLine(
            _arc_latlon(trip, a), color="#999999", weight=2, opacity=0.7
        ).add_to(net)
    net.add_to(m)

    # reference route
    ref = folium.FeatureGroup(name="Reference route", show=True)
    for a in trip.route:
        folium.PolyLine(
            _arc_latlon(trip, int(a)), color="#2ca02c", weight=9, opacity=0.9
        ).add_to(ref)
    ref.add_to(m)

    # predicted route (list of arc ids, in order)
    if pred_arcs is not None:
        pr = folium.FeatureGroup(name="Predicted route", show=True)
        for a in pred_arcs:
            folium.PolyLine(
                _arc_latlon(trip, int(a)),
                color="#d62728",
                weight=3,
                opacity=0.9,
                dash_array="6",
            ).add_to(pr)
        pr.add_to(m)

    # GPS observations (every point, small circles; tooltip shows time)
    gps = folium.FeatureGroup(name="GPS observations", show=True)
    for lo, la, ts in trip.track:
        folium.CircleMarker(
            (float(la), float(lo)),
            radius=1.5,
            color="#1f77b4",
            fill=True,
            fill_opacity=0.9,
            weight=0,
            tooltip=f"t={ts:.0f}s",
        ).add_to(gps)
    gps.add_to(m)
    folium.Marker(
        (float(lat[0]), float(lon[0])), tooltip="Start", icon=folium.Icon(color="green")
    ).add_to(m)
    folium.Marker(
        (float(lat[-1]), float(lon[-1])), tooltip="End", icon=folium.Icon(color="red")
    ).add_to(m)

    m.fit_bounds(
        [[float(lat.min()), float(lon.min())], [float(lat.max()), float(lon.max())]]
    )
    folium.LayerControl(collapsed=False).add_to(m)
    m.get_root().html.add_child(_legend(pred_arcs is not None))
    if title:
        m.get_root().html.add_child(
            folium.Element(
                f'<h4 style="position:fixed;top:8px;left:60px;z-index:9999;background:white;padding:4px 8px">{title}</h4>'
            )
        )
    return m
