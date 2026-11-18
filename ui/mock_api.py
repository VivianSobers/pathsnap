"""Throwaway mock of the PathSnap API contract, only for testing the UI before the real API exists.

Returns synthetic geometry and placeholder numbers; none of it is a result. Never quote its output.
Usage: python ui/mock_api.py [--port 8000]
"""

from __future__ import annotations

import argparse
import json
import math
from http.server import BaseHTTPRequestHandler, HTTPServer

TRIPS = [f"mock_trip_{i}" for i in range(1, 11)]


def geometry(i: int):
    lat0, lon0 = 48.137 + 0.01 * i, 11.575
    route = [[lat0 + 0.0004 * k, lon0 + 0.001 * math.sin(k / 6)] for k in range(60)]
    gps = [[la + 0.00005 * ((k * 7) % 5 - 2), lo + 0.00005 * ((k * 3) % 5 - 2)] for k, (la, lo) in enumerate(route)]
    return gps, [route[:30], route[29:]]


class H(BaseHTTPRequestHandler):
    def _send(self, code, body=None):
        self.send_response(code)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        if body is not None:
            self.wfile.write(json.dumps(body).encode())

    def do_OPTIONS(self):
        self._send(204)

    def do_GET(self):
        if self.path == "/trips":
            self._send(200, {"trips": [{"trip_id": t, "n_points": 60} for t in TRIPS]})
        else:
            self._send(404, {"detail": "not found"})

    def do_POST(self):
        if self.path.startswith("/predict/") and self.path[9:] in TRIPS:
            gps, route = geometry(TRIPS.index(self.path[9:]))
            self._send(200, {"trip_id": self.path[9:], "gps": gps, "route": route,
                             "scores": {"top1_accuracy": 0.0, "route_continuity": 0.0}, "runtime_s": 0.0})
        else:
            self._send(404, {"detail": "unknown trip"})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    HTTPServer(("0.0.0.0", ap.parse_args().port), H).serve_forever()
