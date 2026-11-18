"""Write Folium maps to results/maps/.
    python scripts/make_maps.py                      # first 3 trips of each partition
    python scripts/make_maps.py train 00000048       # one specific trip
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pathsnap.eda_io import RAW_ROOT, trip_dirs, load_trip
from pathsnap.maps import build_trip_map

OUT = Path("results/maps"); OUT.mkdir(parents=True, exist_ok=True)


def one(part, d):
    t = load_trip(d, part)
    path = OUT / f"{part}_{t.trip_id}.html"
    build_trip_map(t, title=f"{part} / trip {t.trip_id}").save(str(path))
    print("wrote", path)


if len(sys.argv) == 3:
    part, tid = sys.argv[1:]
    one(part, RAW_ROOT / part / tid)
else:
    for part in ("train", "student_test"):
        for d in trip_dirs(part)[:3]:
            one(part, d)
