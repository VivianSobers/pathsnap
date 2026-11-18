"""Render report/PathSnap_Report.md to HTML and print it to PDF with headless Chrome or Chromium.

    python report/build_report.py            # writes report/PathSnap_Report.md from results/
    python report/build_pdf.py [output.pdf]  # default output: report/PathSnap_Report.pdf
"""

import shutil
import subprocess
import sys
from pathlib import Path

import markdown

HERE = Path(__file__).resolve().parent
CSS = """@page{size:A4;margin:15mm 13mm}body{font-family:'DejaVu Sans',Arial,sans-serif;font-size:10pt;line-height:1.42;color:#111}
h1{font-size:18pt;margin:0 0 6pt}h2{font-size:13.5pt;margin:16pt 0 6pt;border-bottom:1px solid #bbb;padding-bottom:2pt;page-break-after:avoid}
h3{font-size:11.3pt;margin:12pt 0 4pt;page-break-after:avoid}table{border-collapse:collapse;width:100%;margin:6pt 0 10pt;font-size:8.2pt;page-break-inside:avoid}
th,td{border:1px solid #999;padding:3pt 4pt;text-align:left;vertical-align:top}th{background:#eee}code{font-size:8.6pt;background:#f3f3f3;padding:0 2pt}
img{max-width:100%;display:block;margin:6pt auto;page-break-inside:avoid}td img{max-width:100%;margin:0}p{margin:5pt 0}li{margin:2pt 0}"""


def main():
    out = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else HERE / "PathSnap_Report.pdf"
    body = markdown.markdown((HERE / "PathSnap_Report.md").read_text(), extensions=["tables"])
    html = HERE / "PathSnap_Report.html"
    html.write_text("<!doctype html><html><head><meta charset='utf-8'><title>PathSnap Technical Report</title>"
                    f"<style>{CSS}</style></head><body>{body}</body></html>")
    browser = next((b for b in ("google-chrome", "chromium", "chromium-browser", "chrome") if shutil.which(b)), None)
    if browser is None:
        sys.exit(f"Wrote {html}; no Chrome or Chromium found, so print it to PDF from a browser instead.")
    subprocess.run([browser, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={out}", html.as_uri()], check=True, capture_output=True)
    print(out)


if __name__ == "__main__":
    main()
