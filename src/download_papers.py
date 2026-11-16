"""Download open-access PDFs for every literature-review reference.

Parses literature/references.bib. Per entry:
  1) arXiv PDF if an arXiv:<id> appears anywhere in the entry
  2) otherwise Unpaywall (open-access resolver) for the DOI
Saved to literature/papers/<bibkey>.pdf. Entries with no OA copy are listed
with their DOI for manual institutional (IEEE Xplore) access.
Resume-safe: existing valid PDFs are skipped. arXiv hit spacing: 3 s.
"""
import json
import re
import time
import urllib.request
from pathlib import Path

BIB = Path("literature/references.bib")
OUT = Path("literature/papers")
OUT.mkdir(parents=True, exist_ok=True)
EMAIL = "vinnu052007@gmail.com"
UA = {"User-Agent": "Mozilla/5.0 (academic reference fetch; contact " + EMAIL + ")"}

text = BIB.read_text(encoding="utf-8")
entries = re.findall(r"@(\w+)\s*\{([^,]+),((?:[^@])*)\}", text)


def field(body, name):
    m = re.search(name + r"\s*=\s*\{([^}]*)\}", body)
    return m.group(1).strip() if m else ""


def arxiv_id(body):
    m = re.search(r"arxiv[:\s.]*([0-9]{4}\.[0-9]{4,5})", body, re.I)
    return m.group(1) if m else ""


def fetch(url, dest, timeout=90):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = r.read(30_000_000)
    if data[:4] != b"%PDF" or len(data) < 20_000:
        raise ValueError(f"not a valid PDF ({len(data)} bytes)")
    dest.write_bytes(data)


ok, miss = [], []
for etype, key, body in entries:
    key = key.strip()
    dest = OUT / f"{key}.pdf"
    if dest.exists() and dest.stat().st_size > 20_000:
        ok.append(key)
        print(f"[SKIP] {key} (already have)")
        continue
    doi = field(body, "doi")
    ax = arxiv_id(body)
    try:
        if ax:
            fetch(f"https://arxiv.org/pdf/{ax}", dest)
            time.sleep(3)
        elif doi:
            with urllib.request.urlopen(
                urllib.request.Request(
                    f"https://api.unpaywall.org/v2/{doi}?email={EMAIL}", headers=UA
                ),
                timeout=60,
            ) as r:
                upw = json.loads(r.read().decode())
            loc = upw.get("best_oa_location") or {}
            url = loc.get("url_for_pdf")
            if not url:
                raise ValueError("no OA copy registered")
            fetch(url, dest)
        else:
            raise ValueError("no arXiv id and no DOI")
        ok.append(key)
        print(f"[OK]   {key}")
    except Exception as e:
        miss.append((key, ax or doi, str(e)[:70]))
        print(f"[MISS] {key}: {e}")

print("\n==== SUMMARY ====")
print(f"downloaded : {len(ok)} / {len(entries)}")
print(f"no OA copy : {len(miss)} — grab via campus IEEE Xplore / interlibrary:")
for key, ident, why in miss:
    print(f"  {key:28s} {ident:32s} ({why})")
