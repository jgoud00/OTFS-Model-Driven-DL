"""Round 3: recover remaining paywalled papers via Semantic Scholar.

For each bib entry still missing a PDF: query the Semantic Scholar API by DOI.
If it reports an arXiv ID -> arXiv PDF. Else if it reports an openAccessPdf
URL -> try that (validated as a real PDF). Special cases: MDPI (direct PDF,
browser headers) and Authorea (scrape the landing page for the PDF link).
Resume-safe; ~1.5 s spacing to respect the free API tier.
"""
import json
import re
import time
import urllib.request
from pathlib import Path

BIB = Path("literature/references.bib")
OUT = Path("literature/papers")
OUT.mkdir(parents=True, exist_ok=True)
UA = {"User-Agent": "Mozilla/5.0 (academic reference fetch; contact vinnu052007@gmail.com)"}
BROWSER = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                         "(KHTML, like Gecko) Chrome/126.0 Safari/537.36",
           "Accept": "text/html,application/xhtml+xml,application/pdf,*/*"}

text = BIB.read_text(encoding="utf-8")
entries = re.findall(r"@(\w+)\s*\{([^,]+),((?:[^@])*)\}", text)


def field(body, name):
    m = re.search(name + r"\s*=\s*\{([^}]*)\}", body)
    return m.group(1).strip() if m else ""


def fetch(url, dest, headers=UA, timeout=90):
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = r.read(30_000_000)
    if data[:4] != b"%PDF" or len(data) < 20_000:
        raise ValueError(f"not a valid PDF ({len(data)} bytes)")
    dest.write_bytes(data)


def s2(doi):
    url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}?fields=externalIds,openAccessPdf"
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


ok, miss = [], []
for etype, key, body in entries:
    key = key.strip()
    dest = OUT / f"{key}.pdf"
    if dest.exists() and dest.stat().st_size > 20_000:
        continue
    doi = field(body, "doi")
    try:
        time.sleep(1.5)
        if doi.startswith("10.3390/"):  # MDPI open access; earlier 403 was bot filtering
            m = re.match(r"10\.3390/([a-z]+?)(\d+)(\d{2})(\d+)$", doi)
            fetch(f"https://www.mdpi.com/2077-1312/{m.group(2)}/{m.group(3)}/{m.group(4)}/pdf",
                  dest, headers=BROWSER)
            ok.append((key, "MDPI"))
            print(f"[OK]   {key} (MDPI)")
            continue
        if "authorea" in doi.lower() or doi.startswith("10.22541/"):
            with urllib.request.urlopen(
                urllib.request.Request(f"https://www.authorea.com/doi/{doi}", headers=BROWSER),
                timeout=60) as r:
                html = r.read().decode("utf-8", "ignore")
            pdfs = re.findall(r'https?://[^\s"\']+\.pdf[^\s"\']*', html)
            if not pdfs:
                raise ValueError("no pdf link on Authorea page")
            fetch(pdfs[0], dest, headers=BROWSER)
            ok.append((key, "Authorea"))
            print(f"[OK]   {key} (Authorea)")
            continue
        try:
            info = s2(doi)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise ValueError("not indexed in Semantic Scholar")
            if e.code == 429:
                time.sleep(10)
                info = s2(doi)
            else:
                raise
        ax = (info.get("externalIds") or {}).get("ArXiv")
        oa = (info.get("openAccessPdf") or {}).get("url")
        if ax:
            time.sleep(2)
            fetch(f"https://arxiv.org/pdf/{ax}", dest)
            ok.append((key, f"arXiv:{ax}"))
            print(f"[OK]   {key} (arXiv:{ax})")
        elif oa:
            fetch(oa, dest, headers=BROWSER)
            ok.append((key, "S2-OA"))
            print(f"[OK]   {key} (S2-OA)")
        else:
            raise ValueError("no OA or arXiv id in Semantic Scholar")
    except Exception as e:
        miss.append((key, doi, str(e)[:60]))
        print(f"[MISS] {key}: {e}")

print("\n==== ROUND-3 SUMMARY ====")
print(f"recovered  : {len(ok)}")
for key, src in ok:
    print(f"  {key} <- {src}")
print(f"still miss : {len(miss)}")
for key, doi, why in miss:
    print(f"  {key:28s} {doi}  ({why})")
