"""Round 2: recover missing PDFs by searching arXiv by title.

Runs after download_papers.py (resume-safe: skips entries already downloaded).
For each remaining bib entry: search arXiv for the title phrase, accept a hit
only if the title-token overlap is high, then download the PDF. MDPI DOIs
(10.3390/...) go straight to the journal's open PDF URL.
"""
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

BIB = Path("literature/references.bib")
OUT = Path("literature/papers")
OUT.mkdir(parents=True, exist_ok=True)
UA = {"User-Agent": "Mozilla/5.0 (academic reference fetch; contact vinnu052007@gmail.com)"}
MDPI_ISSN = {"jmse": "2077-1312"}

text = BIB.read_text(encoding="utf-8")
entries = re.findall(r"@(\w+)\s*\{([^,]+),((?:[^@])*)\}", text)


def title_field(body):  # brace-aware (titles contain nested {OTFS})
    m = re.search(r"title\s*=\s*\{", body)
    if not m:
        return ""
    depth, i, out = 1, m.end(), []
    while i < len(body) and depth:
        c = body[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        if depth:
            out.append(c)
        i += 1
    return "".join(out)


def norm(s):
    return re.findall(r"[a-z0-9]+", s.lower())


def fetch(url, dest, timeout=90):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = r.read(30_000_000)
    if data[:4] != b"%PDF" or len(data) < 20_000:
        raise ValueError(f"not a valid PDF ({len(data)} bytes)")
    dest.write_bytes(data)


def field(body, name):
    m = re.search(name + r"\s*=\s*\{([^}]*)\}", body)
    return m.group(1).strip() if m else ""


ok, miss = [], []
for etype, key, body in entries:
    key = key.strip()
    dest = OUT / f"{key}.pdf"
    if dest.exists() and dest.stat().st_size > 20_000:
        continue
    time.sleep(4)
    doi = field(body, "doi")
    try:
        # MDPI journals are open access: DOI encodes vol/issue/article
        m = re.match(r"10\.3390/([a-z]+?)(\d+)(\d{2})(\d+)$", doi or "")
        if m and m.group(1) in MDPI_ISSN:
            url = (f"https://www.mdpi.com/{MDPI_ISSN[m.group(1)]}/"
                   f"{m.group(2)}/{m.group(3)}/{m.group(4)}/pdf")
            fetch(url, dest)
            ok.append(key)
            print(f"[OK]   {key} (MDPI)")
            continue
        toks = norm(title_field(body))
        if len(toks) < 4:
            raise ValueError("no usable title")
        phrase = " ".join(toks[:8])
        q = urllib.parse.quote(f'ti:"{phrase}"')
        with urllib.request.urlopen(
            urllib.request.Request(
                f"http://export.arxiv.org/api/query?search_query={q}&max_results=5",
                headers=UA),
            timeout=60,
        ) as r:
            xml = r.read().decode("utf-8", "ignore")
        hits = re.findall(
            r"<entry>.*?<id>http://arxiv\.org/abs/([0-9]{4}\.[0-9]{4,5})v?\d*</id>.*?"
            r"<title>(.*?)</title>.*?</entry>", xml, re.S)
        want = set(toks)
        picked = None
        for aid, t in hits:
            if len(want & set(norm(t))) / max(1, len(want | set(norm(t)))) >= 0.75:
                picked = aid
                break
        if not picked:
            raise ValueError("no arXiv title match")
        time.sleep(3)
        fetch(f"https://arxiv.org/pdf/{picked}", dest)
        ok.append(key)
        print(f"[OK]   {key} (arXiv:{picked})")
        time.sleep(3)
    except Exception as e:
        miss.append((key, doi, str(e)[:60]))
        print(f"[MISS] {key}: {e}")

print("\n==== ROUND-2 SUMMARY ====")
print(f"recovered  : {len(ok)}")
print(f"still miss : {len(miss)}")
for key, doi, why in miss:
    print(f"  {key:28s} {doi}")
