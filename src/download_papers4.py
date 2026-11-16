"""Round 4: final open-access recovery attempt for the remaining papers.

Earlier zero-result arXiv searches used all:"..." AND all:"..." (quoted
phrases + AND under the all: field) — that form silently returns 0 even for
known-existing phrases, so "no preprint exists" was never actually tested.
Verified working forms (probed): ti:"phrase" (round 2), abs:"phrase",
all:x AND all:y (unquoted terms).

Per still-missing bib entry (resume-safe; skip-list for known no-OA items):
  1) arXiv exact-title phrase   ti:"first 6 title tokens"
  2) arXiv abstract phrase      abs:"first 6 title tokens"  (retitled preprints)
  3) arXiv unquoted terms       all:p1 AND all:p2 AND all:p3 AND all:p4
  4) OpenAlex by DOI (different OA index than Unpaywall) -> location pdf URLs
  5) MDPI direct PDF with browser headers (bot-blocked in earlier rounds)
Accepted arXiv matches are printed with the arXiv title for eyeball checking
(round-2 lesson: the CRC monograph matched a same-title arXiv paper).
Pacing: 4 s before every arXiv call; on 429 sleep 45 s and retry once.
"""
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

BIB = Path("literature/references.bib")
OUT = Path("literature/papers")
OUT.mkdir(parents=True, exist_ok=True)
EMAIL = "vinnu052007@gmail.com"
UA = {"User-Agent": "Mozilla/5.0 (academic reference fetch; contact " + EMAIL + ")"}
BROWSER = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/126.0 Safari/537.36",
           "Accept": "application/pdf,*/*"}
MDPI_ISSN = {"jmse": "2077-1312"}
# No legal OA copy exists; do not search: CRC monograph (book), IEEE symposium
# digest, Authorea preprint (S2 stub + 403 in rounds 2-3).
SKIP = {"otfs2022crc", "hadani2017mmwave", "isacmitigation2024"}
STOP = {"a", "an", "the", "for", "of", "with", "and", "in", "on", "by", "to"}

text = BIB.read_text(encoding="utf-8")
entries = re.findall(r"@(\w+)\s*\{([^,]+),((?:[^@])*)\}", text)


def field(body, name):
    m = re.search(name + r"\s*=\s*\{([^}]*)\}", body)
    return m.group(1).strip() if m else ""


def title_field(body):
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
    return [t for t in re.findall(r"[a-z0-9]+", s.lower()) if t not in STOP]


def fetch(url, dest, headers=UA, timeout=90):
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = r.read(30_000_000)
    if data[:4] != b"%PDF" or len(data) < 20_000:
        raise ValueError(f"not a valid PDF ({len(data)} bytes)")
    dest.write_bytes(data)


def arxiv_search(q):
    url = ("http://export.arxiv.org/api/query?search_query="
           + urllib.parse.quote(q) + "&max_results=5")
    for attempt in (1, 2):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA),
                                        timeout=60) as r:
                return r.read().decode("utf-8", "ignore")
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt == 1:
                time.sleep(45)
                continue
            raise


def arxiv_hits(xml):
    return re.findall(
        r"<entry>.*?<id>http://arxiv\.org/abs/([0-9]{4}\.[0-9]{4,5})v?\d*</id>"
        r".*?<title>(.*?)</title>.*?</entry>", xml, re.S)


def openalex_pdfs(doi):
    url = f"https://api.openalex.org/works/doi:{doi}?mailto={EMAIL}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA),
                                timeout=60) as r:
        w = json.loads(r.read().decode())
    locs = [w.get("best_oa_location") or {}, w.get("open_access") or {}]
    locs += w.get("locations") or []
    seen, urls = set(), []
    for loc in locs:
        u = (loc or {}).get("pdf_url") or (loc or {}).get("oa_url")
        if u and u not in seen:
            seen.add(u)
            urls.append(u)
    if not urls:
        raise ValueError("no pdf location in OpenAlex")
    return urls


ok, miss = [], []
for etype, key, body in entries:
    key = key.strip()
    dest = OUT / f"{key}.pdf"
    if dest.exists() and dest.stat().st_size > 20_000:
        continue
    if key in SKIP:
        print(f"[SKIP] {key} (no legal OA expected)")
        miss.append((key, "skip-list"))
        continue
    doi = field(body, "doi")
    got, reasons = None, []

    # 0) MDPI journals are open access; earlier 403 was bot filtering.
    m = re.match(r"10\.3390/([a-z]+?)(\d+)(\d{2})(\d+)$", doi or "")
    if m and m.group(1) in MDPI_ISSN:
        try:
            fetch(f"https://www.mdpi.com/{MDPI_ISSN[m.group(1)]}/{m.group(2)}/"
                  f"{m.group(3)}/{m.group(4)}/pdf", dest, headers=BROWSER)
            got = "MDPI"
        except Exception as e:
            reasons.append(f"MDPI: {str(e)[:40]}")

    # 1-3) arXiv, three verified query forms.
    if not got:
        toks = norm(title_field(body))
        if len(toks) >= 4:
            want = set(toks)
            phrase = " ".join(toks[:6])
            forms = [(f'ti:"{phrase}"', 0.75),
                     (f'abs:"{phrase}"', 0.65),
                     (" AND ".join("all:" + t for t in toks[:4]), 0.65)]
            for form, need in forms:
                try:
                    time.sleep(4)
                    hits = arxiv_hits(arxiv_search(form))
                except Exception as e:
                    reasons.append(f"arXiv[{form.split(':')[0]}]: {str(e)[:30]}")
                    continue
                for aid, t in hits:
                    j = len(want & set(norm(t))) / max(1, len(want | set(norm(t))))
                    if j >= need:
                        time.sleep(4)
                        fetch(f"https://arxiv.org/pdf/{aid}", dest)
                        got = f"arXiv:{aid} ({form.split(':')[0]}) {t[:60]}"
                        break
                if got:
                    break
        else:
            reasons.append("no usable title")

    # 4) OpenAlex last resort.
    if not got:
        try:
            time.sleep(0.5)
            for u in openalex_pdfs(doi)[:4]:
                try:
                    fetch(u, dest, headers=BROWSER)
                    got = f"OpenAlex: {u[:60]}"
                    break
                except Exception as e:
                    reasons.append(f"OA-fetch: {str(e)[:30]}")
        except Exception as e:
            reasons.append(f"OpenAlex: {str(e)[:40]}")

    if got:
        ok.append((key, got))
        print(f"[OK]   {key} <- {got}")
    else:
        miss.append((key, doi))
        print(f"[MISS] {key}  {'; '.join(reasons[:3])}")

print("\n==== ROUND-4 SUMMARY ====")
print(f"recovered  : {len(ok)}")
for key, src in ok:
    print(f"  {key} <- {src}")
print(f"still miss : {len(miss)}")
for key, doi in miss:
    print(f"  {key:28s} {doi}")
