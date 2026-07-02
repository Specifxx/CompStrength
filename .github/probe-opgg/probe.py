"""One-shot probe of OP.GG's internal champion-stats API.

Run from GitHub Actions (unrestricted egress). Answers, with raw evidence:
- how deep the per-patch `versions` history goes (need 15.01-16.12)
- which tier / region params work
- exact response schema: winrate, game counts, champion identifiers
Results are written to /tmp/opgg_probe for publication to an orphan branch.
"""
import gzip
import json
import os
import time
import urllib.error
import urllib.request

BASE = "https://lol-api-champion.op.gg/api"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
}
OUT = "/tmp/opgg_probe"
os.makedirs(OUT, exist_ok=True)
index = []


def get(url):
    req = urllib.request.Request(url, headers=HEADERS)
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            body = r.read()
            status = r.status
    except urllib.error.HTTPError as e:
        status = e.code
        body = e.read()[:2000]
    except Exception as e:  # noqa: BLE001 - record any failure verbatim
        status = -1
        body = str(e).encode()
    index.append({"url": url, "status": status, "bytes": len(body), "secs": round(time.time() - t0, 2)})
    print(f"{status} {len(body):>9}B {url}", flush=True)
    return status, body


def save(name, data, compress=False):
    path = os.path.join(OUT, name)
    if compress:
        with gzip.open(path, "wb", compresslevel=9) as f:
            f.write(data)
    else:
        with open(path, "wb") as f:
            f.write(data)


def norm_versions(payload):
    """Versions endpoint shape is unverified; normalise to a list of strings."""
    x = payload
    if isinstance(x, dict):
        x = x.get("data") or x.get("versions") or []
    out = []
    if isinstance(x, list):
        for e in x:
            if isinstance(e, str):
                out.append(e)
            elif isinstance(e, (int, float)):
                out.append(str(e))
            elif isinstance(e, dict):
                for k in ("version", "name", "id"):
                    if k in e:
                        out.append(str(e[k]))
                        break
    return out


def slim_pos(p):
    out = {}
    for k, v in p.items():
        if isinstance(v, dict):
            out[k] = {kk: vv for kk, vv in v.items() if not isinstance(vv, (list, dict))}
        elif not isinstance(v, list):
            out[k] = v
    return out


def slim_champ(c):
    out = {}
    for k, v in c.items():
        if k == "positions" and isinstance(v, list):
            out[k] = [slim_pos(p) for p in v if isinstance(p, dict)]
        elif isinstance(v, dict):
            if k == "average_stats":
                out[k] = v
        elif isinstance(v, list):
            continue  # drop heavy nested builds/runes/items
        else:
            out[k] = v
    return out


def slim_payload(body):
    j = json.loads(body)
    if isinstance(j, dict) and isinstance(j.get("data"), list):
        out = {k: v for k, v in j.items() if k != "data"}
        out["data"] = [slim_champ(c) if isinstance(c, dict) else c for c in j["data"]]
        return json.dumps(out).encode()
    return body


# 1. versions list per region -----------------------------------------------
versions = {}
for region in ["kr", "na", "euw", "global"]:
    s, b = get(f"{BASE}/{region}/champions/ranked/versions")
    save(f"versions_{region}.json", b)
    if s == 200:
        try:
            versions[region] = norm_versions(json.loads(b))
        except Exception as e:  # noqa: BLE001
            print(f"versions parse failed for {region}: {e}")

kr_versions = versions.get("kr") or []
print(f"KR versions ({len(kr_versions)}):", kr_versions)
latest = kr_versions[0] if kr_versions else None

# 2. tier param variants on the latest version ------------------------------
if latest:
    for tier in [
        "all", "gold_plus", "platinum_plus", "emerald_plus", "diamond_plus",
        "master_plus", "grandmaster", "challenger", "master",
    ]:
        s, b = get(f"{BASE}/kr/champions/ranked?tier={tier}&version={latest}")
        if s == 200 and tier == "emerald_plus":
            save(f"champions_kr_{latest}_emerald_plus.raw.json.gz", b, compress=True)

# 3. per-version data for every listed KR version (trimmed, capped at 80) ---
for v in kr_versions[:80]:
    s, b = get(f"{BASE}/kr/champions/ranked?tier=emerald_plus&version={v}")
    if s == 200:
        try:
            save(f"champions_kr_{v}.slim.json.gz", slim_payload(b), compress=True)
        except Exception:  # noqa: BLE001 - keep raw if trim fails
            save(f"champions_kr_{v}.slim.json.gz", b, compress=True)
    time.sleep(0.4)

# 4. one other region + global, latest version ------------------------------
for region in ["na", "global"]:
    vs = versions.get(region) or []
    if vs:
        s, b = get(f"{BASE}/{region}/champions/ranked?tier=emerald_plus&version={vs[0]}")
        if s == 200:
            save(f"champions_{region}_{vs[0]}.slim.json.gz", slim_payload(b), compress=True)

# 5. no-version request (live current patch) --------------------------------
s, b = get(f"{BASE}/kr/champions/ranked?tier=emerald_plus")
if s == 200:
    save("champions_kr_current.slim.json.gz", slim_payload(b), compress=True)

# 6. Data Dragon champion id->name mapping (for identifier verification) ----
s, b = get("https://ddragon.leagueoflegends.com/api/versions.json")
save("ddragon_versions.json", b)
if s == 200:
    dv = json.loads(b)[0]
    s2, b2 = get(f"https://ddragon.leagueoflegends.com/cdn/{dv}/data/en_US/champion.json")
    if s2 == 200:
        save("ddragon_champion.json.gz", b2, compress=True)

save("index.json", json.dumps(index, indent=1).encode())
print("probe complete;", len(index), "requests")
