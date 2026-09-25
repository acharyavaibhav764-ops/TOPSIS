import requests, json

KEY = "3f8addd839e18787bd2e2776aa94b95e66d67a88a0bb966f89de978d1c8a2662"

# ── C1: CryptoCompare Country field format ───────────────────────────────
print("=" * 60)
print("CryptoCompare: country field values")
print("=" * 60)

r = requests.get(
    "https://min-api.cryptocompare.com/data/exchanges/general",
    params={"api_key": KEY},
    timeout=20,
)
cc = r.json()
data = cc.get("Data", {})
print(f"Total exchanges in response: {len(data)}")

countries = {}
for name, entry in data.items():
    c = entry.get("Country", "")
    countries[c] = countries.get(c, 0) + 1

print(f"Distinct country values: {len(countries)}")
print("\nTop 30 by count:")
for country, cnt in sorted(countries.items(), key=lambda x: -x[1])[:30]:
    print(f"  {cnt:3d}  {country!r}")

# Sample entry with a country
for name, entry in data.items():
    if entry.get("Country"):
        vol = entry.get("TOTALVOLUME24H", "N/A")
        print(f"\nSample exchange with country: {name!r} -> Country={entry['Country']!r}, Vol={vol}")
        break

# ── C8: WGI correct endpoint ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("WGI: testing correct endpoint routes")
print("=" * 60)

# Route 1: standard v2 with source=3 query param
url1 = "https://api.worldbank.org/v2/country/US/indicator/RQ.EST"
r1 = requests.get(url1, params={"format": "json", "per_page": 5, "mrv": 5, "source": 3}, timeout=20)
print(f"\nRoute 1 (source=3 param): status={r1.status_code}, body_len={len(r1.text)}")
if r1.status_code == 200 and r1.text.strip():
    try:
        d = r1.json()
        print(f"  type={type(d)}, len={len(d) if isinstance(d, list) else 'n/a'}")
        if isinstance(d, list) and len(d) > 1:
            print(f"  rows: {d[1]}")
        else:
            print(f"  content: {str(d)[:200]}")
    except Exception as e:
        print(f"  JSON error: {e} | raw: {r1.text[:200]}")

# Route 2: v2/sources path
url2 = "https://api.worldbank.org/v2/sources/3/country/USA/indicator/RQ.EST"
r2 = requests.get(url2, params={"format": "json", "per_page": 5}, timeout=20)
print(f"\nRoute 2 (v2/sources/3): status={r2.status_code}, body_len={len(r2.text)}")
if r2.status_code == 200 and r2.text.strip():
    print(f"  raw: {r2.text[:400]}")

# Route 3: known-working WGI code via standard v2 (government effectiveness)
url3 = "https://api.worldbank.org/v2/country/US/indicator/GE.EST"
r3 = requests.get(url3, params={"format": "json", "per_page": 5, "mrv": 5}, timeout=20)
print(f"\nRoute 3 (GE.EST standard): status={r3.status_code}, body_len={len(r3.text)}")
if r3.status_code == 200 and r3.text.strip():
    try:
        d3 = r3.json()
        print(f"  type={type(d3)}, content={str(d3)[:300]}")
    except Exception as e:
        print(f"  JSON error: {e}")

# Route 4: The WGI governance API shortcut codes
# World Bank also publishes RQ through their own API under different series ID
url4 = "https://api.worldbank.org/v2/country/US/indicator/IQ.REG.QUAL.XQ"
r4 = requests.get(url4, params={"format": "json", "per_page": 5, "mrv": 5}, timeout=20)
print(f"\nRoute 4 (IQ.REG.QUAL.XQ): status={r4.status_code}, body_len={len(r4.text)}")
if r4.status_code == 200 and r4.text.strip():
    try:
        d4 = r4.json()
        if isinstance(d4, list) and len(d4) > 1 and d4[1]:
            row = next((x for x in d4[1] if x.get("value") is not None), None)
            print(f"  First non-null: {row}")
        else:
            print(f"  content: {str(d4)[:300]}")
    except Exception as e:
        print(f"  JSON error: {e}")
