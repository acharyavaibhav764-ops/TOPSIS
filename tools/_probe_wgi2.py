"""
Find the correct WGI data access route.
WGI was moved from the standard indicator API.
"""
import requests, json

# The WGI dataset is at World Bank Open Data API source 3 (WGI dataset)
# but requires using the series endpoint, not country/indicator

# Route A: /v2/sources/3 to list series
url_a = "https://api.worldbank.org/v2/sources/3/series"
r = requests.get(url_a, params={"format":"json","per_page":10}, timeout=20)
print(f"Sources/3/series status: {r.status_code}, len: {len(r.text)}")
try:
    d = r.json()
    print("First response:", str(d)[:300])
except:
    print("Raw:", r.text[:300])

print()

# Route B: The bulk WGI download URL (known public file)
# World Bank hosts WGI as a bulk download CSV
wgi_bulk = "https://databank.worldbank.org/data/download/WGI_csv.zip"
r2 = requests.head(wgi_bulk, timeout=10)
print(f"WGI bulk ZIP: status={r2.status_code}, Content-Type={r2.headers.get('Content-Type','?')}")

print()

# Route C: WGI API via their dedicated endpoint
# The World Bank has a separate Governance API
wgi_api = "https://govdata360-backend.worldbank.org/api/v1/data"
r3 = requests.get(wgi_api, params={
    "indicator":"WGI_RQ_EST",
    "countries":"USA",
    "frequency":"year",
    "fields":"value,years"
}, timeout=20)
print(f"GovData360: status={r3.status_code}, len={len(r3.text)}")
print("GovData360 raw:", r3.text[:400])

print()

# Route D: RESTCOUNTRIES-style: try the new WGI API path
wgi_new = "https://api.worldbank.org/v2/en/indicator/GV.RGV.EST"
r4 = requests.get(wgi_new, params={"format":"json","per_page":5,"country":"US","mrv":5}, timeout=20)
print(f"GV.RGV.EST: status={r4.status_code}")
try:
    d4 = r4.json()
    print("payload:", str(d4)[:300])
except:
    print("raw:", r4.text[:300])
