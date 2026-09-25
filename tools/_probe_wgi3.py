"""Find exact WGI series ID and how to query per-country data."""
import requests, json

# Step 1: List all WGI series to find Regulatory Quality exact ID
url = "https://api.worldbank.org/v2/sources/3/series"
r = requests.get(url, params={"format": "json", "per_page": 50}, timeout=20)
d = r.json()
series_list = d.get("source", [{}])[0].get("concept", [{}])[0].get("variable", [])
print("All WGI series:")
for s in series_list:
    print(f"  {s['id']!r:45s} {s['value']!r}")

# Step 2: Query the country data for USA for the Regulatory Quality series
# Try GOV_WGI_RQ.EST
print()
target = "GOV_WGI_RQ.EST"
url2 = f"https://api.worldbank.org/v2/sources/3/country/USA/series/{target}/time/all"
r2 = requests.get(url2, params={"format": "json"}, timeout=20)
print(f"Per-country query status: {r2.status_code}")
try:
    d2 = r2.json()
    print("Response:", str(d2)[:500])
except:
    print("Raw:", r2.text[:500])
