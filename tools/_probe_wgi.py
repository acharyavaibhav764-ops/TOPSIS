"""Find working WGI Regulatory Quality endpoint."""
import requests, json

# WGI Regulatory Quality is available through the World Bank API under
# several possible series codes. Test them all.
CANDIDATES = [
    ("IQ.REG.QUAL.XQ",   "WGI Regulatory Quality (IQ.REG.QUAL.XQ)"),
    ("RQ.EST",            "WGI RQ.EST (should fail)"),
    ("GE.EST",            "WGI GE.EST (should fail)"),
    ("IQ.GOV.QUAL.XQ",   "WGI Gov Quality (IQ.GOV.QUAL.XQ)"),
    ("IC.REG.PROC",       "Reg procedures (IC.REG.PROC)"),
    ("IQ.SCI.OVRL.XQ",   "Overall governance (IQ.SCI.OVRL.XQ)"),
]

for code, label in CANDIDATES:
    url = f"https://api.worldbank.org/v2/country/US/indicator/{code}"
    r = requests.get(url, params={"format": "json", "per_page": 5, "mrv": 5}, timeout=20)
    try:
        d = r.json()
        if isinstance(d, list) and len(d) > 1 and d[1]:
            # Find first non-null
            row = next((x for x in d[1] if x.get("value") is not None), None)
            if row:
                print(f"FOUND  {label}: value={row['value']}, year={row['date']}")
            else:
                print(f"NULL   {label}: all null")
        else:
            print(f"EMPTY  {label}: {str(d)[:80]}")
    except Exception as e:
        print(f"ERROR  {label}: {e}")
