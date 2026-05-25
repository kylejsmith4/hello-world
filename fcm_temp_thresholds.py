import io
import requests
import pandas as pd

url = (
    "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
    "?station=FCM"
    "&data=tmpf"
    "&year1=2026&month1=1&day1=1"
    "&year2=2026&month2=5&day2=26"
    "&tz=America/Chicago"
    "&format=onlycomma"
    "&latlon=no&elev=no"
    "&missing=M&trace=T&direct=no"
    "&report_type=1&report_type=3&report_type=4"
)

resp = requests.get(url, headers={"User-Agent": "python-requests/fcm-temp-analysis"}, timeout=30)
resp.raise_for_status()
df = pd.read_csv(io.StringIO(resp.text), na_values=["M"])
df["valid"] = pd.to_datetime(df["valid"])
df["date"] = df["valid"].dt.date

daily = df.groupby("date", as_index=False)["tmpf"].max()
daily["gt80"] = daily["tmpf"] > 80
daily["gt85"] = daily["tmpf"] > 85

def consecutive_days(mask_col):
    runs = []
    for i in range(len(daily) - 1):
        if daily.loc[i, mask_col] and daily.loc[i + 1, mask_col]:
            runs.append((daily.loc[i, "date"], daily.loc[i + 1, "date"],
                         daily.loc[i, "tmpf"], daily.loc[i + 1, "tmpf"]))
    return runs

print(">80°F consecutive:", consecutive_days("gt80"))
print(">85°F consecutive:", consecutive_days("gt85"))
