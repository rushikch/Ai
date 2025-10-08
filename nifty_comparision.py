import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# 1️⃣ Load Data
# -----------------------------
data = {
    "Year": [2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020],
    "NIFTY_100_TRI": [40,56,-53,79,18,-25,32,7,34,-2,5,33,3,12,16],
    "NIFTY_50_TRI": [42,54,-51,72,18,-24,29,7,32,-4,4,30,6,14,16],
    "NIFTY_NEXT_50_TRI": [30,74,-64,124,16,-31,49,4,45,-7,7,46,-8,2,16]
}

df = pd.DataFrame(data)

# -----------------------------
# 2️⃣ Compute Cumulative Growth
# -----------------------------
def compute_cumulative(returns):
    growth = [1]  # Start with 1x (base)
    for r in returns:
        growth.append(growth[-1] * (1 + r / 100))
    return growth[1:]

df["NIFTY_100_CUM"] = compute_cumulative(df["NIFTY_100_TRI"])
df["NIFTY_50_CUM"] = compute_cumulative(df["NIFTY_50_TRI"])
df["NIFTY_NEXT_50_CUM"] = compute_cumulative(df["NIFTY_NEXT_50_TRI"])

# -----------------------------
# 3️⃣ Calculate CAGR
# -----------------------------
years = len(df)
def calc_cagr(end_value):
    return (end_value)**(1/years) - 1

cagr = {
    "NIFTY 100": calc_cagr(df["NIFTY_100_CUM"].iloc[-1]),
    "NIFTY 50": calc_cagr(df["NIFTY_50_CUM"].iloc[-1]),
    "NIFTY Next 50": calc_cagr(df["NIFTY_NEXT_50_CUM"].iloc[-1]),
}

# -----------------------------
# 4️⃣ Visualization
# -----------------------------
plt.figure(figsize=(10,6))
plt.plot(df["Year"], df["NIFTY_50_CUM"], label="NIFTY 50", marker='o')
plt.plot(df["Year"], df["NIFTY_100_CUM"], label="NIFTY 100", marker='o')
plt.plot(df["Year"], df["NIFTY_NEXT_50_CUM"], label="NIFTY Next 50", marker='o')
plt.title("NIFTY Indices – Cumulative Growth (2006–2020)")
plt.xlabel("Year")
plt.ylabel("Growth (× times initial)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# -----------------------------
# 5️⃣ Yearly Returns Bar Chart
# -----------------------------
df.plot(x="Year", y=["NIFTY_50_TRI", "NIFTY_100_TRI", "NIFTY_NEXT_50_TRI"], kind="bar", figsize=(12,6))
plt.title("Yearly Returns (%) of NIFTY Indices (2006–2020)")
plt.ylabel("Annual Return (%)")
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# -----------------------------
# 6️⃣ Summary Table
# -----------------------------
summary = pd.DataFrame({
    "CAGR (%)": [round(v*100,2) for v in cagr.values()],
    "Final Growth (×)": [
        round(df["NIFTY_100_CUM"].iloc[-1],2),
        round(df["NIFTY_50_CUM"].iloc[-1],2),
        round(df["NIFTY_NEXT_50_CUM"].iloc[-1],2),
    ]
}, index=cagr.keys())

print("📊 Performance Summary (2006–2020):\n")
print(summary)
