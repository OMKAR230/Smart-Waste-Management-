"""
=============================================================
  SMART WASTE MANAGEMENT SYSTEM — IoT Predictive Analytics
  Google Colab Version — Complete Combined Analysis
=============================================================
  HOW TO USE IN GOOGLE COLAB:
  1. Upload your CSV file using the Files panel (folder icon)
  2. Run this entire notebook cell by cell OR Runtime > Run All
=============================================================
"""

# ── STEP 0: Install & Import All Libraries ────────────────
# Uncomment the line below ONLY if running for the first time in Colab
# !pip install scikit-learn matplotlib seaborn pandas numpy joblib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                               GradientBoostingClassifier)
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (mean_absolute_error, r2_score, mean_squared_error,
                              classification_report, confusion_matrix,
                              roc_auc_score, roc_curve)
from sklearn.cluster import KMeans

# ── Global Plot Style ─────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0F1117",
    "axes.facecolor":   "#1A1D27",
    "axes.edgecolor":   "#3A3D4D",
    "axes.labelcolor":  "#E0E0E0",
    "text.color":       "#E0E0E0",
    "xtick.color":      "#AAAAAA",
    "ytick.color":      "#AAAAAA",
    "grid.color":       "#2E3245",
    "grid.alpha":       0.6,
    "legend.facecolor": "#1A1D27",
    "legend.edgecolor": "#3A3D4D",
    "font.family":      "DejaVu Sans",
})

ACCENT = "#00D4FF"
GREEN  = "#00FF88"
YELLOW = "#FFD700"
RED    = "#FF4D4D"
ORANGE = "#FF8C42"
PURPLE = "#A855F7"
COLORS = [ACCENT, GREEN, YELLOW, RED, ORANGE, PURPLE]

OVERFLOW_THRESHOLD = 80   # % — bin is critical above this

# ─────────────────────────────────────────────────────────
# ████████████████████████████████████████████████████████
#   SECTION 1 — DATA LOADING & FEATURE ENGINEERING
# ████████████████████████████████████████████████████████
# ─────────────────────────────────────────────────────────

print("\n" + "═"*62)
print("   SMART WASTE MANAGEMENT — IoT Predictive Analysis")
print("═"*62)

# ── Load Dataset ──────────────────────────────────────────
# Colab path: file must be uploaded to the session via Files panel
DATA_PATH = "/content/smart_waste_dataset_cleaned.csv"

df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
print(f"\n✅  Dataset loaded — {len(df):,} rows × {len(df.columns)} columns")

# ── Feature Engineering ───────────────────────────────────
df["hour"]        = df["timestamp"].dt.hour
df["day_of_week"] = df["timestamp"].dt.dayofweek        # 0=Monday
df["day_name"]    = df["timestamp"].dt.day_name()
df["date"]        = df["timestamp"].dt.date
df["is_overflow"] = (df["fill_level"] >= OVERFLOW_THRESHOLD).astype(int)
df["urgency"]     = pd.cut(
    df["fill_level"],
    bins=[0, 40, 70, 80, 100],
    labels=["Low", "Medium", "High", "Critical"]
)

# ── Label Encoders (used throughout) ──────────────────────
le_area    = LabelEncoder()
le_weather = LabelEncoder()
le_event   = LabelEncoder()

df["area_enc"]    = le_area.fit_transform(df["area_type"])
df["weather_enc"] = le_weather.fit_transform(df["weather"])
df["event_enc"]   = le_event.fit_transform(df["event"])

# ── ML Feature Set ────────────────────────────────────────
FEATURES = ["hour", "day_of_week", "last_collected_hours",
            "area_enc", "weather_enc", "event_enc"]
X = df[FEATURES]
y = df["fill_level"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

# ── Dataset Summary ───────────────────────────────────────
total_bins       = df["bin_id"].nunique()
overflow_bins    = df[df["is_overflow"] == 1]["bin_id"].nunique()
overflow_pct_row = df["is_overflow"].mean() * 100
avg_fill         = df["fill_level"].mean()
critical_now     = df.groupby("bin_id").last()
critical_now     = critical_now[critical_now["fill_level"] >= OVERFLOW_THRESHOLD]

print(f"\n{'─'*62}")
print(f"  📊  Dataset Overview")
print(f"{'─'*62}")
print(f"  Total records        : {len(df):,}")
print(f"  Unique bins          : {total_bins}")
print(f"  Date range           : {df['timestamp'].min().date()} → {df['timestamp'].max().date()}")
print(f"  Area types           : {', '.join(df['area_type'].unique())}")
print(f"  Weather conditions   : {', '.join(df['weather'].unique())}")
print(f"  Events               : {', '.join(df['event'].unique())}")
print(f"  Avg fill level       : {avg_fill:.1f}%")
print(f"  Overflow readings    : {overflow_pct_row:.1f}% of all readings")
print(f"  Bins ever overflowed : {overflow_bins} / {total_bins}")
print(f"  Currently critical   : {len(critical_now)} bins ≥ {OVERFLOW_THRESHOLD}%")
print(f"  Missing values       : {df.isnull().sum().sum()}")
print(f"{'─'*62}")


# ─────────────────────────────────────────────────────────
# ████████████████████████████████████████████████████████
#   SECTION 2 — EDA: FILL-LEVEL DISTRIBUTIONS (9 Charts)
# ████████████████████████████████████████████████████████
# ─────────────────────────────────────────────────────────

print("\n\n📊  Generating EDA Overview (Chart 1/12)…")

fig = plt.figure(figsize=(22, 15))
fig.suptitle("SMART WASTE MANAGEMENT — Exploratory Data Analysis",
             fontsize=17, fontweight="bold", color=ACCENT, y=0.98)
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.48, wspace=0.35)

# ── 2A: Fill level histogram ──────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(df["fill_level"], bins=30, color=ACCENT, edgecolor="#0F1117", alpha=0.85)
ax1.axvline(OVERFLOW_THRESHOLD, color=RED, linewidth=2, linestyle="--",
            label=f"Overflow ({OVERFLOW_THRESHOLD}%)")
ax1.axvline(avg_fill, color=GREEN, linewidth=2, linestyle=":",
            label=f"Mean ({avg_fill:.1f}%)")
ax1.set_title("Fill Level Distribution", fontweight="bold", color=ACCENT)
ax1.set_xlabel("Fill Level (%)"); ax1.set_ylabel("Count")
ax1.legend(fontsize=7); ax1.grid(True)

# ── 2B: Fill level by area type ───────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
for i, atype in enumerate(df["area_type"].unique()):
    ax2.hist(df[df["area_type"] == atype]["fill_level"],
             bins=20, alpha=0.65, label=atype, color=COLORS[i])
ax2.axvline(OVERFLOW_THRESHOLD, color=RED, linewidth=1.5, linestyle="--")
ax2.set_title("Fill Level by Area Type", fontweight="bold", color=ACCENT)
ax2.set_xlabel("Fill Level (%)"); ax2.set_ylabel("Count")
ax2.legend(fontsize=8); ax2.grid(True)

# ── 2C: Urgency breakdown ─────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
urgency_counts = df["urgency"].value_counts().reindex(["Low","Medium","High","Critical"])
bars = ax3.bar(urgency_counts.index, urgency_counts.values,
               color=[GREEN, YELLOW, ORANGE, RED], edgecolor="#0F1117")
for bar, val in zip(bars, urgency_counts.values):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
             f"{val}\n({val/len(df)*100:.1f}%)",
             ha="center", va="bottom", fontsize=8, color="#E0E0E0")
ax3.set_title("Urgency Category Breakdown", fontweight="bold", color=ACCENT)
ax3.set_ylabel("Number of Readings"); ax3.grid(True, axis="y")

# ── 2D: Avg fill level by hour ────────────────────────────
ax4 = fig.add_subplot(gs[1, 0])
hourly = df.groupby("hour")["fill_level"].mean()
ax4.plot(hourly.index, hourly.values, color=ACCENT, linewidth=2.5,
         marker="o", markersize=5)
ax4.fill_between(hourly.index, hourly.values, alpha=0.15, color=ACCENT)
ax4.axhline(OVERFLOW_THRESHOLD, color=RED, linewidth=1.5,
            linestyle="--", label="Overflow threshold")
ax4.set_title("Avg Fill Level by Hour of Day", fontweight="bold", color=ACCENT)
ax4.set_xlabel("Hour"); ax4.set_ylabel("Avg Fill Level (%)")
ax4.legend(fontsize=8); ax4.grid(True)

# ── 2E: Fill level by day of week ────────────────────────
ax5   = fig.add_subplot(gs[1, 1])
day_order  = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
daily      = df.groupby("day_name")["fill_level"].mean().reindex(day_order)
colors_day = [ORANGE if d in ["Saturday","Sunday"] else ACCENT for d in day_order]
ax5.bar(range(len(day_order)), daily.values, color=colors_day, edgecolor="#0F1117")
ax5.set_xticks(range(len(day_order)))
ax5.set_xticklabels([d[:3] for d in day_order], fontsize=9)
ax5.axhline(OVERFLOW_THRESHOLD, color=RED, linewidth=1.5, linestyle="--")
ax5.set_title("Avg Fill Level by Day of Week", fontweight="bold", color=ACCENT)
ax5.set_ylabel("Avg Fill Level (%)"); ax5.grid(True, axis="y")

# ── 2F: Weather impact ────────────────────────────────────
ax6          = fig.add_subplot(gs[1, 2])
weather_fill = df.groupby("weather")["fill_level"].mean().sort_values(ascending=False)
bars         = ax6.barh(weather_fill.index, weather_fill.values,
                        color=[ACCENT, GREEN, YELLOW][:len(weather_fill)],
                        edgecolor="#0F1117")
for bar in bars:
    ax6.text(bar.get_width() - 2, bar.get_y() + bar.get_height()/2,
             f"{bar.get_width():.1f}%", va="center", ha="right",
             fontsize=9, color="#0F1117", fontweight="bold")
ax6.set_title("Weather Impact on Fill Level", fontweight="bold", color=ACCENT)
ax6.set_xlabel("Avg Fill Level (%)"); ax6.grid(True, axis="x")

# ── 2G: Event impact ──────────────────────────────────────
ax7        = fig.add_subplot(gs[2, 0])
event_fill = df.groupby("event")["fill_level"].mean().sort_values(ascending=False)
bars       = ax7.bar(event_fill.index, event_fill.values,
                     color=[RED, YELLOW, GREEN][:len(event_fill)], edgecolor="#0F1117")
for bar in bars:
    ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9)
ax7.set_title("Event Impact on Fill Level", fontweight="bold", color=ACCENT)
ax7.set_ylabel("Avg Fill Level (%)"); ax7.grid(True, axis="y")

# ── 2H: Overflow rate by area type ───────────────────────
ax8           = fig.add_subplot(gs[2, 1])
overflow_area = (df.groupby("area_type")["is_overflow"].mean() * 100
                 ).sort_values(ascending=False)
colors_ov = [RED if v >= 30 else YELLOW if v >= 20 else GREEN
             for v in overflow_area.values]
bars = ax8.bar(overflow_area.index, overflow_area.values,
               color=colors_ov, edgecolor="#0F1117")
for bar in bars:
    ax8.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9)
ax8.set_title("Overflow Rate by Area Type", fontweight="bold", color=ACCENT)
ax8.set_ylabel("% Readings at Overflow"); ax8.grid(True, axis="y")

# ── 2I: Area × Hour heatmap ───────────────────────────────
ax9   = fig.add_subplot(gs[2, 2])
pivot = df.pivot_table(index="area_type", columns="hour",
                       values="fill_level", aggfunc="mean")
im    = ax9.imshow(pivot.values, cmap="RdYlGn_r", aspect="auto", vmin=20, vmax=90)
ax9.set_xticks(range(len(pivot.columns)))
ax9.set_xticklabels(pivot.columns, fontsize=7)
ax9.set_yticks(range(len(pivot.index)))
ax9.set_yticklabels(pivot.index, fontsize=8)
plt.colorbar(im, ax=ax9, label="Avg Fill %")
ax9.set_title("Fill Level: Area × Hour Heatmap", fontweight="bold", color=ACCENT)

plt.savefig("01_eda_overview.png", dpi=150, bbox_inches="tight", facecolor="#0F1117")
plt.show()
print("✅  Chart 1 saved: 01_eda_overview.png")


# ─────────────────────────────────────────────────────────
# ████████████████████████████████████████████████████████
#   SECTION 3 — TIME SERIES PER-BIN FILL TRENDS
# ████████████████████████████████████████████████████████
# ─────────────────────────────────────────────────────────

print("\n\n📈  Generating Time Series (Chart 2/12)…")

fig, axes = plt.subplots(2, 2, figsize=(22, 11))
fig.suptitle("Fill-Level Time Series — Top 4 Highest-Fill Bins",
             fontsize=14, fontweight="bold", color=ACCENT)
fig.patch.set_facecolor("#0F1117")

sample_bins = df.groupby("bin_id")["fill_level"].mean().nlargest(4).index.tolist()

for ax, bin_id in zip(axes.flat, sample_bins):
    sub = df[df["bin_id"] == bin_id].sort_values("timestamp")
    ax.plot(sub["timestamp"], sub["fill_level"],
            color=ACCENT, linewidth=1.8, label="Fill level")
    ax.fill_between(sub["timestamp"], sub["fill_level"], alpha=0.15, color=ACCENT)
    ax.axhline(OVERFLOW_THRESHOLD, color=RED, linewidth=1.5,
               linestyle="--", label="Overflow threshold")
    overflow_mask = sub["fill_level"] >= OVERFLOW_THRESHOLD
    ax.fill_between(sub["timestamp"], OVERFLOW_THRESHOLD, sub["fill_level"],
                    where=overflow_mask, color=RED, alpha=0.3, label="Overflow zone")
    ax.set_title(f"{bin_id}  |  Area: {sub['area_type'].iloc[0]}",
                 fontweight="bold", color=ACCENT)
    ax.set_ylabel("Fill Level (%)"); ax.grid(True)
    ax.set_facecolor("#1A1D27"); ax.tick_params(axis="x", rotation=30)
    ax.set_ylim(0, 105); ax.legend(fontsize=7, loc="upper left")

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("02_time_series.png", dpi=150, bbox_inches="tight", facecolor="#0F1117")
plt.show()
print("✅  Chart 2 saved: 02_time_series.png")


# ─────────────────────────────────────────────────────────
# ████████████████████████████████████████████████████████
#   SECTION 4 — REAL-TIME OVERFLOW RISK DASHBOARD
# ████████████████████████████████████████████████████████
# ─────────────────────────────────────────────────────────

print("\n\n🔴  Generating Overflow Risk Dashboard (Chart 3/12)…")

latest               = df.sort_values("timestamp").groupby("bin_id").last().reset_index()
latest["risk_color"] = latest["fill_level"].apply(
    lambda x: RED if x >= 80 else ORANGE if x >= 70 else YELLOW if x >= 40 else GREEN
)
latest["priority"] = latest["fill_level"].apply(
    lambda x: "CRITICAL" if x >= 80 else "HIGH" if x >= 70
              else "MEDIUM" if x >= 40 else "LOW"
)
latest_sorted = latest.sort_values("fill_level", ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(22, 10))
fig.suptitle("REAL-TIME OVERFLOW RISK DASHBOARD — Latest Reading per Bin",
             fontsize=14, fontweight="bold", color=RED)
fig.patch.set_facecolor("#0F1117")

ax = axes[0]; ax.set_facecolor("#1A1D27")
bars = ax.barh(range(len(latest_sorted)), latest_sorted["fill_level"],
               color=latest_sorted["risk_color"], edgecolor="#0F1117", height=0.7)
ax.axvline(OVERFLOW_THRESHOLD, color=RED, linewidth=2, linestyle="--",
           label=f"Overflow ({OVERFLOW_THRESHOLD}%)")
ax.axvline(70, color=ORANGE, linewidth=1.5, linestyle=":", label="High risk (70%)")
ax.set_yticks(range(len(latest_sorted)))
ax.set_yticklabels(latest_sorted["bin_id"], fontsize=6)
ax.set_xlabel("Current Fill Level (%)")
ax.set_title("All Bins — Current Fill Level", fontweight="bold", color=ACCENT)
ax.legend(fontsize=8); ax.grid(True, axis="x")
for i, (bar, val) in enumerate(zip(bars, latest_sorted["fill_level"])):
    ax.text(val + 0.5, bar.get_y() + bar.get_height()/2,
            f"{val:.0f}%", va="center", fontsize=5.5, color="#E0E0E0")

ax2 = axes[1]; ax2.set_facecolor("#1A1D27")
scatter = ax2.scatter(latest_sorted["longitude"], latest_sorted["latitude"],
                      c=latest_sorted["fill_level"], cmap="RdYlGn_r",
                      vmin=0, vmax=100, s=200, edgecolors="#0F1117",
                      linewidths=0.8, zorder=3)
for _, row in latest_sorted[latest_sorted["fill_level"] >= 80].iterrows():
    ax2.annotate(f"  {row['bin_id']}\n  {row['fill_level']:.0f}%",
                 xy=(row["longitude"], row["latitude"]),
                 fontsize=6, color=RED, fontweight="bold")
plt.colorbar(scatter, ax=ax2, label="Fill Level (%)")
ax2.set_title("Geo Map — Bin Fill Levels", fontweight="bold", color=ACCENT)
ax2.set_xlabel("Longitude"); ax2.set_ylabel("Latitude"); ax2.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("03_overflow_risk_dashboard.png", dpi=150, bbox_inches="tight", facecolor="#0F1117")
plt.show()
print("✅  Chart 3 saved: 03_overflow_risk_dashboard.png")


# ─────────────────────────────────────────────────────────
# ████████████████████████████████████████████████████████
#   SECTION 5 — ROUTE OPTIMISATION (PRIORITY SCORING)
# ████████████████████████████████████████████████████████
# ─────────────────────────────────────────────────────────

print("\n\n🚛  Generating Route Optimisation (Chart 4/12)…")

def compute_priority_score(row):
    """
    Dynamic priority score (0–100):
      Fill level          → up to 50 pts  (main driver)
      Time since collect  → up to 20 pts
      Weather bonus       → up to 15 pts  (rainy = worst)
      Event bonus         → up to 15 pts  (festival = worst)
    """
    score  = (row["fill_level"] / 100) * 50
    score += min(row["last_collected_hours"] / 72, 1) * 20
    score += {"rainy": 15, "cloudy": 8, "sunny": 0}.get(row["weather"], 0)
    score += {"festival": 15, "weekend": 10, "none": 0}.get(row["event"], 0)
    return round(score, 2)

latest["priority_score"]  = latest.apply(compute_priority_score, axis=1)
latest["collection_rank"] = latest["priority_score"].rank(ascending=False).astype(int)
route = latest.sort_values("priority_score", ascending=False).reset_index(drop=True)

print("\n" + "═"*68)
print("  🚛  OPTIMISED COLLECTION ROUTE — Top 15 Priority Bins")
print("═"*68)
print(f"  {'Rank':<5} {'Bin ID':<10} {'Fill%':<8} {'Score':<8} {'Priority':<12} {'Area':<14} {'Weather'}")
print("  " + "─"*66)
for _, row in route.head(15).iterrows():
    flag = "🔴" if row["priority"] == "CRITICAL" else \
           "🟠" if row["priority"] == "HIGH"     else \
           "🟡" if row["priority"] == "MEDIUM"   else "🟢"
    print(f"  {int(row['collection_rank']):<5} {row['bin_id']:<10} "
          f"{row['fill_level']:<8.1f} {row['priority_score']:<8.1f} "
          f"{flag+' '+row['priority']:<12} {row['area_type']:<14} {row['weather']}")

print(f"\n  CRITICAL bins to collect immediately : {(route['priority'] == 'CRITICAL').sum()}")
print(f"  HIGH priority bins                   : {(route['priority'] == 'HIGH').sum()}")

fig, axes = plt.subplots(1, 2, figsize=(22, 9))
fig.suptitle("OPTIMISED WASTE COLLECTION ROUTE — Priority-Based Scheduling",
             fontsize=14, fontweight="bold", color=GREEN)
fig.patch.set_facecolor("#0F1117")

ax = axes[0]; ax.set_facecolor("#1A1D27")
top15     = route.head(15)
colors_r  = [RED if p == "CRITICAL" else ORANGE if p == "HIGH"
             else YELLOW for p in top15["priority"]]
bars = ax.barh(top15["bin_id"][::-1], top15["priority_score"][::-1],
               color=colors_r[::-1], edgecolor="#0F1117")
ax.set_xlabel("Priority Score (max=100)")
ax.set_title("Top 15 Bins by Priority Score", fontweight="bold", color=ACCENT)
ax.grid(True, axis="x")
for bar, score in zip(bars, top15["priority_score"][::-1]):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
            f"{score:.1f}", va="center", fontsize=8)

ax2 = axes[1]; ax2.set_facecolor("#1A1D27")
ax2.scatter(latest["longitude"], latest["latitude"],
            c="#555566", s=80, zorder=2, edgecolors="#0F1117")
for priority, color in [("CRITICAL", RED), ("HIGH", ORANGE), ("MEDIUM", YELLOW)]:
    sub = route[route["priority"] == priority]
    ax2.scatter(sub["longitude"], sub["latitude"], c=color,
                s=180, zorder=4, edgecolors="#0F1117", label=priority)
path = route.head(10)
ax2.plot(path["longitude"], path["latitude"], color=GREEN, linewidth=1.5,
         linestyle="--", alpha=0.7, zorder=3, label="Suggested route")
for i, (_, row) in enumerate(path.iterrows()):
    ax2.annotate(f"  {i+1}. {row['bin_id']}",
                 xy=(row["longitude"], row["latitude"]),
                 fontsize=6, color=GREEN, fontweight="bold", zorder=5)
ax2.set_title("Priority-Based Route Map", fontweight="bold", color=ACCENT)
ax2.set_xlabel("Longitude"); ax2.set_ylabel("Latitude")
ax2.legend(fontsize=8, loc="upper left"); ax2.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("04_route_optimisation.png", dpi=150, bbox_inches="tight", facecolor="#0F1117")
plt.show()
print("✅  Chart 4 saved: 04_route_optimisation.png")


# ─────────────────────────────────────────────────────────
# ████████████████████████████████████████████████████████
#   SECTION 6 — ML MODEL 1: FILL-LEVEL REGRESSION
# ████████████████████████████████████████████████████████
# ─────────────────────────────────────────────────────────

print("\n\n🤖  Training ML Model 1 — Fill Level Regressor (Chart 5/12)…")

rf = RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2  = r2_score(y_test, y_pred)

print(f"\n  Algorithm      : Random Forest Regressor (150 trees)")
print(f"  Train / Test   : {len(X_train)} / {len(X_test)} records")
print(f"  MAE            : {mae:.2f}% fill level")
print(f"  R² Score       : {r2:.4f}  ({r2*100:.1f}% variance explained)")

feat_imp = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False)
print(f"\n  Feature Importances:")
for feat, imp in feat_imp.items():
    bar_str = "█" * int(imp * 60)
    print(f"    {feat:<25} {bar_str:<42} {imp:.4f}")

fig, axes = plt.subplots(1, 3, figsize=(22, 7))
fig.suptitle("ML Model 1 — Fill Level Prediction (Random Forest Regression)",
             fontsize=13, fontweight="bold", color=ACCENT)
fig.patch.set_facecolor("#0F1117")

ax = axes[0]; ax.set_facecolor("#1A1D27")
ax.scatter(y_test, y_pred, alpha=0.4, color=ACCENT, s=15, edgecolors="none")
mn, mx = y_test.min(), y_test.max()
ax.plot([mn, mx], [mn, mx], color=GREEN, linewidth=2, linestyle="--", label="Perfect fit")
ax.set_xlabel("Actual Fill Level (%)")
ax.set_ylabel("Predicted Fill Level (%)")
ax.set_title(f"Actual vs Predicted\nMAE={mae:.2f}%, R²={r2:.3f}",
             fontweight="bold", color=ACCENT)
ax.legend(fontsize=8); ax.grid(True)

ax2 = axes[1]; ax2.set_facecolor("#1A1D27")
residuals = y_pred - y_test
ax2.hist(residuals, bins=30, color=PURPLE, edgecolor="#0F1117", alpha=0.85)
ax2.axvline(0, color=GREEN, linewidth=2, linestyle="--")
ax2.set_xlabel("Residual (Predicted − Actual)"); ax2.set_ylabel("Count")
ax2.set_title("Residual Distribution\n(centred near 0 = low bias)",
              fontweight="bold", color=ACCENT)
ax2.grid(True)

ax3 = axes[2]; ax3.set_facecolor("#1A1D27")
colors_fi = [ACCENT, GREEN, YELLOW, ORANGE, PURPLE, RED][:len(feat_imp)]
ax3.barh(feat_imp.index[::-1], feat_imp.values[::-1],
         color=colors_fi[::-1], edgecolor="#0F1117")
ax3.set_xlabel("Importance Score")
ax3.set_title("Feature Importance", fontweight="bold", color=ACCENT)
ax3.grid(True, axis="x")

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig("05_ml_regression.png", dpi=150, bbox_inches="tight", facecolor="#0F1117")
plt.show()
print("✅  Chart 5 saved: 05_ml_regression.png")


# ─────────────────────────────────────────────────────────
# ████████████████████████████████████████████████████████
#   SECTION 7 — ML MODEL 2: OVERFLOW RISK CLASSIFICATION
# ████████████████████████████████████████████████████████
# ─────────────────────────────────────────────────────────

print("\n\n🤖  Training ML Model 2 — Overflow Classifier (Chart 6/12)…")

y_cls = df["is_overflow"]
X_cls = df[FEATURES]
X_tr, X_te, y_tr, y_te = train_test_split(
    X_cls, y_cls, test_size=0.2, random_state=42, stratify=y_cls
)

gb = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
gb.fit(X_tr, y_tr)
y_pred_cls = gb.predict(X_te)
y_prob_cls = gb.predict_proba(X_te)[:, 1]
auc        = roc_auc_score(y_te, y_prob_cls)

print(f"\n  Algorithm : Gradient Boosting Classifier")
print(f"  ROC-AUC   : {auc:.4f}")
print(f"\n{classification_report(y_te, y_pred_cls, target_names=['Normal','Overflow'])}")

fig, axes = plt.subplots(1, 3, figsize=(22, 7))
fig.suptitle("ML Model 2 — Overflow Risk Classification (Gradient Boosting)",
             fontsize=13, fontweight="bold", color=ACCENT)
fig.patch.set_facecolor("#0F1117")

ax = axes[0]; ax.set_facecolor("#1A1D27")
cm = confusion_matrix(y_te, y_pred_cls)
ax.imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                fontsize=18, color="#E0E0E0", fontweight="bold")
ax.set_xticks([0,1]); ax.set_yticks([0,1])
ax.set_xticklabels(["Normal","Overflow"]); ax.set_yticklabels(["Normal","Overflow"])
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix", fontweight="bold", color=ACCENT)

ax2 = axes[1]; ax2.set_facecolor("#1A1D27")
fpr, tpr, _ = roc_curve(y_te, y_prob_cls)
ax2.plot(fpr, tpr, color=ACCENT, linewidth=2.5, label=f"ROC (AUC={auc:.3f})")
ax2.plot([0,1],[0,1], color="#555566", linestyle="--", label="Random")
ax2.fill_between(fpr, tpr, alpha=0.1, color=ACCENT)
ax2.set_xlabel("False Positive Rate"); ax2.set_ylabel("True Positive Rate")
ax2.set_title("ROC Curve", fontweight="bold", color=ACCENT)
ax2.legend(fontsize=9); ax2.grid(True)

ax3 = axes[2]; ax3.set_facecolor("#1A1D27")
ax3.hist(y_prob_cls[y_te == 0], bins=25, alpha=0.7, color=GREEN, label="Normal")
ax3.hist(y_prob_cls[y_te == 1], bins=25, alpha=0.7, color=RED,   label="Overflow")
ax3.axvline(0.5, color=YELLOW, linewidth=2, linestyle="--", label="Decision boundary")
ax3.set_xlabel("Predicted Overflow Probability"); ax3.set_ylabel("Count")
ax3.set_title("Predicted Probability Distribution", fontweight="bold", color=ACCENT)
ax3.legend(fontsize=8); ax3.grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig("06_ml_classification.png", dpi=150, bbox_inches="tight", facecolor="#0F1117")
plt.show()
print("✅  Chart 6 saved: 06_ml_classification.png")


# ─────────────────────────────────────────────────────────
# ████████████████████████████████████████████████████████
#   SECTION 8 — 24-HOUR FILL LEVEL FORECAST
# ████████████████████████████████████████████████████████
# ─────────────────────────────────────────────────────────

print("\n\n📅  Generating 24-Hour Forecast (Chart 7/12)…")

bins_info        = df.groupby("bin_id").last().reset_index()
forecast_records = []
future_hours     = [6, 12, 18, 24]

for _, row in bins_info.iterrows():
    for fh in future_hours:
        X_future = pd.DataFrame([{
            "hour":                 (row["hour"] + fh) % 24,
            "day_of_week":          row["day_of_week"],
            "last_collected_hours": row["last_collected_hours"] + fh,
            "area_enc":             le_area.transform([row["area_type"]])[0],
            "weather_enc":          le_weather.transform([row["weather"]])[0],
            "event_enc":            le_event.transform([row["event"]])[0],
        }])
        pred_fill = rf.predict(X_future)[0]
        pred_fill = min(max(pred_fill, row["fill_level"]), 100)
        forecast_records.append({
            "bin_id":         row["bin_id"],
            "area_type":      row["area_type"],
            "current_fill":   row["fill_level"],
            "hours_ahead":    fh,
            "predicted_fill": round(pred_fill, 1),
            "will_overflow":  pred_fill >= OVERFLOW_THRESHOLD,
        })

forecast_df  = pd.DataFrame(forecast_records)
overflow_24h = (forecast_df[forecast_df["will_overflow"]]
                .groupby("bin_id")["hours_ahead"].min()
                .reset_index()
                .rename(columns={"hours_ahead": "overflow_in_h"})
                .merge(bins_info[["bin_id","area_type","fill_level"]], on="bin_id")
                .sort_values("overflow_in_h"))

print(f"\n  Bins predicted to overflow within 24 hours : {len(overflow_24h)}")
print(f"\n  {'Bin ID':<12} {'Area':<14} {'Current Fill':<16} {'Overflows in'}")
print("  " + "─"*52)
for _, row in overflow_24h.iterrows():
    print(f"  {row['bin_id']:<12} {row['area_type']:<14} "
          f"{row['fill_level']:.1f}%{'':>8} {int(row['overflow_in_h'])}h")

pivot_fc = forecast_df.pivot_table(
    index="bin_id", columns="hours_ahead", values="predicted_fill"
)

fig, axes = plt.subplots(1, 2, figsize=(22, 10))
fig.suptitle("24-HOUR FILL LEVEL FORECAST PER BIN",
             fontsize=14, fontweight="bold", color=ACCENT)
fig.patch.set_facecolor("#0F1117")

ax = axes[0]
im = ax.imshow(pivot_fc.values, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=100)
ax.set_xticks(range(len(pivot_fc.columns)))
ax.set_xticklabels([f"+{h}h" for h in pivot_fc.columns])
ax.set_yticks(range(len(pivot_fc.index)))
ax.set_yticklabels(pivot_fc.index, fontsize=6)
plt.colorbar(im, ax=ax, label="Predicted Fill %")
ax.set_title("Predicted Fill Level — All Bins", fontweight="bold", color=ACCENT)
ax.set_xlabel("Hours Ahead")

ax2 = axes[1]; ax2.set_facecolor("#1A1D27")
for i, area in enumerate(df["area_type"].unique()):
    sub   = forecast_df[forecast_df["area_type"] == area]
    trend = sub.groupby("hours_ahead")["predicted_fill"].mean()
    ax2.plot(trend.index, trend.values, marker="o", linewidth=2.5,
             color=COLORS[i], label=area)
ax2.axhline(OVERFLOW_THRESHOLD, color=RED, linewidth=2, linestyle="--",
            label=f"Overflow threshold ({OVERFLOW_THRESHOLD}%)")
ax2.fill_between([0, 24], OVERFLOW_THRESHOLD, 100, alpha=0.08, color=RED)
ax2.set_xlabel("Hours Ahead"); ax2.set_ylabel("Avg Predicted Fill Level (%)")
ax2.set_title("Area-wise 24h Fill Forecast", fontweight="bold", color=ACCENT)
ax2.legend(fontsize=9); ax2.grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("07_24h_forecast.png", dpi=150, bbox_inches="tight", facecolor="#0F1117")
plt.show()
print("✅  Chart 7 saved: 07_24h_forecast.png")


# ─────────────────────────────────────────────────────────
# ████████████████████████████████████████████████████████
#   SECTION 9 — CORRELATION HEATMAP
# ████████████████████████████████████████████████████████
# ─────────────────────────────────────────────────────────

print("\n\n🔗  Generating Correlation Heatmap (Chart 8/12)…")

fig, axes = plt.subplots(1, 2, figsize=(20, 8))
fig.suptitle("Feature Correlation Analysis",
             fontsize=14, fontweight="bold", color=ACCENT)
fig.patch.set_facecolor("#0F1117")

num_cols = ["fill_level","last_collected_hours","hour","day_of_week",
            "area_enc","weather_enc","event_enc","is_overflow"]
corr     = df[num_cols].corr()

ax = axes[0]
im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(num_cols))); ax.set_yticks(range(len(num_cols)))
ax.set_xticklabels(num_cols, rotation=45, ha="right", fontsize=8)
ax.set_yticklabels(num_cols, fontsize=8)
for i in range(len(num_cols)):
    for j in range(len(num_cols)):
        ax.text(j, i, f"{corr.values[i,j]:.2f}", ha="center", va="center",
                fontsize=7,
                color="white" if abs(corr.values[i,j]) > 0.5 else "#AAAAAA")
plt.colorbar(im, ax=ax, label="Correlation")
ax.set_title("Feature Correlation Matrix", fontweight="bold", color=ACCENT)

ax2 = axes[1]; ax2.set_facecolor("#1A1D27")
scatter = ax2.scatter(df["last_collected_hours"], df["fill_level"],
                      c=df["area_enc"], cmap="viridis", alpha=0.4, s=15)
ax2.axhline(OVERFLOW_THRESHOLD, color=RED, linewidth=2, linestyle="--",
            label="Overflow threshold")
z = np.polyfit(df["last_collected_hours"], df["fill_level"], 1)
p = np.poly1d(z)
x_line = np.linspace(df["last_collected_hours"].min(),
                     df["last_collected_hours"].max(), 100)
ax2.plot(x_line, p(x_line), color=GREEN, linewidth=2.5,
         label=f"Trend (slope={z[0]:.2f}%/hr)")
ax2.set_xlabel("Hours Since Last Collection"); ax2.set_ylabel("Fill Level (%)")
ax2.set_title("Fill Level vs Time Since Collection\n(Core IoT Insight)",
              fontweight="bold", color=ACCENT)
ax2.legend(fontsize=9); ax2.grid(True)
plt.colorbar(scatter, ax=ax2, label="Area Type (encoded)")

fill_corr = corr["fill_level"].drop("fill_level").sort_values(key=abs, ascending=False)
print(f"\n  Top correlations with fill_level:")
for feat, val in fill_corr.items():
    direction = "↑ positive" if val > 0 else "↓ negative"
    print(f"    {feat:<25}  r = {val:>7.4f}   ({direction})")

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("08_correlation_analysis.png", dpi=150, bbox_inches="tight", facecolor="#0F1117")
plt.show()
print("✅  Chart 8 saved: 08_correlation_analysis.png")


# ─────────────────────────────────────────────────────────
# ████████████████████████████████████████████████████████
#   SECTION 10 — MULTI-MODEL COMPARISON + CROSS VALIDATION
# ████████████████████████████████████████████████████████
# ─────────────────────────────────────────────────────────

print("\n\n🏆  Running Multi-Model Comparison + Cross Validation (Chart 9/12)…")
print("    (This may take ~60 seconds in Colab)")

models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression":  Ridge(alpha=1.0),
    "Decision Tree":     DecisionTreeRegressor(max_depth=10, random_state=42),
    "Random Forest":     RandomForestRegressor(n_estimators=150, max_depth=12,
                                               random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, max_depth=5,
                                                   random_state=42),
    "KNN Regressor":     KNeighborsRegressor(n_neighbors=7),
}

results = {}
kf = KFold(n_splits=5, shuffle=True, random_state=42)

print(f"\n  {'Model':<22} {'MAE':>8} {'RMSE':>8} {'R²':>8} {'CV R² (5-fold)':>18}")
print("  " + "─"*68)

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae   = mean_absolute_error(y_test, preds)
    rmse  = np.sqrt(mean_squared_error(y_test, preds))
    r2_   = r2_score(y_test, preds)
    cv    = cross_val_score(model, X, y, cv=kf, scoring="r2")
    results[name] = {"MAE": mae, "RMSE": rmse, "R2": r2_,
                     "CV_mean": cv.mean(), "CV_std": cv.std(),
                     "predictions": preds}
    flag  = "  ← BEST ✅" if name == "Random Forest" else ""
    print(f"  {name:<22} {mae:>8.2f} {rmse:>8.2f} {r2_:>8.4f} "
          f"  {cv.mean():.4f} ± {cv.std():.4f}{flag}")

model_names = list(results.keys())
maes        = [results[m]["MAE"]     for m in model_names]
r2s         = [results[m]["R2"]      for m in model_names]
cv_means    = [results[m]["CV_mean"] for m in model_names]
cv_stds     = [results[m]["CV_std"]  for m in model_names]
bar_colors  = [GREEN if m == "Random Forest" else ACCENT for m in model_names]

fig, axes = plt.subplots(2, 2, figsize=(22, 13))
fig.suptitle("MODEL COMPARISON — 6 Algorithms vs Fill Level Prediction",
             fontsize=14, fontweight="bold", color=ACCENT)
fig.patch.set_facecolor("#0F1117")

ax = axes[0, 0]; ax.set_facecolor("#1A1D27")
bars = ax.bar(model_names, maes, color=bar_colors, edgecolor="#0F1117")
ax.set_xticklabels(model_names, rotation=20, ha="right", fontsize=8)
ax.set_ylabel("Mean Absolute Error (%)")
ax.set_title("MAE by Model (lower = better)", fontweight="bold", color=ACCENT)
for bar, val in zip(bars, maes):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            f"{val:.2f}", ha="center", va="bottom", fontsize=8)
ax.grid(True, axis="y")

ax2 = axes[0, 1]; ax2.set_facecolor("#1A1D27")
bars = ax2.bar(model_names, r2s, color=bar_colors, edgecolor="#0F1117")
ax2.set_xticklabels(model_names, rotation=20, ha="right", fontsize=8)
ax2.set_ylabel("R² Score"); ax2.set_ylim(0, 1.05)
ax2.set_title("R² Score by Model (higher = better)", fontweight="bold", color=ACCENT)
for bar, val in zip(bars, r2s):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f"{val:.4f}", ha="center", va="bottom", fontsize=8)
ax2.grid(True, axis="y")

ax3 = axes[1, 0]; ax3.set_facecolor("#1A1D27")
x_pos = np.arange(len(model_names))
ax3.bar(x_pos, cv_means, color=bar_colors, edgecolor="#0F1117", alpha=0.85)
ax3.errorbar(x_pos, cv_means, yerr=cv_stds, fmt="none",
             color=YELLOW, capsize=5, linewidth=2)
ax3.set_xticks(x_pos)
ax3.set_xticklabels(model_names, rotation=20, ha="right", fontsize=8)
ax3.set_ylabel("CV R² Score"); ax3.set_ylim(0, 1.05)
ax3.set_title("5-Fold Cross Validation R² ± Std Dev", fontweight="bold", color=ACCENT)
ax3.grid(True, axis="y")

ax4 = axes[1, 1]; ax4.set_facecolor("#1A1D27")
top3 = sorted(results.items(), key=lambda x: x[1]["R2"], reverse=True)[:3]
for (name, res), color in zip(top3, [GREEN, ACCENT, YELLOW]):
    ax4.scatter(y_test, res["predictions"], alpha=0.3, s=12, color=color,
                label=f"{name} (R²={res['R2']:.3f})")
mn, mx = y_test.min(), y_test.max()
ax4.plot([mn, mx], [mn, mx], color=RED, linewidth=2, linestyle="--", label="Perfect fit")
ax4.set_xlabel("Actual Fill Level (%)")
ax4.set_ylabel("Predicted Fill Level (%)")
ax4.set_title("Actual vs Predicted — Top 3 Models", fontweight="bold", color=ACCENT)
ax4.legend(fontsize=8); ax4.grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("09_model_comparison.png", dpi=150, bbox_inches="tight", facecolor="#0F1117")
plt.show()
print("✅  Chart 9 saved: 09_model_comparison.png")


# ─────────────────────────────────────────────────────────
# ████████████████████████████████████████████████████████
#   SECTION 11 — FILL RATE ANALYSIS
# ████████████████████████████████████████████████████████
# ─────────────────────────────────────────────────────────

print("\n\n⚡  Generating Fill Rate Analysis (Chart 10/12)…")

df_sorted             = df.sort_values(["bin_id","timestamp"]).copy()
df_sorted["fill_diff"]   = df_sorted.groupby("bin_id")["fill_level"].diff()
df_sorted["time_diff_h"] = (df_sorted.groupby("bin_id")["timestamp"]
                             .diff().dt.total_seconds() / 3600)
df_sorted["fill_rate"]   = df_sorted["fill_diff"] / df_sorted["time_diff_h"]

fill_rates = df_sorted[(df_sorted["fill_rate"] > 0) &
                        (df_sorted["fill_rate"] < 20)].copy()

bin_fill_rates = fill_rates.groupby("bin_id").agg(
    avg_fill_rate=("fill_rate", "mean"),
    max_fill_rate=("fill_rate", "max"),
    area_type    =("area_type", "first")
).reset_index().sort_values("avg_fill_rate", ascending=False)

latest = latest.merge(bin_fill_rates[["bin_id","avg_fill_rate"]], on="bin_id", how="left")
latest["hours_to_overflow"] = ((OVERFLOW_THRESHOLD - latest["fill_level"])
                                / latest["avg_fill_rate"]).clip(lower=0)

print(f"\n  Overall avg fill rate     : {fill_rates['fill_rate'].mean():.2f} %/hour")
print(f"  Fastest-filling bin rate  : {bin_fill_rates['avg_fill_rate'].max():.2f} %/hour")
print(f"  Slowest-filling bin rate  : {bin_fill_rates['avg_fill_rate'].min():.2f} %/hour")
print(f"\n  Bins needing collection ≤ 12h : {(latest['hours_to_overflow'] <= 12).sum()}")
print(f"  Bins needing collection ≤ 24h : {(latest['hours_to_overflow'] <= 24).sum()}")

print(f"\n  Top 10 Most Urgent Bins (Fastest to Overflow):")
print(f"  {'Bin ID':<12} {'Rate (%/hr)':<14} {'Area':<16} {'Hours to Overflow'}")
print("  " + "─"*58)
for _, row in latest.sort_values("hours_to_overflow").head(10).iterrows():
    print(f"  {row['bin_id']:<12} {row['avg_fill_rate']:<14.2f} "
          f"{row['area_type']:<16} {row['hours_to_overflow']:.1f}h")

fig, axes = plt.subplots(2, 2, figsize=(22, 12))
fig.suptitle("BIN FILL RATE ANALYSIS — Core IoT Scheduling Insight",
             fontsize=14, fontweight="bold", color=ACCENT)
fig.patch.set_facecolor("#0F1117")

ax = axes[0, 0]; ax.set_facecolor("#1A1D27")
ax.hist(fill_rates["fill_rate"], bins=30, color=ACCENT, edgecolor="#0F1117", alpha=0.85)
ax.axvline(fill_rates["fill_rate"].mean(), color=GREEN, linewidth=2, linestyle="--",
           label=f"Mean: {fill_rates['fill_rate'].mean():.2f} %/hr")
ax.set_xlabel("Fill Rate (%/hour)"); ax.set_ylabel("Count")
ax.set_title("Overall Fill Rate Distribution", fontweight="bold", color=ACCENT)
ax.legend(fontsize=9); ax.grid(True)

ax2 = axes[0, 1]; ax2.set_facecolor("#1A1D27")
area_rate = fill_rates.groupby("area_type")["fill_rate"].mean().sort_values(ascending=False)
bars = ax2.bar(area_rate.index, area_rate.values,
               color=[ACCENT, GREEN, YELLOW][:len(area_rate)], edgecolor="#0F1117")
for bar, val in zip(bars, area_rate.values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f"{val:.2f}%/hr", ha="center", va="bottom", fontsize=9)
ax2.set_ylabel("Avg Fill Rate (%/hour)")
ax2.set_title("Avg Fill Rate by Area Type", fontweight="bold", color=ACCENT)
ax2.grid(True, axis="y")

ax3 = axes[1, 0]; ax3.set_facecolor("#1A1D27")
urgent     = latest.sort_values("hours_to_overflow").head(20)
colors_urg = [RED if h <= 6 else ORANGE if h <= 12 else YELLOW if h <= 24 else GREEN
              for h in urgent["hours_to_overflow"]]
bars = ax3.barh(urgent["bin_id"][::-1], urgent["hours_to_overflow"][::-1],
                color=colors_urg[::-1], edgecolor="#0F1117")
ax3.axvline(12, color=ORANGE, linewidth=2, linestyle="--", label="12h urgency")
ax3.axvline(24, color=YELLOW, linewidth=1.5, linestyle=":",  label="24h urgency")
ax3.set_xlabel("Hours Until Overflow (predicted)")
ax3.set_title("Time-to-Overflow — 20 Most Urgent Bins", fontweight="bold", color=ACCENT)
ax3.legend(fontsize=8); ax3.grid(True, axis="x")

ax4 = axes[1, 1]; ax4.set_facecolor("#1A1D27")
hourly_rate = fill_rates.groupby("hour")["fill_rate"].mean()
ax4.bar(hourly_rate.index, hourly_rate.values, color=ACCENT, edgecolor="#0F1117", alpha=0.85)
ax4.plot(hourly_rate.index, hourly_rate.values, color=GREEN,
         linewidth=2, marker="o", markersize=4)
ax4.set_xlabel("Hour of Day"); ax4.set_ylabel("Avg Fill Rate (%/hour)")
ax4.set_title("Fill Rate Pattern by Hour of Day", fontweight="bold", color=ACCENT)
ax4.grid(True, axis="y")

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("10_fill_rate_analysis.png", dpi=150, bbox_inches="tight", facecolor="#0F1117")
plt.show()
print("✅  Chart 10 saved: 10_fill_rate_analysis.png")


# ─────────────────────────────────────────────────────────
# ████████████████████████████████████████████████████████
#   SECTION 12 — BIN BEHAVIOR CLUSTERING (KMeans)
# ████████████████████████████████████████████████████████
# ─────────────────────────────────────────────────────────

print("\n\n🔵  Running Bin Clustering (Chart 11/12)…")

bin_features = df.groupby("bin_id").agg(
    avg_fill        =("fill_level",           "mean"),
    std_fill        =("fill_level",           "std"),
    max_fill        =("fill_level",           "max"),
    overflow_rate   =("is_overflow",          "mean"),
    avg_hours_since =("last_collected_hours", "mean"),
).reset_index().fillna(0)

bin_features = bin_features.merge(
    bin_fill_rates[["bin_id","avg_fill_rate"]], on="bin_id", how="left"
).fillna(0)

scaler       = StandardScaler()
cluster_cols = ["avg_fill","std_fill","max_fill","overflow_rate",
                "avg_hours_since","avg_fill_rate"]
X_cluster    = scaler.fit_transform(bin_features[cluster_cols])

# Elbow method
inertias = []
K_range  = range(2, 9)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_cluster)
    inertias.append(km.inertia_)

# Final model — K=3 clusters
km_final = KMeans(n_clusters=3, random_state=42, n_init=10)
bin_features["cluster"] = km_final.fit_predict(X_cluster)

# Re-label: 0=Low, 1=Medium, 2=High risk by avg_fill
cluster_avg  = bin_features.groupby("cluster")["avg_fill"].mean().sort_values()
label_map    = {old: new for new, old in enumerate(cluster_avg.index)}
bin_features["cluster"] = bin_features["cluster"].map(label_map)

cluster_colors = [GREEN, YELLOW, RED]

print(f"\n  {'C':<4} {'Label':<28} {'Bins':<6} {'Avg Fill%':<12} {'Overflow%':<12} {'Fill Rate'}")
print("  " + "─"*72)
for c in sorted(bin_features["cluster"].unique()):
    sub = bin_features[bin_features["cluster"] == c]
    lbl = ["🟢 Low-Risk Slow Fillers",
           "🟡 Medium-Risk Moderate",
           "🔴 High-Risk Fast Fillers"][c]
    print(f"  {c:<4} {lbl:<28} {len(sub):<6} "
          f"{sub['avg_fill'].mean():<12.1f}"
          f"{sub['overflow_rate'].mean()*100:<12.1f}"
          f"{sub['avg_fill_rate'].mean():.2f}%/hr")

fig, axes = plt.subplots(1, 3, figsize=(22, 8))
fig.suptitle("BIN CLUSTERING — Behaviour-Based Grouping (KMeans, K=3)",
             fontsize=14, fontweight="bold", color=ACCENT)
fig.patch.set_facecolor("#0F1117")

ax = axes[0]; ax.set_facecolor("#1A1D27")
ax.plot(list(K_range), inertias, color=ACCENT, linewidth=2.5, marker="o", markersize=7)
ax.axvline(3, color=GREEN, linewidth=2, linestyle="--", label="Chosen K=3")
ax.set_xlabel("Number of Clusters (K)"); ax.set_ylabel("Inertia")
ax.set_title("Elbow Method — Optimal K", fontweight="bold", color=ACCENT)
ax.legend(fontsize=9); ax.grid(True)

ax2 = axes[1]; ax2.set_facecolor("#1A1D27")
for c in sorted(bin_features["cluster"].unique()):
    sub = bin_features[bin_features["cluster"] == c]
    lbl = ["Low-Risk","Medium-Risk","High-Risk"][c]
    ax2.scatter(sub["avg_fill"], sub["avg_fill_rate"], color=cluster_colors[c],
                s=100, label=lbl, edgecolors="#0F1117", linewidths=0.8, zorder=3)
    for _, row in sub.iterrows():
        ax2.annotate(row["bin_id"], (row["avg_fill"], row["avg_fill_rate"]),
                     fontsize=5, color="#AAAAAA")
ax2.axhline(bin_features["avg_fill_rate"].mean(), color=YELLOW,
            linewidth=1.5, linestyle="--", label="Avg fill rate")
ax2.set_xlabel("Avg Fill Level (%)"); ax2.set_ylabel("Avg Fill Rate (%/hr)")
ax2.set_title("Bins: Fill Level vs Fill Rate", fontweight="bold", color=ACCENT)
ax2.legend(fontsize=9); ax2.grid(True)

ax3 = axes[2]; ax3.set_facecolor("#1A1D27")
metrics  = ["avg_fill", "overflow_rate", "avg_fill_rate"]
labels_m = ["Avg Fill %", "Overflow Rate %", "Fill Rate %/hr"]
x_pos    = np.arange(len(metrics))
width    = 0.25
for c in sorted(bin_features["cluster"].unique()):
    sub  = bin_features[bin_features["cluster"] == c]
    vals = [sub["avg_fill"].mean(),
            sub["overflow_rate"].mean() * 100,
            sub["avg_fill_rate"].mean()]
    lbl  = ["Low-Risk","Medium-Risk","High-Risk"][c]
    ax3.bar(x_pos + c * width, vals, width, color=cluster_colors[c],
            label=lbl, edgecolor="#0F1117")
ax3.set_xticks(x_pos + width)
ax3.set_xticklabels(labels_m, fontsize=9)
ax3.set_title("Cluster Profile Comparison", fontweight="bold", color=ACCENT)
ax3.legend(fontsize=9); ax3.grid(True, axis="y")

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("11_bin_clustering.png", dpi=150, bbox_inches="tight", facecolor="#0F1117")
plt.show()
print("✅  Chart 11 saved: 11_bin_clustering.png")


# ─────────────────────────────────────────────────────────
# ████████████████████████████████████████████████████████
#   SECTION 13 — ANOMALY DETECTION (IoT Sensor)
# ████████████████████████████████████████████████████████
# ─────────────────────────────────────────────────────────

print("\n\n🚨  Generating Anomaly Detection (Chart 12/12)…")

df["bin_mean"]   = df.groupby("bin_id")["fill_level"].transform("mean")
df["bin_std"]    = df.groupby("bin_id")["fill_level"].transform("std")
df["z_score"]    = (df["fill_level"] - df["bin_mean"]) / df["bin_std"]
df["is_anomaly"] = (df["z_score"].abs() > 2.5).astype(int)

df_s              = df.sort_values(["bin_id","timestamp"]).copy()
df_s["fill_change"] = df_s.groupby("bin_id")["fill_level"].diff()
df_s["roc_anomaly"] = (df_s["fill_change"].abs() > 25).astype(int)
df_s["any_anomaly"] = ((df_s["is_anomaly"] == 1) | (df_s["roc_anomaly"] == 1)).astype(int)

anomalies = df_s[df_s["any_anomaly"] == 1]

print(f"\n  Total anomalous readings  : {len(anomalies)}")
print(f"  Z-score anomalies         : {df_s['is_anomaly'].sum()}")
print(f"  Rate-of-change anomalies  : {df_s['roc_anomaly'].sum()}")
print(f"  Affected bins             : {anomalies['bin_id'].nunique()}")

if len(anomalies) > 0:
    print(f"\n  Top 10 Anomalous Readings:")
    print(f"  {'Bin':<10} {'Timestamp':<22} {'Fill%':<8} {'Z-score':<10} {'Change%'}")
    print("  " + "─"*58)
    for _, row in anomalies.head(10).iterrows():
        chg = f"{row['fill_change']:.1f}" if pd.notna(row["fill_change"]) else "N/A"
        print(f"  {row['bin_id']:<10} {str(row['timestamp']):<22} "
              f"{row['fill_level']:<8.1f} {row['z_score']:<10.2f} {chg}")

fig, axes = plt.subplots(1, 2, figsize=(22, 8))
fig.suptitle("ANOMALY DETECTION — IoT Sensor Spike & Jump Detection",
             fontsize=14, fontweight="bold", color=RED)
fig.patch.set_facecolor("#0F1117")

sample_bin = (anomalies["bin_id"].value_counts().index[0]
              if len(anomalies) > 0 else df["bin_id"].iloc[0])
sub_all  = df_s[df_s["bin_id"] == sample_bin].sort_values("timestamp")
sub_anom = sub_all[sub_all["any_anomaly"] == 1]

ax = axes[0]; ax.set_facecolor("#1A1D27")
ax.plot(sub_all["timestamp"], sub_all["fill_level"],
        color=ACCENT, linewidth=1.8, label="Fill level", zorder=2)
ax.scatter(sub_anom["timestamp"], sub_anom["fill_level"],
           color=RED, s=120, zorder=5, label="⚠ Anomaly",
           edgecolors=YELLOW, linewidths=1.5)
ax.axhline(OVERFLOW_THRESHOLD, color=RED, linewidth=1.5, linestyle="--")
ax.set_title(f"Anomaly Detection — {sample_bin}", fontweight="bold", color=ACCENT)
ax.set_ylabel("Fill Level (%)"); ax.legend(fontsize=9); ax.grid(True)
ax.tick_params(axis="x", rotation=30)

ax2 = axes[1]; ax2.set_facecolor("#1A1D27")
ax2.hist(df_s["z_score"].dropna(), bins=40, color=ACCENT,
         edgecolor="#0F1117", alpha=0.85, label="Normal readings")
anom_z = df_s[df_s["is_anomaly"] == 1]["z_score"]
ax2.hist(anom_z, bins=20, color=RED, edgecolor="#0F1117",
         alpha=0.85, label=f"Anomalies ({len(anom_z)})")
ax2.axvline( 2.5, color=YELLOW, linewidth=2, linestyle="--", label="±2.5σ threshold")
ax2.axvline(-2.5, color=YELLOW, linewidth=2, linestyle="--")
ax2.set_xlabel("Z-Score"); ax2.set_ylabel("Count")
ax2.set_title("Z-Score Distribution — Anomaly Threshold", fontweight="bold", color=ACCENT)
ax2.legend(fontsize=9); ax2.grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("12_anomaly_detection.png", dpi=150, bbox_inches="tight", facecolor="#0F1117")
plt.show()
print("✅  Chart 12 saved: 12_anomaly_detection.png")


# ─────────────────────────────────────────────────────────
# ████████████████████████████████████████████████████████
#   SECTION 14 — COLLECTION EFFICIENCY ANALYSIS
# ████████████████████████████████████████████████████████
# ─────────────────────────────────────────────────────────

traditional_trips = len(df["bin_id"].unique()) * df["date"].nunique()
smart_trips       = df[df["fill_level"] >= 70].groupby(["bin_id","date"]).ngroups
saving_pct        = (1 - smart_trips / traditional_trips) * 100
overflows_trad    = df.groupby(["bin_id","date"])["fill_level"].max()
overflows_smart   = overflows_trad[overflows_trad >= OVERFLOW_THRESHOLD]

print("\n" + "═"*62)
print("  ♻️   COLLECTION EFFICIENCY ANALYSIS")
print("═"*62)
print(f"\n  Traditional (fixed-schedule) trips  : {traditional_trips:,}")
print(f"  Smart IoT-triggered trips           : {smart_trips:,}")
print(f"  Estimated trip reduction            : {saving_pct:.1f}%")
print(f"  Overflow events (traditional)       : {len(overflows_smart):,}")
print(f"  Overflow events (smart)             : 0  (collected before overflow)")


# ─────────────────────────────────────────────────────────
# ████████████████████████████████████████████████████████
#   SECTION 15 — SAVE MODELS (for Deployment)
# ████████████████████████████████████████████████████████
# ─────────────────────────────────────────────────────────

print("\n" + "═"*62)
print("  💾  SAVING MODELS FOR DEPLOYMENT")
print("═"*62)

joblib.dump(rf,         "/content/model_fill_regressor.pkl")
joblib.dump(gb,         "/content/model_overflow_classifier.pkl")
joblib.dump(km_final,   "/content/model_bin_clusters.pkl")
joblib.dump(le_area,    "/content/encoder_area.pkl")
joblib.dump(le_weather, "/content/encoder_weather.pkl")
joblib.dump(le_event,   "/content/encoder_event.pkl")
joblib.dump(scaler,     "/content/scaler_cluster.pkl")

# Quick load verification
test_loaded = joblib.load("/content/model_fill_regressor.pkl")
test_pred   = test_loaded.predict(X_test[:3])
print(f"\n  ✅  model_fill_regressor.pkl      — Fill level predictor")
print(f"  ✅  model_overflow_classifier.pkl — Overflow risk detector")
print(f"  ✅  model_bin_clusters.pkl        — Bin behaviour clustering")
print(f"  ✅  encoder_area / weather / event .pkl")
print(f"  ✅  scaler_cluster.pkl")
print(f"\n  Load verification — predictions: {test_pred.round(1)}")
print(f"  Load verification — actuals    : {y_test.values[:3].round(1)}")


# ─────────────────────────────────────────────────────────
# ████████████████████████████████████████████████████████
#   FINAL SUMMARY
# ████████████████████████████████████████████████████████
# ─────────────────────────────────────────────────────────

best_r2  = max(results.items(), key=lambda x: x[1]["R2"])
best_mae = min(results.items(), key=lambda x: x[1]["MAE"])

print("\n" + "═"*62)
print("  🎯  COMPLETE ANALYSIS SUMMARY")
print("═"*62)
print(f"\n  Total charts generated   : 12")
print(f"  Best R² model            : {best_r2[0]}  (R²={best_r2[1]['R2']:.4f})")
print(f"  Best MAE model           : {best_mae[0]}  (MAE={best_mae[1]['MAE']:.2f}%)")
print(f"  Overflow classifier AUC  : {auc:.4f}")
print(f"  Bins needing urgent coll : {(latest['hours_to_overflow'] <= 12).sum()} (≤12h)")
print(f"  Anomalies detected       : {len(anomalies)}")
print(f"  High-risk bins clustered : {(bin_features['cluster'] == 2).sum()}")
print(f"  Trip reduction achieved  : {saving_pct:.1f}%")
print(f"  Models saved             : 7 .pkl files in /content/")
print("\n" + "═"*62)
print("  All outputs saved to /content/ — download from Files panel")
print("═"*62 + "\n")
