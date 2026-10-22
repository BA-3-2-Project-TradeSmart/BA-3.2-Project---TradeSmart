"""
TradeSmart AI - time series analysis: how a car's value changes over time.

The Autotrader dataset is one snapshot (listings scraped in April 2023),
not a history of prices recorded month by month. But each listing has a
model year, so the same model is listed at several ages at once - a 2023,
a 2021 and a 2019 Polo Vivo Trendline side by side. Comparing those gives
the value-over-time curve a true price history would show: how much of
its value a car keeps after 1, 2, 3... years.

Why compare the same model, not overall averages: the median price of all
2021 cars (R349,900) is LOWER than that of all 2020 cars (R379,750) - not
because 2021 cars are worth less, but because a different mix of models
was listed for each year. Comparing each model only with itself removes
that mix effect.

Method:
  1. For each make + model, take the median price per model year.
  2. Divide by that model's newest-year median -> "value retained" after
     1, 2, ... years older.
  3. Fit exponential depreciation, value(t) = value(0) * e^(-r t), by least
     squares on log(value retained); the annual depreciation rate is
     1 - e^(-r).
  4. Repeat per body type (do bakkies hold value better than sedans?).
  5. Forecast an example car's value for the years after the snapshot,
     using the fitted rate.

Limitation (state it in the write-up): the curve covers ages 0-5 only,
because the data only holds 2018-2023 model years; the forecast assumes
depreciation carries on at the same rate. Real price histories (the same
listings scraped again over time) would allow a proper time series model
such as ARIMA or Prophet - see the documentation's improvement plan.

Writes outputs/time_series_analysis.json and three figures in
outputs/figures/. Run from the src folder:
    python time_series_analysis.py
"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from features import load_raw, PROJECT_ROOT

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
SNAPSHOT_YEAR = 2023            # listings were scraped in April 2023
MIN_PAIRS_PER_BODY_TYPE = 30    # body types with fewer model/year pairs are too thin to fit
FORECAST_YEARS = 5
EXAMPLE = ("Volkswagen", "Polo Vivo Hatch 1.4 Trendline", 2019)

# Chart styling (one series per chart, so one hue; fits are recessive grey)
BLUE = "#2a78d6"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"


def value_retained(df):
    """One row per (model, older model year): the median price for that year
    as a share of the same model's newest-year median price."""
    per_year = (df.groupby(["Brand", "Model", "Year"])
                  .agg(price=("Price", "median"), body=("Body Type", "first"))
                  .reset_index())
    newest = per_year.loc[per_year.groupby(["Brand", "Model"]).Year.idxmax(),
                          ["Brand", "Model", "Year", "price"]]
    newest = newest.rename(columns={"Year": "newest_year", "price": "newest_price"})
    pairs = per_year.merge(newest, on=["Brand", "Model"])
    pairs["years_older"] = pairs.newest_year - pairs.Year
    pairs["retained"] = pairs.price / pairs.newest_price
    return pairs[pairs.years_older > 0]


def fit_rate(years_older, retained):
    """Least-squares fit of log(retained) = -r * years_older (through the
    origin: a car keeps 100% of its value at 0 years older)."""
    x = np.asarray(years_older, dtype=float)
    y = np.log(np.clip(np.asarray(retained, dtype=float), 0.05, None))
    return float(-np.sum(x * y) / np.sum(x * x))


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=INK, pad=12)
    ax.set_xlabel(xlabel, color=INK_SECONDARY, fontsize=10)
    ax.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=10)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9, length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def main():
    df = load_raw()
    pairs = value_retained(df)

    by_age = pairs.groupby("years_older").retained.agg(["median", "count"])
    rate = fit_rate(pairs.years_older, pairs.retained)
    annual = 1 - np.exp(-rate)

    # Fit quality: how close the fitted curve is to the observed medians
    fitted_at_ages = np.exp(-rate * by_age.index.values)
    max_gap = float(np.max(np.abs(fitted_at_ages - by_age["median"].values)))

    body = []
    for body_type, group in pairs.groupby("body"):
        if len(group) >= MIN_PAIRS_PER_BODY_TYPE:
            r = fit_rate(group.years_older, group.retained)
            body.append({"body_type": body_type, "annual_depreciation": round(1 - np.exp(-r), 4),
                         "model_year_pairs": int(len(group))})
    body.sort(key=lambda b: b["annual_depreciation"])

    # Mix effect: raw median price per model year, for comparison
    raw_by_year = df.groupby("Year").Price.median()

    # Example forecast
    make, model, year = EXAMPLE
    ex = df[(df.Brand == make) & (df.Model == model) & (df.Year == year)]
    start_value = float(ex.Price.median())
    forecast = [{"year": SNAPSHOT_YEAR + k,
                 "projected_value": round(start_value * np.exp(-rate * k))}
                for k in range(0, FORECAST_YEARS + 1)]

    report = {
        "method": "Like-for-like value retention per make+model across model years "
                  "(April 2023 snapshot), exponential fit value(t) = value(0) * e^(-r t)",
        "model_year_pairs_used": int(len(pairs)),
        "models_with_several_years": int(pairs[["Brand", "Model"]].drop_duplicates().shape[0]),
        "value_retained_by_years_older": {
            int(k): {"median": round(float(v["median"]), 4), "pairs": int(v["count"])}
            for k, v in by_age.iterrows()
        },
        "fitted_rate_r": round(rate, 4),
        "annual_depreciation": round(float(annual), 4),
        "max_gap_fit_vs_observed": round(max_gap, 4),
        "annual_depreciation_by_body_type": body,
        "raw_median_price_by_model_year": {int(k): float(v) for k, v in raw_by_year.items()},
        "example_forecast": {
            "vehicle": f"{year} {make} {model}",
            "listings": int(len(ex)),
            "median_price_2023": start_value,
            "projection": forecast,
        },
    }
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUTS_DIR / "time_series_analysis.json", "w") as f:
        json.dump(report, f, indent=2)

    plt.rcParams["font.family"] = "Arial"

    # Figure 1: value retained vs years older, observed medians + fitted curve
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    ages = np.linspace(0, 8, 200)
    ax.plot(ages[ages <= 5], 100 * np.exp(-rate * ages[ages <= 5]), color=INK_SECONDARY,
            linewidth=1.5, label=f"Fitted: {annual:.1%} a year")
    ax.plot(ages[ages >= 5], 100 * np.exp(-rate * ages[ages >= 5]), color=INK_SECONDARY,
            linewidth=1.5, linestyle=(0, (4, 3)), label="Projection beyond the data")
    obs_x = np.concatenate([[0], by_age.index.values])
    obs_y = np.concatenate([[100], 100 * by_age["median"].values])
    ax.plot(obs_x, obs_y, "o", color=BLUE, markersize=8, markeredgecolor="white",
            markeredgewidth=2, label="Observed (median of same-model comparisons)", zorder=3)
    for x, y in zip(obs_x[1:], obs_y[1:]):
        ax.annotate(f"{y:.0f}%", (x, y), textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=9, color=INK)
    style(ax, "How much value a car keeps as it ages",
          "Years older than the newest listing of the same model", "Value retained (%)")
    ax.set_ylim(0, 108)
    ax.set_xlim(-0.3, 8.3)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "depreciation_curve.png")
    plt.close(fig)

    # Figure 2: annual depreciation by body type, sorted
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
    labels = [b["body_type"] for b in body]
    values = [100 * b["annual_depreciation"] for b in body]
    bars = ax.barh(labels, values, color=BLUE, height=0.6)
    for bar, v, b in zip(bars, values, body):
        ax.text(v + 0.15, bar.get_y() + bar.get_height() / 2,
                f"{v:.1f}%  (n={b['model_year_pairs']})", va="center", fontsize=9, color=INK)
    style(ax, "Annual depreciation by body type", "Value lost per year (%)", "")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_xlim(0, max(values) * 1.35)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "depreciation_by_body_type.png")
    plt.close(fig)

    # Figure 3: example forecast over calendar years
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
    years = [p["year"] for p in forecast]
    vals = [p["projected_value"] / 1000 for p in forecast]
    ax.plot(years, vals, color=BLUE, linewidth=2, linestyle=(0, (4, 3)), label="Projected")
    ax.plot(years[0], vals[0], "o", color=BLUE, markersize=8, markeredgecolor="white",
            markeredgewidth=2, label="2023 median listing price", zorder=3)
    for x, y in zip(years, vals):
        ax.annotate(f"R{y:,.0f}k", (x, y), textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=9, color=INK)
    style(ax, f"Forecast: {year} {make} {model}", "Year", "Value (R thousand)")
    ax.set_ylim(0, max(vals) * 1.25)
    ax.set_xticks(years)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "value_forecast_example.png")
    plt.close(fig)

    print(f"{len(pairs)} same-model comparisons across "
          f"{report['models_with_several_years']} models")
    for k, v in by_age.iterrows():
        print(f"  {k} year(s) older: {v['median']:.1%} of value retained ({int(v['count'])} pairs)")
    print(f"Fitted annual depreciation: {annual:.1%} (largest gap to observed: {max_gap:.1%})")
    for b in body:
        print(f"  {b['body_type']:<14} {b['annual_depreciation']:.1%}  (n={b['model_year_pairs']})")
    print(f"Example: {year} {make} {model}, R{start_value:,.0f} in 2023 -> "
          + ", ".join(f"{p['year']}: R{p['projected_value']:,}" for p in forecast[1:]))
    print(f"Saved {OUTPUTS_DIR / 'time_series_analysis.json'} and 3 figures in {FIGURES_DIR}")


if __name__ == "__main__":
    main()
