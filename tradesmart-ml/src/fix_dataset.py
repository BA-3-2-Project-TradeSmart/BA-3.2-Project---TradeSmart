"""
TradeSmart AI - corrects the Autotrader dataset before training.

The dataset comes from the open-source used-car-price-prediction project
(V. Adewoyin, 2023, MIT licence), which scraped Autotrader South Africa
and published the cleaned CSV. Its cleaning notebook
("2. Autotrader-South-Africa-Project [Data Cleaning].ipynb") introduced
three errors, corrected here:

1. Two-word brands split in half. The notebook takes the first word of the
   listing title as the brand (Name.split(n=2)), so "Land Rover Defender"
   became Brand "Land", Model "Rover Defender". Same for Alfa Romeo and
   Aston Martin. Rejoined here.

2. Fuel consumption lost its leading "1". The notebook used
   .str.strip('/100km'), which strips any of the *characters* / 1 0 k m from
   both ends - so 11.5 L/100km became 0.5, 12.8 became 2.8. Not one car in
   the CSV is listed at 10-19.9 L/100km. The fuel tank capacity and fuel
   range columns weren't affected, so the true figure can be checked as
   capacity * 100 / range: where adding the lost 10 back matches that
   (within 5%), it's restored; values under 2 that can't be verified that
   way are set to blank (missing) rather than kept wrong.

3. Impossible mileage (data-entry errors, e.g. 990,000 km on a 2018 car,
   or a mileage equal to the asking price). Rows over 500,000 km dropped.

Reads  data/autotrader-south-africa-usedcars (cleaned data).csv   (as published)
Writes data/autotrader-south-africa-usedcars (corrected).csv      (used by everything else)

Run this first, from the src folder:
    python fix_dataset.py
"""

import pandas as pd

from features import PROJECT_ROOT, DATA_PATH

SOURCE_PATH = PROJECT_ROOT / "data" / "autotrader-south-africa-usedcars (cleaned data).csv"

SPLIT_BRANDS = {"Land": "Rover", "Alfa": "Romeo", "Aston": "Martin"}
MAX_MILEAGE = 500_000


def fix_brands(df):
    for first, second in SPLIT_BRANDS.items():
        mask = (df["Brand"] == first) & df["Model"].str.startswith(second + " ")
        df.loc[mask, "Brand"] = f"{first} {second}"
        df.loc[mask, "Model"] = df.loc[mask, "Model"].str[len(second) + 1:]
    return int(df["Brand"].isin([f"{a} {b}" for a, b in SPLIT_BRANDS.items()]).sum())


def fix_fuel_consumption(df):
    fc = df["Fuel consumption (average)"]
    derived = df["Fuel capacity"] * 100 / df["Fuel range (average)"]
    restorable = (fc + 10 - derived).abs() / derived < 0.05
    df.loc[restorable, "Fuel consumption (average)"] = fc[restorable] + 10
    still_bad = (df["Fuel consumption (average)"] < 2) & (df["Fuel Type"] != "Electric")
    df.loc[still_bad, "Fuel consumption (average)"] = None
    return int(restorable.sum()), int(still_bad.sum())


def main():
    df = pd.read_csv(SOURCE_PATH)
    total = len(df)

    brands_fixed = fix_brands(df)
    fuel_restored, fuel_blanked = fix_fuel_consumption(df)
    mileage_dropped = int((df["Mileage"] > MAX_MILEAGE).sum())
    df = df[df["Mileage"] <= MAX_MILEAGE]

    df.to_csv(DATA_PATH, index=False)

    print(f"{total} rows in -> {len(df)} rows out ({DATA_PATH.name})")
    print(f"  brands rejoined:          {brands_fixed}")
    print(f"  fuel consumption fixed:   {fuel_restored}")
    print(f"  fuel consumption blanked: {fuel_blanked}")
    print(f"  mileage rows dropped:     {mileage_dropped}")


if __name__ == "__main__":
    main()
