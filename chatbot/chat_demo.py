"""
TradeSmart AI - the chatbot in the terminal.

Talks to the customer step by step, like the dealership's chat assistant:
  1. Which vehicle?           -> matched against the 2,393 known makes/models
  2. Year and mileage?        -> checked and read ("85 000", "85,000", "85k" all work)
  3. Condition?               -> read by the NLP model (condition_nlp.py)
  4. Estimated value + range  -> predicted by the XGBoost model (ml_service.py)

Run from this folder:
    python chat_demo.py
"""

import re
from datetime import datetime

import condition_nlp
import ml_service

BOT = "TradeSmart AI:"
OLDEST_YEAR = 1950
MAX_MILEAGE = 1_000_000
AMOUNT = re.compile(
    r"^\s*r?\s*(\d{1,3}(?:[ ,]\d{3})+|\d+(?:\.\d+)?)(?:\.\d+)?\s*(k)?\s*(?:km|kms|kilometres|kilometers)?\s*$",
    re.IGNORECASE,
)


def rand(n):
    return "R" + f"{round(n):,}".replace(",", " ")


def ask(question):
    print(f"{BOT} {question}")
    return input("You: ").strip()


def parse_amount(text):
    m = AMOUNT.match(text or "")
    if not m:
        return None
    value = float(re.sub(r"[ ,]", "", m.group(1)))
    return int(round(value * 1000 if m.group(2) else value))


def choose_vehicle(service):
    while True:
        query = ask("Which vehicle would you like to value? Type the make and model, e.g. polo vivo")
        matches = service.search_vehicles(query, limit=8) if query else []
        if not matches:
            print(f"{BOT} I couldn't find that vehicle - try fewer words, e.g. just the model name.")
            continue
        if len(matches) == 1:
            print(f"{BOT} Found: {matches[0]['label']}")
            return matches[0]
        for i, m in enumerate(matches, start=1):
            print(f"   {i}. {m['label']}")
        pick = ask("Which one? Type its number")
        if pick.isdigit() and 1 <= int(pick) <= len(matches):
            return matches[int(pick) - 1]
        print(f"{BOT} Please type one of the numbers shown.")


def ask_year():
    newest = datetime.now().year + 1
    while True:
        text = ask("What year is it?")
        if re.fullmatch(r"\d{4}", text) and OLDEST_YEAR <= int(text) <= newest:
            return int(text)
        print(f"{BOT} Please enter the year as four digits between {OLDEST_YEAR} and {newest}, e.g. 2019.")


def ask_mileage():
    while True:
        km = parse_amount(ask("Roughly how many kilometres has it done?"))
        if km is not None and km <= MAX_MILEAGE:
            return km
        print(f"{BOT} I couldn't read that mileage - try something like 85000, 85 000 or 85k.")


def ask_condition():
    accident = ""
    while accident not in ("yes", "no"):
        accident = ask("Has it been in an accident? (yes/no)").lower()
        accident = {"y": "yes", "n": "no"}.get(accident, accident)
    damage = ask("Any dents, scratches, windscreen damage or respray work? (describe it, or press Enter to skip)")
    mods = ask("Any aftermarket modifications? (describe them, or press Enter to skip)")
    return accident, damage, mods


def main():
    print("Loading the AI models...")
    service = ml_service.service
    analyser = condition_nlp.analyser
    print(f"\n{BOT} Hi! Let's get an estimate for your trade-in.\n")

    while True:
        vehicle = choose_vehicle(service)
        year = ask_year()
        mileage = ask_mileage()
        accident, damage, mods = ask_condition()

        v = service.estimate(vehicle["brand"], vehicle["model"], year, mileage)
        findings = analyser.analyse(accident, damage, mods)

        print("\n" + "=" * 64)
        km = f"{mileage:,}".replace(",", " ")
        print(f" {year} {vehicle['label']}, {km} km")
        print("=" * 64)
        if v["out_of_range"]:
            print(f" LOW CONFIDENCE: {v['out_of_range_reason']}\n")
        print(f" Estimated value:  {rand(v['point_estimate'])}")
        print(f" Estimated range:  {rand(v['estimated_low'])} - {rand(v['estimated_high'])}")
        print(f" Model accuracy:   {v['model_r2'] * 100:.0f}% ({v['accuracy_basis']})")
        source = {"exact_model": "exact model spec match", "brand_average": "brand average specs",
                  "global_average": "overall average specs"}[v["spec_source"]]
        print(f" Based on:         XGBoost, {source}")
        if findings:
            print("\n Noted for inspection (NLP - does not change the price):")
            for f in findings:
                name = condition_nlp.CATEGORY_NAMES[f["category"]]
                detail = f" ({f['detail']})" if f["detail"] else ""
                conf = f"{f['confidence'] * 100:.0f}% confidence" if f["confidence"] is not None else "customer answer"
                print(f"  - {name}{detail}  [{conf}]")
                if f["needs_review"]:
                    print(f"      CHECK: {f['review_reason']}")
        else:
            print("\n Nothing noted about the condition.")
        print("\n This is an AI-generated first estimate, not a final offer.")
        print("=" * 64 + "\n")

        if ask("Value another car? (yes/no)").lower() not in ("yes", "y"):
            print(f"{BOT} Thanks for using TradeSmart AI!")
            break


if __name__ == "__main__":
    main()
