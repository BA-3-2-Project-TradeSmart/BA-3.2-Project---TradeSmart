"""
TradeSmart AI - free-text condition parser (project documentation Section 5.5).

Section 5.5 commits to using NLP to turn a customer's free-text answers
("what's been done to the car?", "describe any dents") into structured
features, via three named techniques, plus a lightweight quality check:

  1. Intent recognition        - what is the customer describing?
  2. Entity/number extraction  - pull out figures (mileage, dent size).
  3. Keyword mapping           - detect respray / aftermarket mods /
                                  install quality (reputable vs DIY).
  4. Confidence/consistency check - flag vague or contradictory answers
                                  for a human consultant to review.

Design notes for the write-up:

- No real transcript log of customer free-text answers exists yet for
  this dealership (the chatbot isn't live) - the same honest situation
  the project documentation itself is in when it says its Section 5.2
  sample data table is a "fabricated example ... for demonstration
  only". So the 102 labelled examples in TRAINING_EXAMPLES below are
  hand-authored by the developer to *illustrate the technique* with
  realistic South African customer phrasing, not scraped or invented
  to look like a real transcript log. In production, Section 5.4's
  continuous-learning feedback loop is exactly the mechanism that
  would replace this seed set with real logged answers over time.
- Intent recognition is a genuine trained model (TF-IDF + Logistic
  Regression over the seed sentences) - this is the "lightweight
  intent-classification model" the doc calls for, not a hardcoded
  if/else chain.
- Entity extraction (mileage, dent-size wording) and keyword mapping
  (respray / mods / install quality) are regex + dictionary-based.
  This mirrors real slot-filling NLP pipelines, where a trained model
  picks the *intent* but a deterministic layer extracts the *values* -
  it also means these extractions stay auditable/explainable, which
  Section 5.4 flags as a trust requirement for sales staff.
- This script trains and demonstrates the classifier. The chatbot uses
  the saved model live: chatbot/condition_nlp.py categorises each
  condition mention with it.
"""

import json
import re
import time

import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
import joblib

from features import PROJECT_ROOT

MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
RANDOM_STATE = 42


# ---------------------------------------------------------------------
# 1. Intent recognition - hand-authored illustrative seed sentences,
#    used to train a small TF-IDF + Logistic Regression classifier.
# ---------------------------------------------------------------------
TRAINING_EXAMPLES = [
    # clean_no_issues
    ("The car is in great shape, no issues at all.", "clean_no_issues"),
    ("Nothing wrong with it, it's basically perfect.", "clean_no_issues"),
    ("No damage, no accidents, all original.", "clean_no_issues"),
    ("It's a clean car, never had any problems.", "clean_no_issues"),
    ("Everything is fine, no dents or scratches.", "clean_no_issues"),
    ("Car is spotless, well looked after.", "clean_no_issues"),
    ("No accidents, no dents, still on the original paint.", "clean_no_issues"),
    ("It's in mint condition honestly.", "clean_no_issues"),
    ("Nothing to report, the car is 100%.", "clean_no_issues"),
    ("All good, no issues to mention.", "clean_no_issues"),
    ("Never crashed it, never dented it, all original panels.", "clean_no_issues"),
    ("The car has no accident history and no cosmetic damage.", "clean_no_issues"),
    ("It's clean, I've never had an issue with it.", "clean_no_issues"),
    ("No problems, runs and looks perfect.", "clean_no_issues"),
    # accident_disclosure
    ("It was in a small accident about two years ago.", "accident_disclosure"),
    ("There was a collision on the highway, front bumper got damaged.", "accident_disclosure"),
    ("Someone hit me from behind at a robot, minor accident.", "accident_disclosure"),
    ("I had a bump in a parking lot last year.", "accident_disclosure"),
    ("It was written off once but repaired properly by an approved panel shop.", "accident_disclosure"),
    ("Had an insurance claim after a crash in 2021.", "accident_disclosure"),
    ("Rear-ended someone, so the back bumper was replaced.", "accident_disclosure"),
    ("There was a fender bender a while back.", "accident_disclosure"),
    ("I was in an accident on the N1, the door was damaged.", "accident_disclosure"),
    ("Yes, one accident, a taxi hit the side of the car.", "accident_disclosure"),
    ("It was involved in a collision but nothing structural.", "accident_disclosure"),
    ("Minor accident, insurance handled the repair.", "accident_disclosure"),
    # dent_description
    ("There's a small dent about the size of a golf ball on the rear door.", "dent_description"),
    ("A dent on the bonnet, roughly the size of a tennis ball.", "dent_description"),
    ("There's a ding on the door, maybe 3cm across.", "dent_description"),
    ("Small dent on the boot lid, about a coin size.", "dent_description"),
    ("There's a dent on the front fender, about the size of a fist.", "dent_description"),
    ("A little dent near the wheel arch, quite small.", "dent_description"),
    ("There's a big dent on the side, about the size of a basketball.", "dent_description"),
    ("Hail damage, lots of small dents on the roof.", "dent_description"),
    ("There's a dent the size of an orange on the rear panel.", "dent_description"),
    ("A shopping trolley dinged the door, small mark, about 2cm.", "dent_description"),
    ("There's a fairly large dent on the bumper, size of a dinner plate.", "dent_description"),
    ("Small ding on the mirror cover, barely noticeable.", "dent_description"),
    # windscreen_damage
    ("There's a small chip on the windscreen from a stone.", "windscreen_damage"),
    ("The windscreen has a crack running across the bottom corner.", "windscreen_damage"),
    ("Chip on the windscreen, driver's side.", "windscreen_damage"),
    ("There's a crack in the windscreen that's getting bigger.", "windscreen_damage"),
    ("Small stone chip on the windshield, hasn't spread.", "windscreen_damage"),
    ("The windscreen cracked after a stone hit it on the highway.", "windscreen_damage"),
    ("There's a chip near the wiper, small but visible.", "windscreen_damage"),
    ("Windscreen has a hairline crack on the passenger side.", "windscreen_damage"),
    # respray_mention
    ("The bonnet was resprayed after the accident.", "respray_mention"),
    ("It got a full respray two years ago.", "respray_mention"),
    ("The door was repainted to match after a scrape.", "respray_mention"),
    ("Had the whole car resprayed, new paint job.", "respray_mention"),
    ("The bumper was repainted after the bump.", "respray_mention"),
    ("Paint job was redone on the rear quarter panel.", "respray_mention"),
    ("It's had a partial respray on the front end.", "respray_mention"),
    ("The roof was resprayed due to hail damage.", "respray_mention"),
    # aftermarket_mods
    ("It has aftermarket alloy wheels fitted.", "aftermarket_mods"),
    ("I installed a new sound system with a subwoofer.", "aftermarket_mods"),
    ("There's a custom exhaust fitted by a reputable shop.", "aftermarket_mods"),
    ("Added mag wheels and a lowered suspension kit.", "aftermarket_mods"),
    ("Fitted a new stereo and speakers myself.", "aftermarket_mods"),
    ("There are aftermarket rims, professionally installed at a dealer.", "aftermarket_mods"),
    ("Custom subwoofer and amp in the boot, I fitted it myself.", "aftermarket_mods"),
    ("New alloy wheels, installed by a proper fitment centre.", "aftermarket_mods"),
    ("It has a sound system upgrade, DIY installation.", "aftermarket_mods"),
    ("Coilover suspension fitted at a specialist shop.", "aftermarket_mods"),
    # service_history
    ("Full service history, always serviced at the dealer.", "service_history"),
    ("Serviced every year, all stamps in the book.", "service_history"),
    ("Last service was six months ago at a franchise dealer.", "service_history"),
    ("It's got a partial service history, missed a couple of intervals.", "service_history"),
    ("No service book, I lost the records.", "service_history"),
    ("Regularly serviced, oil changed every 10,000km.", "service_history"),
    ("Full service history with the same dealer since new.", "service_history"),
    ("Serviced privately, not at the main dealer.", "service_history"),
    # unclear / vague
    ("I'm not sure, maybe there's something but I can't remember.", "unclear"),
    ("Not certain, it's been a while since I checked.", "unclear"),
    ("I think it's fine, not 100% sure though.", "unclear"),
    ("Hard to say, I'd have to look again.", "unclear"),
    ("Maybe a small thing somewhere, can't recall exactly.", "unclear"),
    ("I don't really know, my partner usually deals with the car.", "unclear"),
    ("Kind of hard to explain, not sure what you mean.", "unclear"),
    ("Not sure to be honest, never really checked it properly.", "unclear"),
    # Short fragments - the chat's condition boxes get answers like "a few
    # scratches" or "aftermarket rims" far more often than full sentences,
    # and with sentences alone the classifier misread those (e.g. it put
    # "scratches" under clean_no_issues).
    ("small scratch on the bumper", "dent_description"),
    ("a few scratches on the door", "dent_description"),
    ("minor scrape on the rear bumper", "dent_description"),
    ("hail damage on the roof", "dent_description"),
    ("dented rear door", "dent_description"),
    ("scratches", "dent_description"),
    ("windscreen chip", "windscreen_damage"),
    ("cracked windscreen", "windscreen_damage"),
    ("stone chip on the windscreen", "windscreen_damage"),
    ("resprayed bonnet", "respray_mention"),
    ("bumper repainted", "respray_mention"),
    ("new paint on the driver's door", "respray_mention"),
    ("aftermarket rims", "aftermarket_mods"),
    ("lowered suspension", "aftermarket_mods"),
    ("tow bar and canopy", "aftermarket_mods"),
    ("window tint", "aftermarket_mods"),
    ("upgraded exhaust", "aftermarket_mods"),
    ("mag wheels", "aftermarket_mods"),
    ("minor accident last year", "accident_disclosure"),
    ("front-end collision", "accident_disclosure"),
    ("full service history", "service_history"),
    ("no idea", "unclear"),
]

INTENT_LABELS = sorted(set(label for _, label in TRAINING_EXAMPLES))


def train_intent_classifier():
    texts = [t for t, _ in TRAINING_EXAMPLES]
    labels = [l for _, l in TRAINING_EXAMPLES]

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.25, random_state=RANDOM_STATE, stratify=labels
    )

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=1000, C=5.0)),
    ])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)

    # Refit on ALL examples for the version we actually ship/use, now that
    # we've measured held-out accuracy honestly on the split above.
    pipe.fit(texts, labels)
    return pipe, acc, report


# ---------------------------------------------------------------------
# 2. Entity / number extraction
# ---------------------------------------------------------------------
_nlp = spacy.load("en_core_web_sm")

MILEAGE_PATTERN = re.compile(
    r"(\d[\d,]*(?:\.\d+)?)\s*(km|kilometres?|kilometers?|miles?|mi)\b", re.IGNORECASE
)
CM_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\s*(cm|centimet(?:re|er)s?|mm|millimet(?:re|er)s?)\b", re.IGNORECASE
)

# "about the size of a golf ball" style comparisons -> a standard
# small / medium / large severity scale (Section 5.5's own example).
SIZE_REFERENCE = {
    "pea": "small", "coin": "small", "golf ball": "small", "grape": "small",
    "ping pong ball": "small", "R5 coin": "small", "marble": "small",
    "tennis ball": "medium", "egg": "medium", "orange": "medium",
    "fist": "medium", "apple": "medium", "lemon": "medium",
    "basketball": "large", "dinner plate": "large", "saucer": "large",
    "melon": "large", "watermelon": "large", "football": "large",
}


def extract_mileage_km(text: str):
    m = MILEAGE_PATTERN.search(text)
    if not m:
        return None
    value = float(m.group(1).replace(",", ""))
    unit = m.group(2).lower()
    if unit.startswith("mi"):
        value = value * 1.60934
    return round(value)


def extract_dent_severity(text: str):
    """
    Returns (severity, evidence) where severity is 'small'/'medium'/'large'
    or None if nothing was found, and evidence explains why - kept for
    the consistency check and for a human reviewer to audit the call.
    """
    text_lower = text.lower()

    # Explicit measurement takes priority over a vague object comparison.
    cm_match = CM_PATTERN.search(text)
    if cm_match:
        value = float(cm_match.group(1))
        unit = cm_match.group(2).lower()
        cm_value = value / 10 if unit.startswith("m") else value  # mm -> cm
        if cm_value < 2:
            severity = "small"
        elif cm_value <= 5:
            severity = "medium"
        else:
            severity = "large"
        return severity, f"measured at {cm_value:g}cm"

    for phrase, severity in SIZE_REFERENCE.items():
        if phrase.lower() in text_lower:
            return severity, f"compared to a {phrase}"

    return None, None


# ---------------------------------------------------------------------
# 3. Keyword mapping
# ---------------------------------------------------------------------
RESPRAY_KEYWORDS = ["respray", "resprayed", "repaint", "repainted", "new paint", "paint job"]

AFTERMARKET_KEYWORDS = {
    "wheels": ["aftermarket wheel", "aftermarket rim", "mag wheel", "alloy wheel", "new rim", "custom wheel", "custom rim"],
    "sound_system": ["sound system", "subwoofer", "speakers", "stereo", "amp "],
    "suspension": ["lowered", "suspension kit", "coilover"],
    "exhaust": ["exhaust", "muffler"],
}

REPUTABLE_INSTALL_KEYWORDS = [
    "dealer", "franchise", "professionally installed", "installed at a shop",
    "fitment centre", "fitment center", "specialist shop", "approved panel shop", "reputable",
]
DIY_INSTALL_KEYWORDS = [
    "myself", "diy", "did it myself", "at home", "fitted it myself", "installed it myself",
]

ACCIDENT_KEYWORDS = [
    "accident", "collision", "crash", "written off", "insurance claim",
    "rear-ended", "rear ended", "fender bender", "hit me", "hit the",
]
NO_ACCIDENT_PHRASES = [
    "no accident", "never had an accident", "never been in an accident",
    "accident free", "accident-free", "no collision",
]
VAGUE_MARKERS = [
    "not sure", "maybe", "i think", "kind of", "sort of", "not certain",
    "can't remember", "cant remember", "hard to say", "don't really know", "dont really know",
]


def detect_respray(text: str) -> bool:
    text_lower = text.lower()
    return any(k in text_lower for k in RESPRAY_KEYWORDS)


def detect_aftermarket_mods(text: str):
    text_lower = text.lower()
    found = [category for category, kws in AFTERMARKET_KEYWORDS.items()
             if any(k in text_lower for k in kws)]
    return found


def detect_install_quality(text: str):
    text_lower = text.lower()
    reputable = any(k in text_lower for k in REPUTABLE_INSTALL_KEYWORDS)
    diy = any(k in text_lower for k in DIY_INSTALL_KEYWORDS)
    if reputable and not diy:
        return "Reputable"
    if diy and not reputable:
        return "DIY"
    if reputable and diy:
        return "Ambiguous"  # both signals present - let the consistency check flag it
    return None


# ---------------------------------------------------------------------
# 4. Confidence / consistency check
# ---------------------------------------------------------------------
def check_consistency(text: str, intent: str, extracted: dict) -> list:
    """Returns a list of human-readable flags for a consultant to review."""
    text_lower = text.lower()
    flags = []

    if any(marker in text_lower for marker in VAGUE_MARKERS):
        flags.append("Vague answer - contains hedging language, low confidence.")

    no_accident_claim = any(p in text_lower for p in NO_ACCIDENT_PHRASES)
    # "no accident(s)" itself contains the word "accident" - strip the
    # claim phrase out first so it can't trip its own contradiction check.
    text_without_claim = text_lower
    for phrase in NO_ACCIDENT_PHRASES:
        text_without_claim = text_without_claim.replace(phrase, "")
    accident_mentioned_elsewhere = any(k in text_without_claim for k in ACCIDENT_KEYWORDS)
    if no_accident_claim and accident_mentioned_elsewhere:
        flags.append("Contradiction - customer claims no accident history but also references an accident/collision term.")
    elif no_accident_claim and extracted.get("respray"):
        flags.append("Contradiction - customer claims no accident history but also mentions a respray (often accident-related repair work).")

    if intent == "clean_no_issues" and (extracted.get("respray") or extracted.get("aftermarket_mods")):
        flags.append("Contradiction - customer describes the car as issue-free but also mentions a respray or modification.")

    if extracted.get("install_quality") == "Ambiguous":
        flags.append("Ambiguous install quality - both 'reputable shop' and 'DIY' language detected.")

    return flags


# ---------------------------------------------------------------------
# Pulling it all together
# ---------------------------------------------------------------------
class FreeTextConditionParser:
    def __init__(self, intent_classifier):
        self.intent_classifier = intent_classifier

    def parse(self, text: str) -> dict:
        doc = _nlp(text)

        intent = self.intent_classifier.predict([text])[0]
        proba = self.intent_classifier.predict_proba([text])[0]
        intent_confidence = round(float(max(proba)), 3)

        dent_severity, dent_evidence = extract_dent_severity(text)

        numbers_found = [ent.text for ent in doc.ents if ent.label_ in ("CARDINAL", "QUANTITY")]

        extracted = {
            "mileage_km": extract_mileage_km(text),
            "dent_severity": dent_severity,
            "dent_evidence": dent_evidence,
            "respray": detect_respray(text),
            "aftermarket_mods": detect_aftermarket_mods(text),
            "install_quality": detect_install_quality(text),
            "numbers_mentioned": numbers_found,
        }

        flags = check_consistency(text, intent, extracted)

        return {
            "text": text,
            "intent": intent,
            "intent_confidence": intent_confidence,
            "extracted": extracted,
            "review_flags": flags,
            "needs_human_review": len(flags) > 0 or intent_confidence < 0.5,
        }


# ---------------------------------------------------------------------
# Demo / illustrative run over example customer answers (NOT the
# training sentences above - a fresh, held-out set to show the module
# generalises, including deliberately vague/contradictory ones).
# ---------------------------------------------------------------------
DEMO_CUSTOMER_ANSWERS = [
    "There's a small dent about the size of a golf ball on the rear bumper.",
    "It's done 120,000km and has full service history at the dealer.",
    "No accidents at all, the car is completely clean.",
    "Honestly no accidents, but it was resprayed after a small bump last year.",
    "It has aftermarket alloy wheels, fitted at a proper fitment centre.",
    "I put in a new sound system myself, subwoofer and amp.",
    "There's a crack in the windscreen on the passenger side, about 4cm.",
    "Not sure really, maybe something happened but I can't remember exactly.",
    "The whole car was resprayed after a collision on the N3 two years ago.",
    "It's done roughly 85000 miles, mostly highway driving.",
]


def main():
    print("Training intent classifier on", len(TRAINING_EXAMPLES), "hand-authored seed sentences...")
    t0 = time.time()
    classifier, held_out_accuracy, report = train_intent_classifier()
    train_seconds = time.time() - t0
    print(f"Held-out intent accuracy: {held_out_accuracy:.3f}  ({train_seconds:.2f}s)")

    parser = FreeTextConditionParser(classifier)

    print("\n" + "=" * 90)
    print("Parsing illustrative customer answers (held out from the training sentences)")
    print("=" * 90)
    results = []
    for answer in DEMO_CUSTOMER_ANSWERS:
        parsed = parser.parse(answer)
        results.append(parsed)
        print(f"\n\"{answer}\"")
        print(f"  intent={parsed['intent']} (confidence={parsed['intent_confidence']})")
        print(f"  extracted={parsed['extracted']}")
        if parsed["review_flags"]:
            print(f"  FLAGS: {parsed['review_flags']}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(classifier, MODELS_DIR / "nlp_intent_classifier.joblib")

    with open(OUTPUTS_DIR / "nlp_intent_classifier_report.json", "w") as f:
        json.dump({
            "held_out_accuracy": round(held_out_accuracy, 4),
            "train_seconds": round(train_seconds, 2),
            "n_training_sentences": len(TRAINING_EXAMPLES),
            "intent_labels": INTENT_LABELS,
            "classification_report": report,
        }, f, indent=2)

    with open(OUTPUTS_DIR / "nlp_demo_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved classifier to {MODELS_DIR}/nlp_intent_classifier.joblib")
    print(f"Saved reports to {OUTPUTS_DIR}/nlp_intent_classifier_report.json and nlp_demo_results.json")


if __name__ == "__main__":
    main()
