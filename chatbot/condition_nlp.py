"""
TradeSmart AI - turns the customer's condition answers into findings to
check when the car is inspected.

The chat's condition step asks about accidents, dents, windscreen damage,
respray work and aftermarket mods. None of that changes the price: the AI
estimate comes only from make/model, year and mileage (ml_service.py),
because the Autotrader data the model learned from says nothing about a
car's condition. What a customer describes is instead turned into
structured findings - one per thing mentioned - to check at inspection.

How the text becomes findings (NLP):
  1. Split each answer into separate mentions ("small dent on the door,
     windscreen chip" -> two), skipping "no ..." / "... is fine" mentions.
  2. Classify each mention with the intent classifier trained in
     tradesmart-ml/src/nlp_parser.py (TF-IDF + Logistic Regression),
     keeping its confidence.
  3. Extract the useful detail with rules a person can audit: dent size
     ("golf ball" -> small, "4cm" -> medium), mod type, who fitted it.
  4. Flag mentions to look at first: low classifier
     confidence, hedging ("not sure"), or contradictions (customer said
     "no accident" but described collision repairs).
"""

import re

import joblib

from features import PROJECT_ROOT

CLASSIFIER_PATH = PROJECT_ROOT / "models" / "nlp_intent_classifier.joblib"
LOW_CONFIDENCE = 0.40  # 8 categories, so a random guess would score ~0.125

# Display names for the eight categories - the same rows as the seeded
# dbo.CONDITION_CATEGORY lookup table in TradeSmartAI_Database.sql.
CATEGORY_NAMES = {
    "accident_disclosure": "Accident history",
    "dent_description": "Dents / scratches",
    "windscreen_damage": "Windscreen damage",
    "respray_mention": "Respray / paintwork",
    "aftermarket_mods": "Aftermarket modifications",
    "service_history": "Service history",
    "clean_no_issues": "No issues reported",
    "unclear": "Unclear answer",
}

# A fragment containing none of these words is a continuation of the
# previous mention, not a new one ("aftermarket rims, fitted by a
# professional shop" is one mention, not two).
TOPIC_WORDS = (
    "dent", "ding", "scratch", "scrape", "hail", "rust", "chip", "crack", "windscreen",
    "windshield", "respray", "repaint", "paint", "rim", "wheel", "mag", "exhaust",
    "suspension", "lowered", "coilover", "sound", "subwoofer", "speaker", "stereo", "amp",
    "tint", "canopy", "tow", "bar", "accident", "collision", "crash", "bump", "service",
    "mod", "aftermarket", "bumper", "door", "bonnet", "panel",
)
NEGATION = re.compile(
    r"^\s*(no|none|nothing|never|n/?a|stock|standard|original)\b"
    r"|\b(is|are|was|were)\s+(fine|ok|okay|good|perfect|clean|original)\b",
    re.IGNORECASE,
)

SIZE_REFERENCE = {
    "pea": "small", "coin": "small", "golf ball": "small", "grape": "small", "marble": "small",
    "tennis ball": "medium", "egg": "medium", "orange": "medium", "fist": "medium",
    "apple": "medium", "lemon": "medium",
    "basketball": "large", "dinner plate": "large", "saucer": "large", "football": "large",
}
SIZE_WORDS = {"small": "small", "minor": "small", "tiny": "small", "little": "small",
              "big": "large", "large": "large", "huge": "large", "major": "large"}
MEASURE = re.compile(r"(\d+(?:\.\d+)?)\s*(cm|mm)\b", re.IGNORECASE)

MOD_TYPES = {
    "wheels": ("rim", "wheel", "mag"),
    "suspension": ("lowered", "suspension", "coilover"),
    "exhaust": ("exhaust", "muffler"),
    "sound system": ("sound system", "subwoofer", "speaker", "stereo", "amp"),
    "window tint": ("tint",),
    "canopy": ("canopy",),
    "tow bar": ("tow bar", "towbar"),
    "nudge bar": ("nudge bar",),
}
REPUTABLE_INSTALL = ("dealer", "professional", "fitment centre", "fitment center",
                     "specialist", "reputable", "approved", "shop")
DIY_INSTALL = ("myself", "diy", "at home", "a friend")

ACCIDENT_WORDS = ("accident", "collision", "crash", "written off", "write-off", "rear-ended",
                  "insurance claim", "fender bender")
VAGUE_WORDS = ("not sure", "maybe", "i think", "can't remember", "cant remember",
               "hard to say", "don't know", "dont know", "no idea")


def _has_word(text, words):
    return any(re.search(r"\b" + re.escape(w), text) for w in words)


def split_mentions(text):
    pieces = [p.strip() for p in re.split(r"[.;!?\n]|,|\band\b|\bbut\b", text or "") if p.strip()]
    mentions = []
    for piece in pieces:
        if mentions and not _has_word(piece.lower(), TOPIC_WORDS):
            mentions[-1] = f"{mentions[-1]}, {piece}"
        else:
            mentions.append(piece)
    return [m for m in mentions if not NEGATION.search(m)]


def _dent_size(text):
    m = MEASURE.search(text)
    if m:
        cm = float(m.group(1)) / (10 if m.group(2).lower() == "mm" else 1)
        size = "small" if cm < 2 else "medium" if cm <= 5 else "large"
        return f"{size}, measured at {cm:g}cm"
    for phrase, size in SIZE_REFERENCE.items():
        if phrase in text:
            return f"{size}, about the size of a {phrase}"
    for word, size in SIZE_WORDS.items():
        if re.search(rf"\b{word}\b", text):
            return size
    return None


def _detail(category, text):
    t = text.lower()
    if category == "dent_description":
        return _dent_size(t)
    if category == "windscreen_damage":
        kind = "crack" if "crack" in t else "chip" if "chip" in t else "damage"
        size = _dent_size(t)
        return f"Windscreen {kind}" + (f", {size}" if size else "")
    if category == "aftermarket_mods":
        types = [name for name, words in MOD_TYPES.items() if _has_word(t, words)]
        fitted = ("reputable installer" if _has_word(t, REPUTABLE_INSTALL)
                  else "DIY install" if _has_word(t, DIY_INSTALL) else None)
        parts = [", ".join(types) or "type not clear"] + ([fitted] if fitted else [])
        return "; ".join(parts)
    return None


class ConditionAnalyser:
    def __init__(self):
        self.classifier = joblib.load(CLASSIFIER_PATH)

    def _classify(self, text):
        proba = self.classifier.predict_proba([text])[0]
        best = proba.argmax()
        return str(self.classifier.classes_[best]), round(float(proba[best]), 3)

    def analyse(self, accident, damage_text, mods_text):
        """accident is 'yes' / 'no' (the radio buttons); the two texts are the
        free-text boxes. Returns a list of finding dicts, in the order the
        customer mentioned them."""
        findings = []
        if accident == "yes":
            findings.append({
                "category": "accident_disclosure", "customer_text": "Answered 'yes' to: any accidents?",
                "detail": None, "detected_by": "Customer", "confidence": None,
                "needs_review": True, "review_reason": "Customer disclosed an accident - inspect for repair quality.",
            })

        for mention in split_mentions(damage_text) + split_mentions(mods_text):
            category, confidence = self._classify(mention)
            if category == "clean_no_issues":
                continue
            lower = mention.lower()
            reasons = []
            if round(confidence, 2) < LOW_CONFIDENCE:  # same rounding as the "40%" the dashboard shows
                reasons.append(f"Low NLP confidence ({confidence:.0%}) - check the category.")
            if _has_word(lower, VAGUE_WORDS):
                reasons.append("Vague answer - customer wasn't sure.")
            if accident == "no" and _has_word(lower, ACCIDENT_WORDS):
                reasons.append("Contradiction - customer answered 'no accidents' but describes one here.")
            findings.append({
                "category": category, "customer_text": mention[:500],
                "detail": _detail(category, mention), "detected_by": "NLP", "confidence": confidence,
                "needs_review": bool(reasons), "review_reason": " ".join(reasons) or None,
            })
        return findings


def summary_text(findings):
    """One line for the chat and the conversation log, e.g.
    'Dents / scratches (small), Windscreen damage (Windscreen chip)'."""
    return ", ".join(
        CATEGORY_NAMES[f["category"]] + (f" ({f['detail']})" if f["detail"] else "")
        for f in findings
    )


# Loaded once per process, like ml_service.service.
analyser = ConditionAnalyser()
