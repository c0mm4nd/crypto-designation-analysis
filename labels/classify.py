"""
Map raw labels from web3resear.ch / SlowMist to canonical categories used
for verified-negative selection in AUC evaluation.

Source-aware
------------
web3resear.ch aggregates three providers, each with its own schema:

  allium : the `label` field is the tag itself (`cex`, `binance`, `mixer`,
           `mixer_user`, `stablecoin`, `token-contract`, `blacklisted`, ...).
  rabby  : the `label` field is `<kind>:<value>`  (`cex:Binance`,
           `protocol:Tether`, `dex:Uniswap V2`, `thirdparty:etherscan`, ...).
  oklink : the `label` field is a generic type (`entityTag`, `tokenTag`,
           `fullEntityTag`); the `name_tag` carries the human-readable
           description (`Exchange: Binance. DepositAndWithdraw_1`,
           `Token: Tether USD`, `Mixer: Tornado Cash`).

Categories
----------
CEX          centralised exchange                    → verified negative
DEX          decentralised exchange                  → verified negative
PROTOCOL     recognised DeFi / infrastructure        → verified negative
CONTRACT     stablecoin / token / utility contract   → verified negative
BRIDGE       cross-chain bridge                      → verified negative
INSTITUTION  custodian / market-maker / MM / OTC     → verified negative
SUSPICIOUS   mixer itself / sanctioned / blacklisted / scam / hack → NOT a negative
OTHER        labels present but none match the whitelist
UNKNOWN      no labels

Note that `mixer_user` (Allium: counterparty of a mixer) is NOT SUSPICIOUS —
it appears on major stablecoin contracts and exchange hot wallets for
aggregation reasons.
"""

from __future__ import annotations

LEGIT_CATEGORIES = {"CEX", "DEX", "PROTOCOL", "CONTRACT", "BRIDGE", "INSTITUTION"}


# ── Allium categorical tags ───────────────────────────────────────────────────
# the `label` string IS the tag
_ALLIUM_SUSPICIOUS = {
    "blacklisted", "sanctioned", "ofac", "darknet", "darkmarket", "darknet-market",
    "mixer", "tornado", "tornado_cash", "tornado-cash",
    "scam", "scammer", "phishing", "phisher",
    "hack", "hacker", "exploit", "exploited", "ransomware",
    "fraud", "theft", "thief", "illicit", "ponzi",
    "stealer", "drainer", "terrorism", "terrorist",
}
_ALLIUM_LEGIT = {
    "cex":                 "CEX",
    "exchange":            "CEX",
    "dex":                 "DEX",
    "amm":                 "DEX",
    "stablecoin":          "CONTRACT",
    "token-contract":      "CONTRACT",
    "token_contract":      "CONTRACT",
    "bridge":              "BRIDGE",
    "protocol":            "PROTOCOL",
    "defi":                "PROTOCOL",
    "lending":             "PROTOCOL",
    "liquid-staking":      "PROTOCOL",
    "staking":             "PROTOCOL",
    "validator":           "INSTITUTION",
    "oracle":              "INSTITUTION",
    "market-maker":        "INSTITUTION",
    "market_maker":        "INSTITUTION",
    "custodian":           "INSTITUTION",
    "payment":             "INSTITUTION",
    "payment-processor":   "INSTITUTION",
    "otc":                 "INSTITUTION",
}
# Allium name-tag keywords (lowercased); used when `label` names a specific
# entity whose category we know.
_ALLIUM_NAMED_CEX = {
    "binance", "coinbase", "kraken", "okx", "okex", "huobi", "htx",
    "bitfinex", "kucoin", "bybit", "bitstamp", "gemini", "ftx",
    "mexc", "gate.io", "gate", "bitget", "crypto.com", "poloniex",
    "upbit", "bithumb", "whitebit", "bitflyer", "deribit",
    "wazirx",
}
_ALLIUM_NAMED_DEX = {
    "uniswap", "sushiswap", "sushi", "curve", "pancake",
    "pancakeswap", "1inch", "balancer", "dydx", "kyber", "kyberswap",
    "matcha", "sunswap", "justlend",
}
_ALLIUM_NAMED_PROTOCOL = {
    "aave", "lido", "compound", "makerdao", "yearn",
    "rocketpool", "convex", "frax", "ethena", "morpho",
    "eigenlayer", "pendle", "gmx", "radiant", "spark",
}


# ── Rabby prefixes ────────────────────────────────────────────────────────────
_RABBY_PREFIX_TO_CAT = {
    "cex:":        "CEX",
    "exchange:":   "CEX",
    "dex:":        "DEX",
    "protocol:":   "PROTOCOL",
    "defi:":       "PROTOCOL",
    "bridge:":     "BRIDGE",
    "token:":      "CONTRACT",
}


# ── Oklink name_tag prefixes ──────────────────────────────────────────────────
_OKLINK_NAME_PREFIX_CAT = [
    ("exchange:",      "CEX"),
    ("mixer:",         "SUSPICIOUS"),
    ("sanctioned:",    "SUSPICIOUS"),
    ("darknet",        "SUSPICIOUS"),
    ("scam",           "SUSPICIOUS"),
    ("phish",          "SUSPICIOUS"),
    ("hack",           "SUSPICIOUS"),
    ("bridge:",        "BRIDGE"),
    ("token:",         "CONTRACT"),
    ("defi:",          "PROTOCOL"),
    ("protocol:",      "PROTOCOL"),
    ("project:",       "OTHER"),
]


# ── Category precedence (SUSPICIOUS > LEGIT > OTHER > UNKNOWN) ────────────────
_LEGIT_RANK = ["CEX", "DEX", "PROTOCOL", "BRIDGE", "INSTITUTION", "CONTRACT"]


def _classify_allium(label: str, name_tag: str) -> str | None:
    """Return canonical category from one Allium label, or None."""
    lab = (label or "").strip().lower()
    if not lab:
        return None
    if lab in _ALLIUM_SUSPICIOUS:
        return "SUSPICIOUS"
    cat = _ALLIUM_LEGIT.get(lab)
    if cat:
        return cat
    # named-entity style: label is the exchange name directly
    if lab in _ALLIUM_NAMED_CEX:
        return "CEX"
    if lab in _ALLIUM_NAMED_DEX:
        return "DEX"
    if lab in _ALLIUM_NAMED_PROTOCOL:
        return "PROTOCOL"
    return None


def _classify_rabby(label: str) -> str | None:
    lab = (label or "").strip().lower()
    if not lab:
        return None
    for pref, cat in _RABBY_PREFIX_TO_CAT.items():
        if lab.startswith(pref):
            return cat
    return None


def _classify_oklink(label: str, name_tag: str) -> str | None:
    lab = (label or "").strip().lower()
    name = (name_tag or "").strip().lower()
    # oklink `label` is a generic type (entityTag / tokenTag); the information
    # lives in name_tag.
    if not name:
        return None
    # tokenTag → token contract
    if lab == "tokentag" or name.startswith("token:"):
        return "CONTRACT"
    if lab in ("entitytag", "fullentitytag"):
        for pref, cat in _OKLINK_NAME_PREFIX_CAT:
            if pref in name:
                return cat
    return None


def _slowmist_categories(sm: dict | None) -> list[str]:
    """MistTrack /v1/address_labels returns {data:{label_list, label_type}}."""
    if not sm:
        return []
    data = sm.get("data") if isinstance(sm, dict) else None
    if not data:
        return []
    out: list[str] = []
    label_type = str(data.get("label_type") or "").strip().lower()
    label_list = [str(x).strip().lower() for x in (data.get("label_list") or [])]
    joined = " ".join([label_type] + label_list)

    if any(k in joined for k in (
        "sanction", "ofac", "blacklist", "scam", "phish",
        "mixer", "tornado", "hack", "exploit", "fraud",
        "theft", "ransom", "illicit", "darknet", "ponzi",
    )):
        out.append("SUSPICIOUS")
    if label_type in ("exchange", "cex"):
        out.append("CEX")
    elif label_type in ("dex", "amm"):
        out.append("DEX")
    elif label_type in ("token", "stablecoin", "token-contract"):
        out.append("CONTRACT")
    elif label_type in ("defi", "protocol", "staking", "lending"):
        out.append("PROTOCOL")
    elif label_type in ("bridge", "cross-chain"):
        out.append("BRIDGE")
    elif label_type in ("market maker", "mm", "custodian", "fund", "otc"):
        out.append("INSTITUTION")
    return out


def classify(
    w3r_labels: list | None = None,
    slowmist:   dict | None = None,
) -> tuple[str, list[str]]:
    """
    Return (category, evidence_tokens).

    Precedence: SUSPICIOUS > any LEGIT > OTHER > UNKNOWN.
    Among LEGIT categories, precedence is CEX > DEX > PROTOCOL > BRIDGE >
    INSTITUTION > CONTRACT (so a node tagged both "cex" and "token-contract"
    is treated as an exchange).
    """
    found_cats: list[str] = []
    evidence: list[str] = []

    for item in (w3r_labels or []):
        if not isinstance(item, dict):
            continue
        src = (item.get("source") or "").lower()
        label = item.get("label") or ""
        name  = item.get("name_tag") or ""
        cat = None
        if src == "allium":
            cat = _classify_allium(label, name)
        elif src == "rabby":
            cat = _classify_rabby(label)
        elif src == "oklink":
            cat = _classify_oklink(label, name)
        if cat:
            found_cats.append(cat)
            evidence.append(f"{src}:{label}")

    found_cats.extend(_slowmist_categories(slowmist))

    if not found_cats:
        return ("UNKNOWN", [])

    if "SUSPICIOUS" in found_cats:
        return ("SUSPICIOUS", evidence)
    for cat in _LEGIT_RANK:
        if cat in found_cats:
            return (cat, evidence)
    return ("OTHER", evidence)


def is_verified_negative(category: str) -> bool:
    return category in LEGIT_CATEGORIES


def is_suspicious(category: str) -> bool:
    return category == "SUSPICIOUS"


if __name__ == "__main__":
    tests = [
        ("USDT", [
            {"label":"tokenTag","source":"oklink","name_tag":"Token: Tether USD"},
            {"label":"name","source":"rabby","name_tag":"Token: USDT"},
            {"label":"bitfinex","source":"allium","name_tag":"Tether: USDT Stablecoin"},
            {"label":"mixer_user","source":"allium","name_tag":"Tether: USDT Stablecoin"},
            {"label":"stablecoin","source":"allium","name_tag":"Tether: USDT Stablecoin"},
            {"label":"token-contract","source":"allium","name_tag":"Tether: USDT Stablecoin"},
        ]),
        ("Binance", [
            {"label":"entityTag","source":"oklink","name_tag":"Binance. DepositAndWithdraw_1"},
            {"label":"fullEntityTag","source":"oklink","name_tag":"Exchange: Binance. DepositAndWithdraw_1"},
            {"label":"cex:Binance","source":"rabby","name_tag":"Binance"},
            {"label":"binance","source":"allium","name_tag":"Binance 14"},
            {"label":"cex","source":"allium","name_tag":"Binance 14"},
            {"label":"mixer_user","source":"allium","name_tag":"Binance 14"},
        ]),
        ("Tornado", [
            {"label":"mixer","source":"allium","name_tag":"Tornado Cash: 0.1 ETH"},
            {"label":"fullEntityTag","source":"oklink","name_tag":"Mixer: Tornado Cash"},
        ]),
        ("Uniswap", [
            {"label":"dex","source":"allium","name_tag":"Uniswap V2: Router 2"},
            {"label":"uniswap","source":"allium","name_tag":"Uniswap V2: Router 2"},
        ]),
        ("Sanctioned addr (Binance 14 but flagged)", [
            {"label":"blacklisted","source":"allium","name_tag":"..."},
            {"label":"cex","source":"allium","name_tag":"..."},
        ]),
        ("Mixer user only",[
            {"label":"mixer_user","source":"allium","name_tag":"Some addr"},
        ]),
        ("Empty", []),
    ]
    for name, labels in tests:
        cat, ev = classify(labels)
        print(f"{name:50s} -> {cat:11s}  {ev}")
