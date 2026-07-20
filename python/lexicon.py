"""lexicon.py — corpus of plausible values injected by the substitution.

The forger paints an "edited value" on the document (what a fraudster would write).
This module provides a LARGE, varied corpus — French / English / dates / numbers /
codes / characters / phrases — so the content is **highly diverse**.

Why diversity helps: if all substitutions looked alike (e.g. a euro amount), the
model could learn that CONTENT as a shortcut instead of the ELA compression signal.
By varying the injected text strongly, we force the model to rely on the compression
inconsistency, not on what is written.

Technical constraint: the text is drawn with OpenCV's **Hershey fonts**
(`cv2.putText`), which render **ASCII only**. Any non-ASCII glyph (e.g. '€',
accents 'é è ç') would show as '???'. -> `_ascii(...)` transliterates everything to
ASCII (é->e, ç->c, …) on output, so the source corpus may contain accents without
ever producing '???'.

API: `plausible_token(rng, size_class) -> str` (rng = numpy.random.Generator).
"""
from __future__ import annotations

import string
import unicodedata

# --- Words (FR / EN): invoice / receipt / commercial-document vocabulary ---------
_WORDS_FR = [
    "facture", "montant", "total", "client", "date", "quantite", "remise", "solde",
    "paiement", "especes", "carte", "recu", "article", "prix", "taxe", "net", "brut",
    "avoir", "acompte", "echeance", "reference", "fournisseur", "adresse", "telephone",
    "numero", "ticket", "caisse", "vendeur", "remboursement", "livraison", "commande",
    "devis", "bon", "unitaire", "designation", "libelle", "credit", "debit", "monnaie",
    "rendu", "sous-total", "acquitte", "regle", "valide", "annule", "duplicata",
]
_WORDS_EN = [
    "invoice", "amount", "total", "customer", "date", "quantity", "discount", "balance",
    "payment", "cash", "card", "receipt", "item", "price", "tax", "net", "gross",
    "credit", "deposit", "due", "reference", "supplier", "address", "phone", "number",
    "ticket", "cashier", "seller", "refund", "change", "subtotal", "delivery", "order",
    "quote", "unit", "description", "label", "debit", "currency", "paid", "void",
    "approved", "cancelled", "duplicate", "receipt", "grand", "nett",
]

# --- Short phrases (FR / EN): typical document mentions ---------------------------
_PHRASES_FR = [
    "Total a payer", "Merci de votre visite", "Facture acquittee", "Net a payer",
    "Bon pour accord", "Paiement recu", "Sous total", "Remise appliquee", "Montant du",
    "TVA incluse", "Date d'echeance", "Reference client", "Solde restant",
    "Payer avant le", "Aucun remboursement", "Prix unitaire", "Mode de paiement",
    "A regler sous 30 jours", "Merci de votre confiance", "Exemplaire client",
]
_PHRASES_EN = [
    "Thank you", "Total amount", "Balance due", "Amount paid", "Please come again",
    "Paid in full", "Sub total", "Grand total", "Cash change", "Tax invoice",
    "No refund", "Due on receipt", "Customer copy", "Order number", "Payment received",
    "Thank you come again", "Price inclusive of tax", "Amount tendered",
    "Have a nice day", "Goods sold are not returnable",
]

# --- Names (people / store names): ASCII, multi-origin --------------------------
_NAMES = [
    "Martin Dubois", "Jean Leroy", "Sarah Tan", "Ahmad Bin Ali", "Wong Mei",
    "Lee Chong Wei", "Marie Petit", "David Lim", "Nur Aisyah", "Rajesh Kumar",
    "Sophie Bernard", "Paul Moreau", "Chen Wei", "Fatimah Zahra", "Kumar Sons",
    "Bar Wang Rice", "Unihakka Intl", "Global Trading", "Sunrise Mart", "Le Comptoir",
]

# --- Characters / short marks ----------------------------------------------------
_MARKS = ["N/A", "OK", "X", "--", "TBD", "VOID", "COPY", "PAID", "n/a", "***",
          "#", "-", "/", "@", "%", "No.", "Ref.", "Qty", "Tel."]

# Abbreviated months (dates like "12 Mar 2018", common on receipts)
_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# Target length (number of characters) by the region's size class.
_SIZE_NCHARS = {"small": (2, 6), "medium": (4, 9), "large": (6, 13), "very_large": (9, 18)}


def _ascii(s: str) -> str:
    """Transliterate to pure ASCII (é->e, ç->c, …) and drop any non-ASCII remainder, so
    that cv2.putText (Hershey, ASCII only) never produces '???'."""
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", "ignore").decode("ascii")
    return "".join(ch for ch in s if 32 <= ord(ch) <= 126)


def _pick(rng, seq):
    return seq[int(rng.integers(0, len(seq)))]


# --- Procedural generators (infinite diversity) ---------------------------------
def _amount(rng, big=False) -> str:
    """Neutral ASCII amount: thousands ',' + decimal '.' (e.g. 12,574.05). No
    currency symbol (the '€' gave '???'; we stay currency-neutral)."""
    digits = int(rng.integers(3, 7)) if big else int(rng.integers(1, 4))
    whole = int(rng.integers(1, 10 ** digits))
    return f"{whole:,}.{int(rng.integers(0, 100)):02d}"


def _date(rng) -> str:
    d = int(rng.integers(1, 29)); m = int(rng.integers(1, 13)); y = int(rng.integers(1995, 2026))
    fmt = int(rng.integers(0, 4))
    if fmt == 0:
        return f"{d:02d}/{m:02d}/{y}"
    if fmt == 1:
        return f"{d:02d}-{m:02d}-{y}"
    if fmt == 2:
        return f"{d:02d} {_MONTHS[m - 1]} {y}"
    return f"{_MONTHS[m - 1]} {d}, {y}"


def _integer(rng, lo_d=1, hi_d=6) -> str:
    d = int(rng.integers(lo_d, hi_d + 1))
    d = max(1, d)
    return str(int(rng.integers(10 ** (d - 1), 10 ** d)))


def _code(rng) -> str:
    letters = "".join(_pick(rng, list(string.ascii_uppercase))
                      for _ in range(int(rng.integers(2, 5))))
    style = int(rng.integers(0, 4))
    num = int(rng.integers(100, 999999))
    if style == 0:
        return f"{letters}-{num}"
    if style == 1:
        return f"{letters}#{num}"
    if style == 2:
        return f"No. {num}"
    return f"{letters}{num}"


def _word(rng) -> str:
    return _pick(rng, _WORDS_FR if rng.integers(0, 2) == 0 else _WORDS_EN)


def _phrase(rng) -> str:
    return _pick(rng, _PHRASES_FR if rng.integers(0, 2) == 0 else _PHRASES_EN)


# Categories allowed per size class (shortest to longest).
_BY_SIZE = {
    "small":      ["mark", "int_s", "amount_s", "word", "code"],
    "medium":     ["amount", "date", "word", "code", "int", "mark"],
    "large":      ["word", "code", "name", "amount_big", "date", "phrase"],
    "very_large": ["phrase", "name", "code", "amount_big", "date"],
}


def plausible_token(rng, size_class) -> str:
    """Plausible ASCII value (guaranteed free of '???') drawn from a varied corpus
    FR/EN/date/numbers/code/character/phrase, adapted to the size class."""
    cats = _BY_SIZE.get(size_class, _BY_SIZE["medium"])
    cat = _pick(rng, cats)
    lo, hi = _SIZE_NCHARS.get(size_class, (3, 8))

    if cat == "mark":
        tok = _pick(rng, _MARKS)
    elif cat == "int_s":
        tok = _integer(rng, 1, max(1, hi - 2))
    elif cat == "int":
        tok = _integer(rng, 1, 6)
    elif cat == "amount_s":
        tok = _amount(rng, big=False)
    elif cat == "amount":
        tok = _amount(rng, big=bool(rng.integers(0, 2)))
    elif cat == "amount_big":
        tok = _amount(rng, big=True)
    elif cat == "date":
        tok = _date(rng)
    elif cat == "code":
        tok = _code(rng)
    elif cat == "name":
        tok = _pick(rng, _NAMES)
    elif cat == "phrase":
        tok = _phrase(rng)
    else:  # "word"
        tok = _word(rng)

    return _ascii(tok)
