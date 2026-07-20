"""lexicon.py — corpus de valeurs plausibles injectées par la substitution.

Le forger peint sur le document une "valeur éditée" (ce qu'un fraudeur écrirait).
Ce module fournit un GROS corpus varié — français / anglais / dates / chiffres /
codes / caractères / phrases — pour que le contenu soit **très diversifié**.

Pourquoi la diversité aide : si toutes les substitutions se ressemblaient (ex. un
montant en euro), le modèle pourrait apprendre ce CONTENU comme raccourci au lieu du
signal de compression ELA. En variant fortement le texte injecté, on force le modèle
à s'appuyer sur l'incohérence de compression, pas sur ce qui est écrit.

Contrainte technique : le texte est dessiné avec les polices **Hershey d'OpenCV**
(`cv2.putText`), qui ne rendent **que l'ASCII**. Tout glyphe hors ASCII (ex. '€',
accents 'é è ç') s'afficherait en '???'. -> `_ascii(...)` translittère tout en ASCII
(é->e, ç->c, …) à la sortie, donc le corpus source peut contenir des accents sans
jamais produire de '???'.

API : `plausible_token(rng, size_class) -> str` (rng = numpy.random.Generator).
"""
from __future__ import annotations

import string
import unicodedata

# --- Mots (FR / EN) : vocabulaire de facture / reçu / document commercial --------
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

# --- Phrases courtes (FR / EN) : mentions typiques d'un document -----------------
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

# --- Noms (personnes / enseignes) : ASCII, multi-origines -----------------------
_NAMES = [
    "Martin Dubois", "Jean Leroy", "Sarah Tan", "Ahmad Bin Ali", "Wong Mei",
    "Lee Chong Wei", "Marie Petit", "David Lim", "Nur Aisyah", "Rajesh Kumar",
    "Sophie Bernard", "Paul Moreau", "Chen Wei", "Fatimah Zahra", "Kumar Sons",
    "Bar Wang Rice", "Unihakka Intl", "Global Trading", "Sunrise Mart", "Le Comptoir",
]

# --- Caractères / marques courtes ------------------------------------------------
_MARKS = ["N/A", "OK", "X", "--", "TBD", "VOID", "COPY", "PAID", "n/a", "***",
          "#", "-", "/", "@", "%", "No.", "Ref.", "Qty", "Tel."]

# Mois abrégés (dates style "12 Mar 2018", fréquent sur les reçus)
_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# Longueur cible (nb de caractères) selon la classe de taille de la zone.
_SIZE_NCHARS = {"small": (2, 6), "medium": (4, 9), "large": (6, 13), "very_large": (9, 18)}


def _ascii(s: str) -> str:
    """Translittère en ASCII pur (é->e, ç->c, …) et retire tout reste non-ASCII, pour
    que cv2.putText (Hershey, ASCII only) ne produise jamais de '???'."""
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", "ignore").decode("ascii")
    return "".join(ch for ch in s if 32 <= ord(ch) <= 126)


def _pick(rng, seq):
    return seq[int(rng.integers(0, len(seq)))]


# --- Générateurs procéduraux (diversité infinie) --------------------------------
def _amount(rng, big=False) -> str:
    """Montant ASCII neutre : milliers ',' + décimale '.' (ex. 12,574.05). Pas de
    symbole monétaire (le '€' donnait '???' ; on reste neutre en devise)."""
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


# Catégories autorisées par classe de taille (du plus court au plus long).
_BY_SIZE = {
    "small":      ["mark", "int_s", "amount_s", "word", "code"],
    "medium":     ["amount", "date", "word", "code", "int", "mark"],
    "large":      ["word", "code", "name", "amount_big", "date", "phrase"],
    "very_large": ["phrase", "name", "code", "amount_big", "date"],
}


def plausible_token(rng, size_class) -> str:
    """Valeur plausible ASCII (garantie sans '???') tirée d'un corpus varié
    FR/EN/date/chiffres/code/caractere/phrase, adaptée à la classe de taille."""
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
