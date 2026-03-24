import hashlib
import re
from typing import List

import pandas as pd

from float_categorization.constant import NAME_TO_ID, NAME_UNKNOWN_ID


class FeatureProcessor:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.df.columns = self.df.columns.str.strip().str.casefold()

    # Payee keyword patterns for domain-specific categories.
    # These capture signals that survive even when the payee is unseen.
    PAYEE_KEYWORD_FLAGS = {
        "payee_k_travel": (
            r"(?i)(?:hotel|motel|resort|fairmont|regis|marriott|hilton|hyatt"
            r"|intercontinental|pinnacle|bayleaf|winford|shoreline|best western"
            r"|edition foh|london edition|red fox"
            r"|air can|westjet|airasia|air.?india|cebu air|singaporeair|harbour air"
            r"|flair air|porter air|swoop"
            r"|sixt|hertz|avis|budget rent|lyft|uber.*trip|taxi|charter bus"
            r"|enterprise rent|evo car share"
            r"|parking|yvr|airport|expedia|booking\.com|airbnb|ibibo"
            r"|manulife travel|whistler rides"
            # Vancouver hotels
            r"|pan pacific|shangri.?la|wedgewood|loden|rosewood"
            r"|sutton place|sandman|four seasons|westin|sheraton"
            r"|holiday inn|hampton inn|radisson|novotel"
            # Delhi / Gurgaon hotels
            r"|oberoi\b|leela\s+\w+|taj\s*palace|taj\s*mahal|itc\s*maurya"
            r"|claridges|the\s*lalit|le\s*meridien|trident\b|the\s*lodhi"
            r"|crowne plaza|hyatt\s*regency|pullman\b"
            r"|lemon\s*tree|country\s*inn|roseate\b|the\s*ashok"
            r"|makemytrip|goibibo|cleartrip)"
        ),
        "payee_k_food": (
            r"(?i)(?:restaurant|cafe|caffe|cactus|joey|dirty apron"
            r"|richmond station|uber eat|doordash|skip.*dish"
            r"|grubhub|starbucks|tim horton|mcdonald|subway"
            r"|pavilion|straker|dover street|ramen|sushi|pizza"
            r"|kitchen|bistro|grill|diner|eatery|tavern|pub\b|bar\b"
            r"|bigtr?ee entertain|nero|hudson.*arpt"
            r"|liquor|bakery|coffee|ice cream|hotpot|hot pot"
            r"|keg\b|donuts|chocolat|swiggy|grab\b|domino"
            r"|breka|fortes|nightingale|loose moose|tacofino"
            r"|autostrada|folke|raku|central taps"
            # Vancouver corporate dining
            r"|hawksworth|gotham|miku\b|minami|chambar"
            r"|earls\b|white spot|steamworks|flying pig"
            r"|joe fortes|glowbal"
            r"|boston pizza|milestone|moxies|original joe"
            # Delhi / Gurgaon dining
            r"|haldiram|sagar ratna|bikanervala|zomato"
            r"|swiggy|berco.?s|saravana bhavan|moti mahal"
            r"|bukhara|dum pukht|karim.?s|pind balluchi"
            r"|dhaba\b|barbeque nation|social\s+offline"
            r"|cyber hub|punjabi by nature|olive\s+bar"
            r"|catering|fresh prep|fantuan)"
        ),
        "payee_k_software": (
            r"(?i)(?:aws|google.*cloud|azure|github|gitlab|slack|asana"
            r"|jira|atlassian|zoom|dropbox|adobe|figma|canva"
            r"|notion|browserstack|grammarly|anthropic|claude"
            r"|openai|cursor|murf\.ai|namecheap|name-cheap"
            r"|google.*workspace|linkedin.*prea|middleware)"
        ),
        "payee_k_amazon": r"(?i)(?:amazon|amzn)",
    }

    def __call__(self, *args, **kwds):
        self.extract_date_features()
        self.encode_payee()

        cols_to_drop = ["date", "funding account", "currency"]
        self.drop_columns(cols_to_drop)

        name_based_features = ["spender", "approver", "payment authorizer"]
        # self.drop_columns(name_based_features)
        self.encode_name_based_fields(name_based_features)

        self.one_hot_encode_only_for_null_values()
        self.add_detail_null_flag()
        self.process_further()

        return self.df

    def drop_columns(self, cols_to_drop: List[str]):
        self.df.drop(cols_to_drop, axis=1, inplace=True, errors="ignore")

    def one_hot_encode_only_for_null_values(self):
        cols_to_one_hot = [
            "account",
            "initial amount",
            "initial currency",
            "last 4 digits",
            "invoice number",
            "fx initial currency",
            "fx initial amount",
            "conversion rate",
        ]

        for col in cols_to_one_hot:
            if col in self.df.columns:
                self.df[col + "_is_null"] = self.df[col].isnull().astype(int)

        self.drop_columns(cols_to_one_hot)

    def add_detail_null_flag(self):
        if "transaction detail" in self.df.columns:
            self.df["detail_is_null"] = (
                self.df["transaction detail"].isnull().astype(int)
            )

    def one_hot_encode(self, cols: List[str]):
        dummies = pd.get_dummies(
            self.df[cols], columns=cols, drop_first=False, dtype=int
        )
        self.df = pd.concat([self.df, dummies], axis=1)

    @staticmethod
    def _normalize_name(name) -> str:
        """Lowercase, collapse whitespace, keep only first and last name."""
        if pd.isna(name):
            return ""
        name = str(name).strip().casefold()
        name = re.sub(r"\s+", " ", name)
        parts = name.split()
        if len(parts) > 2:
            # Keep first + last to unify "hector alfonso pertierra marin" style
            # but check the full name first in the dictionary
            return name  # return full; lookup will try full then first+last
        return name

    @staticmethod
    def _name_to_id(name: str) -> int:
        """Map a single normalized name string to its integer ID."""
        if not name:
            return 0
        if name in NAME_TO_ID:
            return NAME_TO_ID[name]
        # Try first + last only (handles middle-name mismatch)
        parts = name.split()
        if len(parts) > 2:
            short = f"{parts[0]} {parts[-1]}"
            if short in NAME_TO_ID:
                return NAME_TO_ID[short]
        return NAME_UNKNOWN_ID

    def extract_date_features(self):
        """Extract month and day-of-week before the date column is dropped."""
        if "date" in self.df.columns:
            dt = pd.to_datetime(self.df["date"], errors="coerce")
            self.df["month"] = dt.dt.month.fillna(0).astype(int)
            self.df["day_of_week"] = dt.dt.dayofweek.fillna(0).astype(int)

    @staticmethod
    def _normalize_payee(x) -> str:
        """Normalize payee: uppercase, strip reference IDs and trailing tokens."""
        if pd.isna(x):
            return "UNKNOWN"
        s = str(x).strip().upper()
        if s in {"", "NONE", "NAN"}:
            return "UNKNOWN"
        s = re.sub(r"\*.*$", "", s)  # Remove everything after *
        s = re.sub(r"\s+[A-Z0-9]{8,}$", "", s)  # Remove trailing long IDs
        s = re.sub(r"\s+", " ", s)
        return s.strip()

    @staticmethod
    def _stable_hash(s: str, modulus: int = 10000) -> int:
        """Deterministic hash stable across Python processes."""
        return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16) % modulus

    def encode_payee(self):
        """Create hash-encoded payee feature + keyword flags from payee name."""
        if "payee" not in self.df.columns:
            return

        payee_norm = self.df["payee"].apply(self._normalize_payee)
        self.df["payee_encoded"] = payee_norm.apply(self._stable_hash)

        # Add keyword flags based on raw payee (before normalization strips info)
        payee_lower = self.df["payee"].fillna("").astype(str).str.lower()
        for flag_name, pattern in self.PAYEE_KEYWORD_FLAGS.items():
            self.df[flag_name] = payee_lower.str.contains(
                pattern, regex=True, na=False
            ).astype(int)

    def encode_name_based_fields(self, cols: List[str]):
        # First pass: normalize and build a per-row cache so the same person
        # appearing in multiple columns on the same row gets the same ID.
        for col in cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].apply(self._normalize_name)

        for col in cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].apply(self._name_to_id)

        # Ensure integer dtype
        for col in cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(int)

    def process_further(self):
        def parse_amount(val):
            val = str(val).strip()
            if not val or val.lower() == "nan":
                return 0.0
            is_negative = val.startswith("(") and val.endswith(")")
            val = val.strip("()")
            val = val.replace(",", "")
            result = float(val)
            return -result if is_negative else result

        self.df["amount_parsed"] = self.df["amount"].apply(parse_amount)
        self.df.drop("amount", axis=1, inplace=True)

        self.fil_null_values_numer()
        self.fill_null_values_text(col_names=["transaction type"])

    def fil_null_values_numer(self, fill_with: float = 0.0):
        col_names = ["initial amount", "fx initial amount", "conversion rate"]
        for col in col_names:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna(fill_with)

    def fill_null_values_text(self, col_names: List[str], fill_with: str = "unknown"):
        for col in col_names:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna(fill_with)

        self.one_hot_encode(col_names)
