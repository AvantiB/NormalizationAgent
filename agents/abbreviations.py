# agents/abbreviation.py

import pandas as pd
from utils.text import tokenize_preserve


class AbbreviationResolver:
    def __init__(
        self,
        csv_path: str | None,
        expand_only_uppercase: bool = True,
        standalone_only: bool = True,
    ):
        self.map = {}
        self.expand_only_uppercase = expand_only_uppercase
        self.standalone_only = standalone_only

        if csv_path:
            self._load(csv_path)

    def _load(self, csv_path):
        df = pd.read_csv(csv_path)

        for _, row in df.iterrows():
            abbr = str(row["Abbreviation/Shorthand"]).strip()
            meaning = str(row["Meaning"]).strip()

            if not abbr or not meaning:
                continue

            # Enforce uppercase-only if requested
            if self.expand_only_uppercase and not abbr.isupper():
                continue

            # Avoid single-character abbreviations
            if len(abbr) < 2:
                continue

            if abbr not in self.map:
                self.map[abbr] = meaning
            elif self.map[abbr] != meaning:
                # ambiguous abbreviation → drop
                self.map[abbr] = None

        # Remove ambiguous ones
        self.map = {k: v for k, v in self.map.items() if v}

    def expand(self, text: str) -> str:
        if not self.map:
            return text

        tokens = tokenize_preserve(text)

        expanded = []
        for tok in tokens:
            if self.standalone_only and tok in self.map:
                expanded.append(self.map[tok])
            else:
                expanded.append(tok)

        return " ".join(expanded)
