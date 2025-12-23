import re


STOPWORDS = {
     "a", "an", "and", "are", "as","at",
    "be","but","by","due",
    "for", "from", "if", "in","into","is","it",
    "no","not","of","on","or",
    "such","that","the","their","then","there","these","they","this","to",
    "was","will","with","without"
}

def normalize_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    return " ".join(s.strip().lower().split())


def token_overlap_score(query: str, candidate: str) -> float:
    q_tokens = [t for t in query.split() if t not in STOPWORDS]
    c_tokens = [t for t in candidate.split() if t not in STOPWORDS]
    if not q_tokens or not c_tokens:
        return 0.0
    inter = len(set(q_tokens) & set(c_tokens))
    return inter / len(q_tokens)

def tokenize_preserve(text: str):
    return re.findall(r"[A-Za-z]+|\d+|[^\w\s]", text)
