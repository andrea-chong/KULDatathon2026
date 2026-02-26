import re
import pandas as pd


def parse_word_from_lexeme_string(s: str) -> str:
    """
    Parse `word` from `lexeme_string`.

    Rules you specified:
    - Normal case (no '<*sf>' marker): take everything before the first '<'
        'sua/seu<det><pos><f><sg>' -> 'sua/seu'
    - If '<*sf>' occurs: take everything before the second '<'
        '<*sf>/ter<vblex><pri><*pers><*numb>' -> '<*sf>/ter'
      (i.e., include the '<*sf>' prefix and the lemma after '/')
    """
    if pd.isna(s):
        return None
    s = str(s).strip()

    if "<*sf>" not in s:
        return s.split("<", 1)[0].strip()

    # With '<*sf>': return substring before the second '<'
    first = s.find("<")
    if first == -1:
        return s  # no tags at all
    second = s.find("<", first + 1)
    if second == -1:
        return s  # only one '<' found
    return s[:second].strip()

def parse_morph_tags_from_lexeme_string(s: str) -> str:
    """
    Parse `morphological annotations` from `lexeme_string`.
    e.g 'sua/seu<det><pos><f><sg>' -> ["<det>", "<pos>", "<f>", "<sg>"]
    """
    if pd.isna(s):
        return None
    s = str(s).strip()

    return re.findall("\<[^>]+\>", s)

# A fairly safe tokenizer for Portuguese-ish text: letters with diacritics + apostrophe + hyphen
TOKEN_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]+(?:[-'][A-Za-zÀ-ÖØ-öø-ÿ]+)*", re.UNICODE)

def tokens_from_translation(text: str) -> set[str]:
    if pd.isna(text):
        return set()
    toks = TOKEN_RE.findall(str(text).lower())
    return set(toks)