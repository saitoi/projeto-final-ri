import re
import unicodedata
from bs4 import BeautifulSoup

# Regex patterns
LETTER = r"[^\W\d_]"
DASH = r"(?<=[\w])-(?=[\w])"
DOT = r"(?<=[\w])\.(?=[\w])"
APOS = rf"(?<={LETTER})'(?={LETTER})"
SYMBOLS = r"[#:\"()\[\]{}~=_|;<>*\$`\\]"


def extract_text(html: str) -> str:
    """Extract plain text from HTML."""
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text()
    return text


def remove_accents(text: str) -> str:
    """Remove accents from text using Unicode normalization."""
    if text is None:
        return text
    nfd = unicodedata.normalize("NFD", text)
    return ''.join(char for char in nfd if unicodedata.category(char) != 'Mn')


def remove_non_ascii(text: str) -> str | None:
    """Remove non-ASCII characters from text."""
    if text is None:
        return None
    return text.encode('ascii', 'ignore').decode('ascii')


def clean_quotes(text: str) -> str | None:
    """Clean quotes while preserving apostrophes in contractions."""
    if text is None:
        return None
    ph = "\uffff"
    t = re.sub(APOS, ph, text)
    t = re.sub(r"'+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t.replace(ph, "'")


def clean_commas(text: str) -> str | None:
    """Clean commas while preserving numeric separators."""
    if text is None:
        return None
    ph = "\uffff"
    num_comma = r"(?<=\d),(?=\d)"
    t = re.sub(num_comma, ph, text)
    t = re.sub(r",+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t.replace(ph, ",")


def clean_dashes(text: str) -> str | None:
    """Clean dashes while preserving compound words."""
    if text is None:
        return None
    ph = "\uffff"
    t = re.sub(DASH, ph, text)
    t = re.sub(r"-+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t.replace(ph, "-")


def clean_symbols(text: str) -> str | None:
    """Remove special symbols from text."""
    if text is None:
        return None
    t = re.sub(SYMBOLS, " ", text)
    t = re.sub(r"\s+", " ", t).strip()
    return t
