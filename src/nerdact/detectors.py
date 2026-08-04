"""Conservative deterministic detectors for structurally checkable PII."""

from __future__ import annotations

import ipaddress
import re
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass

from .schema import Entity

_EMAIL = re.compile(
    r"(?<![\w.!#$%&'*+/=?^`{|}~-])"
    r"[\w.!#$%&'*+/=?^`{|}~-]+@(?:[\w](?:[\w-]{0,61}[\w])?\.)+[\w](?:[\w-]{0,61}[\w])?",
    re.UNICODE,
)
_URL = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)
_PHONE = re.compile(
    r"(?<!\w)(?:\+\d{1,3}[ .-]*)?(?:\(\d{2,4}\)|\d{2,4})"
    r"[ .-]+\d{2,4}[ .-]+\d{4}(?:\s*(?:ext\.?|x)\s*\d{1,6})?(?!\w)",
    re.IGNORECASE,
)
_IP = re.compile(
    r"(?<![\w:])(?:[0-9A-Fa-f]{0,4}:){2,7}[0-9A-Fa-f]{0,4}(?![\w:])"
    r"|(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?!\d|\.\d)"
)
_CARD = re.compile(r"(?<![\w-])(?:\d[ -]?){12,18}\d(?![\w-])")
_ROUTING = re.compile(r"(?<!\d)\d{9}(?!\d)")
_SSN = re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")
_DEVICE = re.compile(
    r"(?<![\w-])(?:IMEI-\d{15}|(?:DEV|NODE)-[A-Z0-9]+(?:-[A-Z0-9]+)+)(?![\w-])",
    re.IGNORECASE,
)
_API_KEY = re.compile(r"(?<!\w)sk_(?:test|live)_[A-Za-z0-9_]{8,}(?!\w)")
_TRAILING_URL_PUNCTUATION = ".,;:!?"
_URL_DELIMITERS = {")": "(", "]": "[", "}": "{"}


@dataclass(frozen=True, slots=True)
class _Rule:
    name: str
    label: str
    matches: Callable[[str], Iterable[tuple[int, int]]]


def _regex_matches(pattern: re.Pattern[str], text: str) -> Iterator[tuple[int, int]]:
    for match in pattern.finditer(text):
        yield match.span()


def _email_matches(text: str) -> Iterator[tuple[int, int]]:
    for match in _EMAIL.finditer(text):
        start, end = match.span()
        local_part = text[start : text.index("@", start, end)]
        if local_part.casefold().startswith("email="):
            start += len("email=")
        yield start, end


def _url_matches(text: str) -> Iterator[tuple[int, int]]:
    for match in _URL.finditer(text):
        start, end = match.span()
        while end > start:
            last = text[end - 1]
            if last in _TRAILING_URL_PUNCTUATION:
                end -= 1
                continue
            opener = _URL_DELIMITERS.get(last)
            candidate = text[start:end]
            if opener is not None and candidate.count(last) > candidate.count(opener):
                end -= 1
                continue
            break
        if end > start:
            yield start, end


def _ip_matches(text: str) -> Iterator[tuple[int, int]]:
    for match in _IP.finditer(text):
        try:
            ipaddress.ip_address(match.group())
        except ValueError:
            continue
        yield match.span()


def _phone_matches(text: str) -> Iterator[tuple[int, int]]:
    for match in _PHONE.finditer(text):
        value = match.group()
        prefix = text[max(0, match.start() - 24) : match.start()].casefold()
        if value.startswith("+") or "(" in value or re.search(
            r"\b(?:call|phone|number|reach)\b", prefix
        ):
            yield match.span()


def _luhn_valid(value: str) -> bool:
    digits = [int(character) for character in value if character.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    parity = len(digits) % 2
    total = 0
    for index, digit in enumerate(digits):
        if index % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


def _card_matches(text: str) -> Iterator[tuple[int, int]]:
    for match in _CARD.finditer(text):
        if _luhn_valid(match.group()):
            yield match.span()


def _routing_matches(text: str) -> Iterator[tuple[int, int]]:
    for match in _ROUTING.finditer(text):
        prefix = text[max(0, match.start() - 32) : match.start()].casefold()
        if "routing" not in prefix:
            continue
        digits = [int(character) for character in match.group()]
        checksum = 3 * sum(digits[0::3]) + 7 * sum(digits[1::3]) + sum(digits[2::3])
        if checksum % 10 == 0:
            yield match.span()


_RULES = (
    _Rule("email", "EMAIL_ADDRESS", _email_matches),
    _Rule("url", "URL", _url_matches),
    _Rule("phone", "PHONE_NUMBER", _phone_matches),
    _Rule("ip-address", "IP_ADDRESS", _ip_matches),
    _Rule("payment-card-luhn", "FINANCIAL_ID", _card_matches),
    _Rule("routing-number-checksum", "FINANCIAL_ID", _routing_matches),
    _Rule("us-ssn-format", "GOVERNMENT_ID", lambda text: _regex_matches(_SSN, text)),
    _Rule("device-id-format", "DEVICE_ID", lambda text: _regex_matches(_DEVICE, text)),
    _Rule("stripe-key-prefix", "CREDENTIAL", lambda text: _regex_matches(_API_KEY, text)),
)


def detect_structured_pii(text: str) -> list[Entity]:
    """Return deterministic PII spans with the exact rule recorded as provenance."""
    entities: dict[tuple[int, int, str], Entity] = {}
    for rule in _RULES:
        for start, end in rule.matches(text):
            entity = Entity(
                start,
                end,
                rule.label,
                text[start:end],
                score=1.0,
                provenance=(f"detector:{rule.name}",),
            )
            previous = entities.get(entity.key())
            if previous is None:
                entities[entity.key()] = entity
            else:
                entities[entity.key()] = Entity(
                    start,
                    end,
                    rule.label,
                    text[start:end],
                    score=1.0,
                    provenance=tuple(sorted(set(previous.provenance + entity.provenance))),
                )
    return sorted(entities.values())
