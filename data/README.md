# Synthetic transcript data

`transcripts.jsonl` is fictional, hand-authored data released with this repository
under its MIT license. Names, organizations, and events are invented; resemblance to
real people or organizations is coincidental. Offsets use Unicode Python string indices
and half-open `[start, end)` bounds. Gold labels intentionally stay within CoNLL's four
entity classes. Notes identify examples designed to expose domain mismatch.

The corpus includes conventional positives, repeated values, Unicode, lowercase,
all-caps and erratic casing, punctuation-adjacent boundaries, names with internal
punctuation, spans at transcript edges, and hard negatives. Generic roles, directions,
seasons, and interface values are not entities. A named product may be MISCELLANEOUS;
generic account or support language is not. These are deliberately constructed challenge
cases, not a representative sample of callers, languages, domains, or error frequencies.
