"""Tiny text helpers used as a mutation-testing fixture."""


def clip(s, n=3):
    """Return the first n characters of s ('' for n <= 0)."""
    if not s or n <= 0:
        return ""
    return s[:n]


def count_words(text, min_len=1):
    """Count words with at least min_len characters."""
    total = 0
    for word in text.split():
        if len(word) >= min_len:
            total += 1
    return total


class Span:
    """Half-open integer span [start, end)."""

    def __init__(self, start, end):
        if end < start:
            raise ValueError("end before start")
        self.start = start
        self.end = end

    def width(self):
        return self.end - self.start

    def contains(self, x):
        return self.start <= x < self.end
