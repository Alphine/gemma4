import pytest

from toypkg import Span, clip, count_words

WORDS = "a bb ccc"


def test_clip_basic():
    assert clip("abcdef", 2) == "ab"


def test_clip_empty():
    assert clip("", 2) == ""


@pytest.mark.parametrize("n,expected", [(1, 3), (2, 2), (3, 1)])
def test_count_words_min_len(n, expected):
    assert count_words(WORDS, min_len=n) == expected


class TestSpan:
    def test_width(self):
        s = Span(2, 7)
        assert s.width() == 5

    def test_contains_end_excluded(self):
        assert not Span(0, 3).contains(3)

    def test_bad_span(self):
        with pytest.raises(ValueError):
            Span(5, 1)
