"""Accurate sums for enumeration, including sums far below the float range."""

import math

LOG10_2 = math.log10(2)
# Products smaller than this are rescaled by an exact power of two. Two numbers
# at least this large multiply to at least 2**-1000, safely above the smallest
# normal float (2**-1022), so no product ever underflows or loses precision.
RESCALE_BELOW = 2.0**-500


class ScaledSum:
    """A sum of terms value * 2**exponent that is accurate and never underflows.

    Enumeration rescales tiny world probabilities by exact powers of two and
    passes the exponent along. Terms are kept per exponent and folded with
    math.fsum every 64 additions, so a pass over a million worlds stays
    accurate in little memory. Exponents are combined, exactly, only when a
    value or a ratio is read, so a ratio such as P(event, evidence) /
    P(evidence) stays accurate even when both are far below the smallest float.
    Ordinary models never rescale, and then every term has exponent 0.
    """

    __slots__ = ("parts", "count")

    def __init__(self):
        self.parts = {}
        self.count = 0

    def add(self, value, exponent=0):
        self.count += 1
        terms = self.parts.get(exponent)
        if terms is None:
            self.parts[exponent] = [value]
        else:
            terms.append(value)
            if len(terms) > 64:
                self.parts[exponent] = [math.fsum(terms)]

    @classmethod
    def combined(cls, sums):
        """One sum holding every term of the given sums."""
        result = cls()
        for item in sums:
            result.count += item.count
            for exponent, terms in item.parts.items():
                result.parts.setdefault(exponent, []).append(math.fsum(terms))
        return result

    @property
    def rescaled(self):
        """Whether any term needed rescaling."""
        return any(exponent != 0 for exponent in self.parts)

    def _totals(self):
        return {exponent: math.fsum(terms) for exponent, terms in self.parts.items()}

    @staticmethod
    def _top(totals):
        """The largest exponent with a nonzero total, or None when the sum is zero."""
        nonzero = [exponent for exponent, total in totals.items() if total != 0]
        return max(nonzero) if nonzero else None

    @staticmethod
    def _relative(totals, top):
        """The sum divided by 2**top, combining the exponents exactly."""
        return math.fsum(math.ldexp(total, exponent - top) for exponent, total in totals.items())

    def is_zero(self):
        return self._top(self._totals()) is None

    def value(self):
        """The sum as a float. Below about 1e-308 it underflows to 0.0; see log10()."""
        totals = self._totals()
        top = self._top(totals)
        return 0.0 if top is None else math.ldexp(self._relative(totals, top), top)

    def log10(self):
        """log10 of a positive sum, accurate even far below the float range."""
        totals = self._totals()
        top = self._top(totals)
        return math.log10(self._relative(totals, top)) + top * LOG10_2

    def ratio(self, other):
        """self / other, accurate even when both sums are far below the float range."""
        totals = other._totals()
        top = self._top(totals)
        if top is None:
            raise ZeroDivisionError("ratio to a sum of zero")
        return self._relative(self._totals(), top) / self._relative(totals, top)


def describe(value, log10=None):
    """Format a probability, falling back on its log10 when the float underflowed."""
    if value >= 1e-300 or log10 is None or not math.isfinite(log10):
        return f"{value:.10g}"
    exponent = math.floor(log10)
    mantissa = 10 ** (log10 - exponent)
    if mantissa >= 9.9999999995:
        mantissa, exponent = 1.0, exponent + 1
    return f"{mantissa:.9f}e{exponent}"
