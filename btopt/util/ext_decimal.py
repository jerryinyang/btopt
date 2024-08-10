import math
from decimal import Decimal


class ExtendedDecimal(Decimal):
    def _convert_operand(self, other):
        if isinstance(other, (int, float, str)):
            return ExtendedDecimal(str(other))
        return other

    def __add__(self, other):
        return ExtendedDecimal(super().__add__(self._convert_operand(other)))

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return ExtendedDecimal(super().__sub__(self._convert_operand(other)))

    def __rsub__(self, other):
        return -self + other

    def __mul__(self, other):
        return ExtendedDecimal(super().__mul__(self._convert_operand(other)))

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return ExtendedDecimal(super().__truediv__(self._convert_operand(other)))

    def __rtruediv__(self, other):
        return ExtendedDecimal(self._convert_operand(other) / self)

    def __floordiv__(self, other):
        return ExtendedDecimal(super().__floordiv__(self._convert_operand(other)))

    def __rfloordiv__(self, other):
        return ExtendedDecimal(self._convert_operand(other) // self)

    def __mod__(self, other):
        return ExtendedDecimal(super().__mod__(self._convert_operand(other)))

    def __rmod__(self, other):
        return ExtendedDecimal(self._convert_operand(other) % self)

    def __pow__(self, other):
        return ExtendedDecimal(super().__pow__(self._convert_operand(other)))

    def __rpow__(self, other):
        return ExtendedDecimal(pow(self._convert_operand(other), self))

    def __neg__(self):
        return ExtendedDecimal(super().__neg__())

    def __abs__(self):
        return ExtendedDecimal(super().__abs__())

    def __eq__(self, other):
        return super().__eq__(self._convert_operand(other))

    def __lt__(self, other):
        return super().__lt__(self._convert_operand(other))

    def __le__(self, other):
        return super().__le__(self._convert_operand(other))

    def __gt__(self, other):
        return super().__gt__(self._convert_operand(other))

    def __ge__(self, other):
        return super().__ge__(self._convert_operand(other))

    def __float__(self):
        return float(super().__float__())

    def __int__(self):
        return int(super().__int__())

    def __round__(self, n=None):
        return ExtendedDecimal(round(super().__round__(n)))

    def sqrt(self):
        return ExtendedDecimal(self.sqrt())

    @classmethod
    def from_float(cls, f):
        """
        Convert a float to ExtendedDecimal with proper precision.
        """
        return cls("{0:.15g}".format(f))


# Additional utility methods
def ed_sin(x):
    return ExtendedDecimal(math.sin(float(x)))


def ed_cos(x):
    return ExtendedDecimal(math.cos(float(x)))


def ed_tan(x):
    return ExtendedDecimal(math.tan(float(x)))


def ed_exp(x):
    return ExtendedDecimal(math.exp(float(x)))


def ed_log(x):
    return ExtendedDecimal(math.log(float(x)))


if __name__ == "__main__":
    # # Example usage
    # a = ExtendedDecimal("10.5")
    # b = a + 5  # ExtendedDecimal + int
    # c = 7.5 - a  # float - ExtendedDecimal
    # d = a * 2.5  # ExtendedDecimal * float
    # e = 20 / a  # int / ExtendedDecimal
    # f = a**2  # ExtendedDecimal ** int
    # g = ed_sin(a)  # Sine of ExtendedDecimal

    # print(f"a: {a}, type: {type(a)}")
    # print(f"b: {b}, type: {type(b)}")
    # print(f"c: {c}, type: {type(c)}")
    # print(f"d: {d}, type: {type(d)}")
    # print(f"e: {e}, type: {type(e)}")
    # print(f"f: {f}, type: {type(f)}")
    # print(f"g: {g}, type: {type(g)}")

    print(ExtendedDecimal.ROUNDNUM)
