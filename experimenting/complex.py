from math import *
import cmath
import sys

sys.setrecursionlimit(1000000)

def func(t0, tmax, x0, xmax, step, f):
    if t0 >= tmax:
        result = []
        for i in range(int((xmax - x0) / step)):
            x = i * step + x0
            result += [(t0, x, f(x, t0))]
        return result
    else:
        result = []
        for i in range(int((xmax - x0) / step)):
            x = i * step + x0
            result += [(t0, x, f(x, t0))]
        return result + func(t0 + step, tmax, x0, xmax, step, f)

def wavefunc(x):
    denominator = (2j)**(1/4);
    scalar = sqrt(2) / 4
    exponent = x * (1 + 1j)

    print(x, scalar, exponent, denominator)
    return scalar * (cmath.exp(exponent) - cmath.exp(-exponent)) / denominator

results = func(1, 0, 0, 100, 3.1415 / 128, lambda x, t: wavefunc(x))

output = "# x t real imag"

previoustime = results[0][0]
output = f"# x t real imag\n# time = {previoustime}\n"


for (time, x, result) in results:
    if time != previoustime:
        output += f"\n\n# time = {time}\n"

    output += f"{x} {result.real} {result.imag} \n"
    previoustime = time

with open("out.txt", "w+") as f:
    f.write(output)
