import matplotlib
import numpy
import numpy as np
import sympy as sym
from Helpers import identifier, isCharacter
import math
from numpy import matrix, array, mean, std, max, linspace, ones, sin, cos, tan, arctan, pi, sqrt, exp, arcsin, arccos, arctan2, sinh, cosh, zeros, log, diag, linspace, arange
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, show, xlabel, ylabel, legend, title, savefig, errorbar, grid
import scipy.optimize as opt
from GPII import *
from math import sqrt
pi = math.pi
import scipy


matplotlib.rc('xtick', labelsize=20)
matplotlib.rc('ytick', labelsize=20)







def gauss(term):
    ids = identifier(term)
    symbols = []
    for str1 in ids:
        symbols.append(sym.sympify(str1))
    termSymbol = sym.sympify(term)
    values = []
    for ident in ids:
        exec("values.append(" + ident + ")")

    derivatives = []
    i = 0
    while i < len(symbols):
        r = sym.diff(termSymbol, symbols[i])
        j = 0
        while j < len(symbols):
            # exec('r.evalf(subs={symbols[j]: ' + values[j] + '})')
            r = r.evalf(subs={symbols[j]: values[j]})
            j += 1
        derivatives.append(r.evalf())
        i += 1
    i = 0
    while i < len(derivatives):
        exec("derivatives[i] *= sigma_" + ids[i])
        i = i + 1
    res = 0
    for z in derivatives:
        res += z ** 2
    return math.sqrt(res)

def gaussVec(term):
    ids = identifier(term)
    arrays = []
    for i in range(len(ids)):
        if isinstance(eval(ids[i]), np.ndarray):
            arrays.append(ids[i])
    arrayLength = len(eval(arrays[0]))
    symbols = []
    for str1 in ids:
        symbols.append(sym.sympify(str1))
    termSymbol = sym.sympify(term)
    res = []
    for k in range(arrayLength):
        values = []
        for ident in ids:
            if ident in arrays:
                exec("values.append(" + ident + "[k]" + ")")
            else:
                exec("values.append(" + ident + ")")
        derivatives = []
        i = 0
        while i < len(symbols):
            r = sym.diff(termSymbol, symbols[i])
            j = 0
            while j < len(symbols):
                r = r.evalf(subs={symbols[j]: values[j]})
                j += 1
            derivatives.append(r.evalf())
            i += 1
        i = 0
        sigmaArrays = []
        for t in range(len(ids)):
            if isinstance(eval("sigma_" + ids[t]), np.ndarray):
                sigmaArrays.append(ids[t])
        while i < len(derivatives):
            if ids[i] in sigmaArrays:
                exec("derivatives[i] *= sigma_" + ids[i] + "[k]")
            else:
                exec("derivatives[i] *= sigma_" + ids[i])
            i = i + 1
        resj = 0
        for z in derivatives:
            resj += z ** 2
        res.append(sqrt(resj))
    return array(res)

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

from importieren import Walu1

x = Walu1[:, 1]
t = Walu1[:, 0]
Fs = 500

L = len(x)

#t = x[:, 0]

Y = np.fft.fft(x)


f = Fs/L*np.arange(0, L/2)
P2 = abs(Y)/L*2
P1 = P2[0:L//2+1]


plot(f, P1, label='Alu1', linewidth=5)
reso = f[np.argmax(P1[10:]) + 10]
plt.axvline(x=reso, color='r', linestyle='--', label='Resonanz')
plt.xlim(0, 30)
plt.ylim(0, 0.26)
xlabel('Frequenz in Hz', fontsize=20)
ylabel('Amplitude in V', fontsize=20)
title('Alu1, Frequenzdomäne', fontsize=20)
legend(fontsize=13, loc='upper right')
grid()
plt.tight_layout()
savefig('alu1F')
show()

#berechne emodulus


def decay(x, A, beta, c):
    return A*exp(-beta*x) + c


s = x*25#in mm dann
plot(t, s, label='Alu1', linewidth=2)
peaks, _ = scipy.signal.find_peaks(s)
optimizedParameters1, s = opt.curve_fit(decay, t[peaks], s[peaks])
plot(t[peaks], decay(t[peaks], *optimizedParameters1), label="fit1")
xlabel('Zeit in s', fontsize=20)
ylabel('Position in mm', fontsize=20)
title('Alu1, Zeitdomäne', fontsize=20)
legend(fontsize=13, loc='upper right')
grid()
plt.tight_layout()
savefig('alu1T')
show()

betaAlu1 = optimizedParameters1[1]
sigma_betaAlu1 = diag(s)[1]












