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
m = 8.25/1000
sigma_m = 0.01/1000
L = 0.3
sigma_L = 0.01

b = 0.5/1000
sigma_b = 0.05/1000
a = 2.01/100
sigma_a = 0.05/1000

beta1 = pi*0.59686/(L - 0.03)#einspannung 3cm
sigma_beta1 = gauss("pi*0.59686/(L - 0.03)")

omega = reso*2*pi
sigma_omega = 0.01*omega

EAlu1 = omega**2/beta1**4*m/L*12/a/b**3
sigma_EAlu1 = gauss("omega**2/beta1**4*m/L*12/a/b**3")



def decay(x, A, beta, c):
    return A*exp(-beta*(x)) + c


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



from importieren import Walu2
Walu2 = Walu2[np.argmax(Walu2[:, 1]):, :]

x = Walu2[:, 1]
t = Walu2[:, 0]
Fs = 500

L = len(x)

#t = x[:, 0]

Y = np.fft.fft(x)


f = Fs/L*np.arange(0, L/2)
P2 = abs(Y)/L*2
P1 = P2[0:L//2 + 1]


plot(f, P1, label='Alu2', linewidth=5)
reso = f[np.argmax(P1[10:]) + 10]
plt.axvline(x=reso, color='r', linestyle='--', label='Resonanz')
plt.xlim(0, 30)
plt.ylim(0, 0.26)
xlabel('Frequenz in Hz', fontsize=20)
ylabel('Amplitude in V', fontsize=20)
title('Alu2, Frequenzdomäne', fontsize=20)
legend(fontsize=13, loc='upper right')
grid()
plt.tight_layout()
savefig('alu2F')
show()

#berechne emodulus
m = 32.17/1000
sigma_m = 0.01/1000
L = 0.3
sigma_L = 0.01

b = 2/1000
sigma_b = 0.05/1000
a = 2.01/100
sigma_a = 0.05/1000

beta1 = pi*0.59686/(L - 0.03)#einspannung 3cm
sigma_beta1 = gauss("pi*0.59686/(L - 0.03)")

omega = reso*2*pi
sigma_omega = 0.01*omega

EAlu2 = omega**2/beta1**4*m/L*12/a/b**3
sigma_EAlu2 = gauss("omega**2/beta1**4*m/L*12/a/b**3")



def decay(x, A, beta, c):
    return A*exp(-beta*x) + c


s = x*25#in mm dann
plot(t, s, label='Alu2', linewidth=2)
peaks, _ = scipy.signal.find_peaks(s)
optimizedParameters1, s = opt.curve_fit(decay, t[peaks], s[peaks])
plot(t[peaks], decay(t[peaks], *optimizedParameters1), label="fit1")
xlabel('Zeit in s', fontsize=20)
ylabel('Position in mm', fontsize=20)
title('Alu2, Zeitdomäne', fontsize=20)
legend(fontsize=13, loc='upper right')
grid()
plt.tight_layout()
savefig('alu2T')
show()

betaAlu2 = optimizedParameters1[1]
sigma_betaAlu2 = diag(s)[1]


from importieren import WCu
WCu = WCu[np.argmax(WCu[:, 1]):, :]

x = WCu[:, 1]
t = WCu[:, 0]
Fs = 100

L = len(x)

#t = x[:, 0]

Y = np.fft.fft(x)


f = Fs/L*np.arange(0, L//2 + 1)
P2 = abs(Y)/L*2
P1 = P2[0:L//2 + 1]


plot(f, P1, label='Kupfer', linewidth=5)
reso = f[np.argmax(P1[10:]) + 10]
plt.axvline(x=reso, color='r', linestyle='--', label='Resonanz')
plt.xlim(0, 10)
plt.ylim(0, 0.3)
xlabel('Frequenz in Hz', fontsize=20)
ylabel('Amplitude in V', fontsize=20)
title('Kupfer, Frequenzdomäne', fontsize=20)
legend(fontsize=13, loc='upper left')
grid()
plt.tight_layout()
savefig('CuF')
show()

#berechne emodulus
m = 26.81/1000
sigma_m = 0.01/1000
L = 0.3
sigma_L = 0.01

b = 0.5/1000
sigma_b = 0.05/1000
a = 2.01/100
sigma_a = 0.05/1000

beta1 = pi*0.59686/(L - 0.03)#einspannung 3cm
sigma_beta1 = gauss("pi*0.59686/(L - 0.03)")

omega = reso*2*pi
sigma_omega = 0.01*omega

ECu = omega**2/beta1**4*m/L*12/a/b**3
sigma_ECu = gauss("omega**2/beta1**4*m/L*12/a/b**3")



def decay(x, A, beta, c):
    return A*exp(-beta*x) + c


s = x*25#in mm dann
plot(t, s, label='Kupfer', linewidth=2)
peaks, _ = scipy.signal.find_peaks(s)
optimizedParameters1, s = opt.curve_fit(decay, t[peaks], s[peaks])
plot(t[peaks], decay(t[peaks], *optimizedParameters1), label="fit1")
xlabel('Zeit in s', fontsize=20)
ylabel('Position in mm', fontsize=20)
title('Kupfer, Zeitdomäne', fontsize=20)
legend(fontsize=13, loc='upper right')
grid()
plt.tight_layout()
savefig('CuT')
show()

betaCu = optimizedParameters1[1]
sigma_betaCu = diag(s)[1]


from importieren import WFe1
WFe1 = WFe1[np.argmax(WFe1[:, 1]):, :]

x = WFe1[:, 1]
t = WFe1[:, 0]
Fs = 100

L = len(x)

#t = x[:, 0]

Y = np.fft.fft(x)


f = Fs/L*np.arange(0, L//2 + 1)
P2 = abs(Y)/L*2
P1 = P2[0:L//2 + 1]


plot(f, P1, label='Stahl1', linewidth=5)
reso = f[np.argmax(P1[10:]) + 10]
plt.axvline(x=reso, color='r', linestyle='--', label='Resonanz')
plt.xlim(0, 60)
plt.ylim(0, 0.4)
xlabel('Frequenz in Hz', fontsize=20)
ylabel('Amplitude in V', fontsize=20)
title('Stahl1, Frequenzdomäne', fontsize=20)
legend(fontsize=13, loc='upper right')
grid()
plt.tight_layout()
savefig('Fe1F')
show()

#berechne emodulus
m = 26.77/1000
sigma_m = 0.01/1000
L = 0.3
sigma_L = 0.01

b = 0.7/1000
sigma_b = 0.05/1000
a = 2.01/100
sigma_a = 0.05/1000

beta1 = pi*0.59686/(L - 0.03)#einspannung 3cm
sigma_beta1 = gauss("pi*0.59686/(L - 0.03)")

omega = reso*2*pi
sigma_omega = 0.01*omega

EFe1 = omega**2/beta1**4*m/L*12/a/b**3
sigma_EFe1 = gauss("omega**2/beta1**4*m/L*12/a/b**3")



def decay(x, A, beta, c):
    return A*exp(-beta*x) + c


s = x*25#in mm dann
plot(t, s, label='Stahl1', linewidth=2)
peaks, _ = scipy.signal.find_peaks(s)
optimizedParameters1, s = opt.curve_fit(decay, t[peaks], s[peaks])
plot(t[peaks], decay(t[peaks], *optimizedParameters1), label="fit1")
xlabel('Zeit in s', fontsize=20)
ylabel('Position in mm', fontsize=20)
title('Stahl1, Zeitdomäne', fontsize=20)
legend(fontsize=13, loc='upper right')
grid()
plt.tight_layout()
savefig('Fe1T')
show()

betaFe1 = optimizedParameters1[1]
sigma_betaFe1 = diag(s)[1]


#statische auswertung
def linear(x, a, b):
    return a*x + b


from importieren import WSAlu2

P = toArray(WSAlu2)
start = P[0]
step = 1#mm
omega = linspaceM(start, len(P), step)



plot(omega, P, label='Alu2', marker='*', markersize=10)
optimizedParameters1, s = opt.curve_fit(linear, omega, P)
plot(omega, linear(omega, *optimizedParameters1), label="fit1")
xlabel('Auslenkung in mm', fontsize=20)
ylabel('Kraft in N', fontsize=20)
title('Aluminium2', fontsize=20)
legend(fontsize=13, loc='lower right')
grid()
plt.tight_layout()
savefig('Salu2')
show()


L = 0.25
sigma_L = 0.01

b = 0.5/1000
sigma_b = 0.05/1000
a = 2.01/100
sigma_a = 0.05/1000

steig = optimizedParameters1[0]*1000
sigma_steig = diag(s)[0]*1000

ESAlu2 = L**3/48*steig*12/a/b**3
sigma_ESAlu2 = gauss("L**3/48*steig*12/a/b**3")



from importieren import WSAlu4

P = toArray(WSAlu4)
start = P[0]
step = 0.25#mm
omega = linspaceM(start, len(P), step)



plot(omega, P, label='Alu4', marker='*', markersize=10)
optimizedParameters1, s = opt.curve_fit(linear, omega, P)
plot(omega, linear(omega, *optimizedParameters1), label="fit1")
xlabel('Auslenkung in mm', fontsize=20)
ylabel('Kraft in N', fontsize=20)
title('Aluminium4', fontsize=20)
legend(fontsize=13, loc='lower right')
grid()
plt.tight_layout()
savefig('SAlu4')
show()


L = 0.25
sigma_L = 0.01

b = 2/1000
sigma_b = 0.05/1000
a = 2.01/100
sigma_a = 0.05/1000

steig = optimizedParameters1[0]*1000
sigma_steig = diag(s)[0]*1000

ESAlu4 = L**3/48*steig*12/a/b**3
sigma_ESAlu4 = gauss("L**3/48*steig*12/a/b**3")


from importieren import WSCu

P = toArray(WSCu)
start = P[0]
step = 1#mm
omega = linspaceM(start, len(P), step)



plot(omega, P, label='Cu', marker='*', markersize=10)
optimizedParameters1, s = opt.curve_fit(linear, omega, P)
plot(omega, linear(omega, *optimizedParameters1), label="fit1")
xlabel('Auslenkung in mm', fontsize=20)
ylabel('Kraft in N', fontsize=20)
title('Kupfer', fontsize=20)
legend(fontsize=13, loc='lower right')
grid()
plt.tight_layout()
savefig('SCu')
show()


L = 0.25
sigma_L = 0.01

b = 0.5/1000
sigma_b = 0.05/1000
a = 2.01/100
sigma_a = 0.05/1000

steig = optimizedParameters1[0]*1000
sigma_steig = diag(s)[0]*1000

ESCu = L**3/48*steig*12/a/b**3
sigma_ESCu = gauss("L**3/48*steig*12/a/b**3")


from importieren import WSFe1

P = toArray(WSFe1)
start = P[0]
step = 0.5#mm
omega = linspaceM(start, len(P), step)



plot(omega, P, label='Fe1', marker='*', markersize=10)
optimizedParameters1, s = opt.curve_fit(linear, omega, P)
plot(omega, linear(omega, *optimizedParameters1), label="fit1")
xlabel('Auslenkung in mm', fontsize=20)
ylabel('Kraft in N', fontsize=20)
title('Stahl1', fontsize=20)
legend(fontsize=13, loc='lower right')
grid()
plt.tight_layout()
savefig('SFe1')
show()


L = 0.25
sigma_L = 0.01

b = 0.7/1000
sigma_b = 0.05/1000
a = 2.01/100
sigma_a = 0.05/1000

steig = optimizedParameters1[0]*1000
sigma_steig = diag(s)[0]*1000

ESFe1 = L**3/48*steig*12/a/b**3
sigma_ESFe1 = gauss("L**3/48*steig*12/a/b**3")


from importieren import WSFe2

P = toArray(WSFe2)
start = P[0]
step = 0.5#mm
omega = linspaceM(start, len(P), step)



plot(omega, P, label='Fe2', marker='*', markersize=10)
optimizedParameters1, s = opt.curve_fit(linear, omega, P)
plot(omega, linear(omega, *optimizedParameters1), label="fit1")
xlabel('Auslenkung in mm', fontsize=20)
ylabel('Kraft in N', fontsize=20)
title('Stahl2', fontsize=20)
legend(fontsize=13, loc='lower right')
grid()
plt.tight_layout()
savefig('SFe2')
show()


L = 0.25
sigma_L = 0.01

b = 0.5/1000
sigma_b = 0.05/1000
a = 2.01/100
sigma_a = 0.05/1000

steig = optimizedParameters1[0]*1000
sigma_steig = diag(s)[0]*1000

ESFe2 = L**3/48*steig*12/a/b**3
sigma_ESFe2 = gauss("L**3/48*steig*12/a/b**3")












