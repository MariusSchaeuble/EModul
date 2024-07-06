import numpy as np

Walu1 = np.loadtxt("data/dyn_alu_0,5mm_8,25g.txt", skiprows=4)
Walu2 = np.loadtxt("data/dyn_alu_2mm_32,17g.txt", skiprows=4)
WCu = np.loadtxt("data/dyn_Cu_0,5mm_26,81g.txt", skiprows=4)
WFe1 = np.loadtxt("data/dyn_Fe_0,7mm_267,7g.txt", skiprows=4)
WFe2 = np.loadtxt("data/dyn_Fe_1mm_46,86g.txt", skiprows=4)

WSAlu2 = np.loadtxt("data/stat_Al2.txt", skiprows=1)
WSAlu3 = np.loadtxt("data/stat_Al3.txt", skiprows=1)
WSAlu4 = np.loadtxt("data/stat_Al4.txt", skiprows=1)
WSCu = np.loadtxt("data/stat_Cu1.txt", skiprows=1)
WSFe1 = np.loadtxt("data/stat_Fe1.txt", skiprows=1)
WSFe2 = np.loadtxt("data/stat_Fe2.txt", skiprows=1)

