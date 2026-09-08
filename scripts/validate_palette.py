"""Check a categorical palette for colour-vision-deficiency separation.

Converts sRGB to OKLab, simulates protanopia, deuteranopia and tritanopia with the
Brettel-Vienot-Mollon linear approximations, and reports the smallest perceptual
distance between any pair under each condition. The dataviz guidance treats DeltaE >= 8
(OKLab x100) as the target and 6-8 as a floor that needs a second encoding channel;
below 15 for normal vision is a hard failure.
"""
import itertools
import numpy as np

def srgb_to_lin(c):
    c = np.asarray(c, float) / 255
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)

def lin_to_oklab(rgb):
    M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                   [0.2119034982, 0.6806995451, 0.1073969566],
                   [0.0883024619, 0.2817188376, 0.6299787005]])
    M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                   [1.9779984951, -2.4285922050, 0.4505937099],
                   [0.0259040371, 0.7827717662, -0.8086757660]])
    lms = rgb @ M1.T
    return np.cbrt(np.clip(lms, 0, None)) @ M2.T

SIM = {
 "protanopia":   np.array([[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216], [-0.003882, -0.048116, 1.051998]]),
 "deuteranopia": np.array([[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413], [-0.011820, 0.042940, 0.968881]]),
 "tritanopia":   np.array([[1.255528, -0.076749, -0.178779], [-0.078411, 0.930809, 0.147602], [0.004733, 0.691367, 0.303900]]),
}

def report(hexes, names=None):
    names = names or hexes
    rgb = np.array([[int(h[i:i+2], 16) for i in (1, 3, 5)] for h in hexes])
    lin = srgb_to_lin(rgb)
    conds = {"normal": lin}
    for k, M in SIM.items():
        conds[k] = np.clip(lin @ M.T, 0, 1)
    for cond, L in conds.items():
        lab = lin_to_oklab(L) * 100
        worst, pair = 1e9, None
        for a, b in itertools.combinations(range(len(hexes)), 2):
            d = float(np.linalg.norm(lab[a] - lab[b]))
            if d < worst: worst, pair = d, (names[a], names[b])
        flag = "PASS" if worst >= (15 if cond == "normal" else 8) else ("FLOOR" if worst >= 6 else "FAIL")
        print(f"  {cond:13} min dE = {worst:6.2f}  ({pair[0]} vs {pair[1]})  {flag}")

if __name__ == "__main__":
    import sys
    hexes = sys.argv[1].split(",")
    names = sys.argv[2].split(",") if len(sys.argv) > 2 else None
    report(hexes, names)
