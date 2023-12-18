import numpy as np
import sys


T = np.array([
    [[0, 0, 0], [0, 0, -1], [0, 1, 0]],
    [[0, 0, 1], [0, 0, 0], [-1, 0, 0]],
    [[0, -1, 0], [1, 0, 0], [0, 0, 0]]])

def Psi_1(lattice):
    # <x|x+mu>
    total = 0
    shifts = [[1, 0], [-1, 0], [0, 1], [0, -1]]
    for shift in shifts:
        total += np.einsum("imj,imj", lattice, np.roll(lattice, shift, axis=[0, 1]))
    return total


def dPsi_1(lattice):
    # -2<x|Ta|x+mu>
    total = 0
    shifts = [[1, 0], [-1, 0], [0, 1], [0, -1]]
    for shift in shifts:
        total += np.einsum("imj,jkl,imk->iml", lattice, T, np.roll(lattice, shift, axis=[0, 1]))
    return -2.0 * total


def Psi_2(lattice):
    # <x|x+mu+nu>
    total = 0
    shifts = [[1, 0], [-1, 0], [0, 1], [0, -1]]
    for shift1 in shifts:
        for shift2 in shifts:
            shift = [sum(x) for x in zip(shift1, shift2)]
            total += np.einsum("imj,imj", lattice, np.roll(lattice, shift, axis=[0, 1]))
    return total


def dPsi_2(lattice):
    # -2.0<x|Ta|x+mu+nu>
    total = 0
    shifts = [[1, 0], [-1, 0], [0, 1], [0, -1]]
    for shift1 in shifts:
        for shift2 in shifts:
            shift = [sum(x) for x in zip(shift1, shift2)]
            total += np.einsum("imj,jkl,imk->iml", lattice, T, np.roll(lattice, shift, axis=[0, 1]))
    return -2.0 * total


def Psi_1_1(lattice):
    # <x|x+mu><x|x+nu>
    total = 0
    shifts = [[1, 0], [-1, 0], [0, 1], [0, -1]]
    for shift1 in shifts:
        for shift2 in shifts:
            total += np.einsum("imj,imj->im", lattice, np.roll(lattice, shift1, axis=[0, 1])) * np.einsum("imj,imj->im", lattice, np.roll(lattice, shift2, axis=[0, 1]))
    return np.sum(total)


def dPsi_1_1(lattice):
    # 2<x|Ta|x+mu><x+mu|x+mu+nu> - 2<x|Ta|x+mu><x|x+nu>
    total = 0
    shifts = [[1, 0], [-1, 0], [0, 1], [0, -1]]
    for shift1 in shifts:
        for shift2 in shifts:
            total_shift = [sum(x) for x in zip(shift1, shift2)]
            # <x|Ta|x+mu><x+mu|x+mu+nu>
            total += np.einsum("imj,jkl,imk->iml", lattice, T, np.roll(lattice, shift1, axis=[0, 1])) * np.einsum("imj,imj->im", np.roll(lattice, shift1, axis=[0, 1]), np.roll(lattice, total_shift, axis=[0, 1]))[:, :, None]
            # -1<x|Ta|x+mu><x|x+nu>
            total -= np.einsum("imj,jkl,imk->iml", lattice, T, np.roll(lattice, shift1, axis=[0, 1])) * np.einsum("imj,imj->im", lattice, np.roll(lattice, shift2, axis=[0, 1]))[:, :, None]
    return 2.0 * total


def Psi_1_1f(lattice):
    # <x|x+mu><x|x+mu>
    total = 0
    shifts = [[1, 0], [-1, 0], [0, 1], [0, -1]]
    for shift in shifts:
        total += np.einsum("imj,imj->im", lattice, np.roll(lattice, shift, axis=[0, 1])) ** 2
    return np.sum(total)


def dPsi_1_1f(lattice):
    # -4<x|Ta|x+mu><x|x+mu>
    total = 0
    shifts = [[1, 0], [-1, 0], [0, 1], [0, -1]]
    for shift in shifts:
        total += np.einsum("imj,jkl,imk->iml", lattice, T, np.roll(lattice, shift, axis=[0, 1])) * np.einsum("imj,imj->im", lattice, np.roll(lattice, shift, axis=[0, 1]))[:, :, None]
    return -4.0 * total


def expo(momentum):
    min_float = sys.float_info.min
    norm = np.sqrt(np.einsum('imj,imj->im', momentum, momentum))
    sin = np.sin(norm) / (norm + min_float)
    sin2 = 2 * (np.sin(norm / 2.0) / (norm + min_float)) ** 2
    A = np.einsum('imj,jkl->imkl', momentum, T)
    AA = np.einsum("imjk,imkl->imjl", A, A)
    return np.eye(3, 3) + A * sin[:, :, None, None] + AA * sin2[:, :, None, None]


if __name__ == "__main__":
    L = 5
    testLat = [[0, 0, 1] for _ in range(L * L)]
    testLat[12] = [1, 0, 0]
    testLat = np.reshape(np.array(testLat), (L, L, 3))
    print(testLat)
    print(Psi_2(testLat))
    print(np.sum(dPsi_1_1f(testLat), axis=2))
    print(expo(dPsi_1_1f(testLat)))
    newLat = np.einsum("imjk,imk->imj", expo(dPsi_1_1f(0.3 * testLat)), testLat)
    print(newLat)
    print(np.einsum("imj,imj->im", newLat, newLat))

    print(Psi_1(testLat))
    print(Psi_2(testLat))
    print(Psi_1_1(testLat))
    print(Psi_1_1f(testLat))

