import numpy as np
import psi_functions as psi
import random


def RK4(SFlow, lattice, time, dt, beta):
    k1 = getFlow(SFlow, lattice, time, beta)
    x1 = updateLat(lattice, k1, 0.5 * dt)
    k2 = getFlow(SFlow, x1, time + 0.5 * dt, beta)
    x2 = updateLat(lattice, k2, 0.5 * dt)
    k3 = getFlow(SFlow, x2, time + 0.5 * dt, beta)
    x3 = updateLat(lattice, k3, dt)
    k4 = getFlow(SFlow, x3, time + dt, beta)

    F = (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
    out = updateLat(lattice, F, dt)

    l1 = getLap(SFlow, lattice, time, beta)
    l2 = getLap(SFlow, x1, time + 0.5 * dt, beta)
    l3 = getLap(SFlow, x2, time + 0.5 * dt, beta)
    l4 = getLap(SFlow, x3, time + dt, beta)

    L = (l1 + 2 * l2 + 2 * l3 + l4) / 6.0

    return out, L


def getFlow(SFlow, lattice, time, beta):
    F = 0
    orderCoef = beta
    for order in SFlow:
        for term in order:
            F += orderCoef * getFlowValue(term, lattice)
        orderCoef *= beta * time

    return F


def getFlowValue(term, lattice):
    scalar, func_name = term
    match func_name:
        case "Psi_1":
            return scalar * psi.dPsi_1(lattice)
        case "Psi_2":
            return scalar * psi.dPsi_2(lattice)
        case "Psi_1_1":
            return scalar * psi.dPsi_1_1(lattice)
        case "Psi_1_1f":
            return scalar * psi.dPsi_1_1f(lattice)


def getLap(SFlow, lattice, time, beta):
    L = 0
    orderCoef = beta
    for order in SFlow:
        for term in order:
            L += orderCoef * getLapValue(term, lattice)
        orderCoef *= beta * time

    return L


def getLapValue(term, lattice):
    scalar, func_name = term
    match func_name:
        case "Psi_1":
            return scalar * 4 * psi.Psi_1(lattice)
        case "Psi_2":
            return scalar * 4 * psi.Psi_2(lattice)
        case "Psi_1_1":
            return scalar * (10 * psi.Psi_1_1(lattice) - 2 * psi.Psi_2(lattice) + 2 * psi.Psi_1_1f(lattice))
        case "Psi_1_1f":
            return scalar * 12 * psi.Psi_1_1f(lattice)


def updateLat(lattice, momentum, dt):
    return np.einsum("imjk,imk->imj", psi.expo(dt * momentum), lattice)


def gen_random_lattice(size, seed=None, verbose=False):
    if seed is None:
        seed = random.randint(0, 1_000_000)  # this uses a different rng than numpy
    if verbose:
        print(f"Using seed: {seed}")
    np.random.seed(seed)

    lattice = np.random.normal(0, 1, (size, size, 3))
    norm = np.sqrt(np.einsum("imj,imj->im", lattice, lattice))
    lattice /= norm[:, :, None]
    return lattice


def main():
    import ProgressBar as pb

    LatSize = 5
    beta = 2.0

    Order0 = [(0.125, "Psi_1")]
    Order1 = [(0.05, "Psi_2"), (-0.025, "Psi_1_1"), (0.00416667, "Psi_1_1f")]
    Flow_Action = [Order0, Order1]

    results = []

    steps = 1
    dt = 1.0 / steps
    times = np.arange(0.0, 1.0, dt)

    lattice = gen_random_lattice(LatSize, seed=1111)
    lap_total = 0.0
    pb1 = pb.ProgressBar(len(times), prefix=f"Doing Flow {steps}")
    for i, t in enumerate(times):
        lattice, lap_next = RK4(Flow_Action, lattice, t, dt, beta)
        lap_total += dt * lap_next
        pb1.print(i)
    print(lattice)
    results.append(lattice)
    print(np.einsum("imj,imj->im", lattice, lattice))
    print(lap_total)

    steps = 10
    dt = 1.0 / steps
    times = np.arange(0.0, 1.0, dt)

    lattice = gen_random_lattice(LatSize, seed=1111)
    lap_total = 0.0
    pb1 = pb.ProgressBar(len(times), prefix=f"Doing Flow {steps}")
    for i, t in enumerate(times):
        lattice, lap_next = RK4(Flow_Action, lattice, t, dt, beta)
        lap_total += dt * lap_next
        pb1.print(i)
    print(lattice)
    results.append(lattice)
    print(np.einsum("imj,imj->im", lattice, lattice))
    print(lap_total)

    steps = 100
    dt = 1.0 / steps
    times = np.arange(0.0, 1.0, dt)

    lattice = gen_random_lattice(LatSize, seed=1111)
    lap_total = 0.0
    pb1 = pb.ProgressBar(len(times), prefix=f"Doing Flow {steps}")
    for i, t in enumerate(times):
        lattice, lap_next = RK4(Flow_Action, lattice, t, dt, beta)
        lap_total += dt * lap_next
        pb1.print(i)
    print(lattice)
    results.append(lattice)
    print(np.einsum("imj,imj->im", lattice, lattice))
    print(lap_total)

    steps = 1_000
    dt = 1.0 / steps
    times = np.arange(0.0, 1.0, dt)

    lattice = gen_random_lattice(LatSize, seed=1111)
    lap_total = 0.0
    pb1 = pb.ProgressBar(len(times), prefix=f"Doing Flow {steps}")
    for i, t in enumerate(times):
        lattice, lap_next = RK4(Flow_Action, lattice, t, dt, beta)
        lap_total += dt * lap_next
        pb1.print(i)
    print(lattice)
    results.append(lattice)
    print(np.einsum("imj,imj->im", lattice, lattice))
    print(lap_total)

    steps = 10_000
    dt = 1.0 / steps
    times = np.arange(0.0, 1.0, dt)

    lattice = gen_random_lattice(LatSize, seed=1111)
    lap_total = 0.0
    pb1 = pb.ProgressBar(len(times), prefix=f"Doing Flow {steps}")
    for i, t in enumerate(times):
        lattice, lap_next = RK4(Flow_Action, lattice, t, dt, beta)
        lap_total += dt * lap_next
        pb1.print(i)
    print(lattice)
    results.append(lattice)
    print(np.einsum("imj,imj->im", lattice, lattice))
    print(lap_total)

    steps = 100_000
    dt = 1.0 / steps
    times = np.arange(0.0, 1.0, dt)

    lattice = gen_random_lattice(LatSize, seed=1111)
    lap_total = 0.0
    pb1 = pb.ProgressBar(len(times), prefix=f"Doing Flow {steps}")
    for i, t in enumerate(times):
        lattice, lap_next = RK4(Flow_Action, lattice, t, dt, beta)
        lap_total += dt * lap_next
        pb1.print(i)
    print(lattice)
    results.append(lattice)
    print(np.einsum("imj,imj->im", lattice, lattice))
    print(lap_total)

    ts = [1, 10, 100, 1_000, 10_000, 100_000]
    for result, t in zip(results, ts):
        print(t, result - results[-1])


if __name__ == "__main__":
    main()