import numpy as np
import psi_functions as psi
import flow_functions as flow
import sys
import random
from multiprocessing import Pool


def do_flow(SFlow, lattice, dt, beta, final_t):
    lap_total = 0.0
    times = np.arange(0.0, final_t, dt)

    for t in times:
        lattice, lap_next = flow.RK4(SFlow, lattice, t, dt, beta)
        lap_total += dt * lap_next

    return lattice, lap_total


def get_new_sample(SFlow, dt, lat_size, beta, final_t):
    lat = flow.gen_random_lattice(lat_size)
    return do_flow(SFlow, lat, dt, beta, final_t)


def get_deltaS(Flow_Action, dt, lat_size, beta, final_t):
    lat, lnDetJ = get_new_sample(Flow_Action, dt, lat_size, beta, final_t)
    S = -(beta / 2.0) * psi.Psi_1(lat)
    return S + lnDetJ


SAMPLE_SIZE = 100
DT = 0.01
BETA = float(sys.argv[1])
LAT_SIZE = 12
SEED = 222222


Order0 = [(0.125, "Psi_1")]
Order1 = [(0.05, "Psi_2"), (-0.025, "Psi_1_1"), (0.00416667, "Psi_1_1f")]
Flow_Action0 = [Order0]
Flow_Action1 = [Order0, Order1]

target_times = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def get_many_order0(target_time):
    print(f"Starting target_time: {target_time}")
    np.random.seed(SEED)
    deltaS = []
    for i in range(SAMPLE_SIZE):
        deltaS.append(get_deltaS(Flow_Action0, DT, LAT_SIZE, BETA, target_time))
    print(f"Finished target_time: {target_time}")
    return target_time, np.std(deltaS)


def get_many_order1(target_time):
    print(f"Starting target_time: {target_time}")
    np.random.seed(SEED)
    deltaS = []
    for i in range(SAMPLE_SIZE):
        deltaS.append(get_deltaS(Flow_Action1, DT, LAT_SIZE, BETA, target_time))
    print(f"Finished target_time: {target_time}")
    return target_time, np.std(deltaS)


def main():
    import matplotlib.pyplot as plt

    print(f"\nStarting Run for BETA={BETA}")

    order0 = []
    with Pool(processes=7) as pool:
        for result in pool.imap(get_many_order0, target_times):
            order0.append(result)
    order0 = np.array(order0)

    order1 = []
    with Pool(processes=7) as pool:
        for result in pool.imap(get_many_order1, target_times):
            order1.append(result)
    order1 = np.array(order1)

    plt.scatter(order0[:, 0], order0[:, 1], c="r")
    plt.scatter(order1[:, 0], order1[:, 1], c="b")
    plt.legend(["Order0", "Order1"])
    plt.ylim(bottom=0.0)
    plt.ylabel("std(ΔS)")
    plt.xlabel("flowtime (t)")
    plt.title(f"({LAT_SIZE}x{LAT_SIZE}), beta = {BETA}")
    plt.savefig(f"Plots/O3_beta{BETA}.png", dpi=500)
    print("\n")


if __name__ == "__main__":
    main()
