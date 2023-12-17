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


SAMPLE_SIZE = 500
LAT_SIZE = 12
BETA = 0.5
FINAL_T = 1.0
SEED = 13333337


Order0 = [(0.125, "Psi_1")]
Order1 = [(0.05, "Psi_2"), (-0.025, "Psi_1_1"), (0.00416667, "Psi_1_1f")]
Flow_Action0 = [Order0]
Flow_Action1 = [Order0, Order1]

target_dts = [0.5, 0.25, 0.2, 0.1, 0.05, 0.025]#, 0.0125]

def get_many_order0(target_dt):
    print(f"Starting target_dt: {target_dt}")
    np.random.seed(SEED)
    deltaS = []
    for i in range(SAMPLE_SIZE):
        deltaS.append(get_deltaS(Flow_Action0, target_dt, LAT_SIZE, BETA, FINAL_T))
    print(f"Finished target_dt: {target_dt}")
    return target_dt, deltaS
    
def get_many_order1(target_dt):
    print(f"Starting target_dt: {target_dt}")
    np.random.seed(SEED)
    deltaS = []
    for i in range(SAMPLE_SIZE):
        deltaS.append(get_deltaS(Flow_Action1, target_dt, LAT_SIZE, BETA, FINAL_T))
    print(f"Finished target_dt: {target_dt}")
    return target_dt, deltaS


def main():
    import matplotlib.pyplot as plt
    MAX_CURRENT_PROCESSES = 15


    order0, dt0 = [], []
    with Pool(processes=MAX_CURRENT_PROCESSES) as pool:
        for l, result in pool.imap(get_many_order0, target_dts):
            dt0.append(l)
            order0.append(result)
    order0 = np.array(order0)

    order1, dt1 = [], []
    with Pool(processes=MAX_CURRENT_PROCESSES) as pool:
        for l, result in pool.imap(get_many_order1, target_dts):
            dt1.append(l)
            order1.append(result)
    order1 = np.array(order1)
    
    order0_err = np.empty((len(target_dts), SAMPLE_SIZE))
    order1_err = np.empty((len(target_dts), SAMPLE_SIZE))
    for i in range(SAMPLE_SIZE):
        order0_err[:, i] = np.std(np.concatenate((order0[:, :i], order0[:, i+1:]), axis=1))
        order1_err[:, i] = np.std(np.concatenate((order1[:, :i], order1[:, i+1:]), axis=1))
        
        
    plt.errorbar(dt0, np.std(order0, axis=1), yerr=np.sqrt(SAMPLE_SIZE - 1) * np.std(order0_err, axis=1), c="r", fmt="o", markersize=4)
    plt.errorbar(dt1, np.std(order1, axis=1), yerr=np.sqrt(SAMPLE_SIZE - 1) * np.std(order0_err, axis=1), c="b", fmt="o", markersize=4)

    plt.legend(["Order0", "Order1"])
    plt.ylim(bottom=0.0)
    plt.ylabel("std(ΔS)")
    plt.xlabel("dt (step size)")
    plt.xscale("log")
    plt.title(f"beta={BETA}, t={FINAL_T}")
    plt.savefig(f"Plots/O3_dS_vs_StepSize.png", dpi=500, format="png")
    plt.show()
    print("\n")


if __name__ == "__main__":
    main()
