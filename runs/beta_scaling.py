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
DT = 0.05
FINAL_T = 1.0
LAT_SIZE = 12
SEED = 13333337


Order0 = [(0.125, "Psi_1")]
Order1 = [(0.05, "Psi_2"), (-0.025, "Psi_1_1"), (0.00416667, "Psi_1_1f")]
Flow_Action0 = [Order0]
Flow_Action1 = [Order0, Order1]

target_betas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]


def get_many_order0(target_beta):
    print(f"Starting target_beta: {target_beta}")
    np.random.seed(SEED)
    deltaS = []
    for i in range(SAMPLE_SIZE):
        deltaS.append(get_deltaS(Flow_Action0, DT, LAT_SIZE, target_beta, FINAL_T))
    print(f"Finished target_beta: {target_beta}")
    return target_beta, deltaS
    

def get_many_order1(target_beta):
    print(f"Starting target_beta: {target_beta}")
    np.random.seed(SEED)
    deltaS = []
    for i in range(SAMPLE_SIZE):
        deltaS.append(get_deltaS(Flow_Action1, DT, LAT_SIZE, target_beta, FINAL_T))
    print(f"Finished target_beta: {target_beta}")
    return target_beta, deltaS


def main():
    import matplotlib.pyplot as plt
    MAX_CURRENT_PROCESSES = 15


    order0, beta0 = [], []
    with Pool(processes=MAX_CURRENT_PROCESSES) as pool:
        for beta, result in pool.imap(get_many_order0, target_betas):
            beta0.append(beta)
            order0.append(result)
    order0 = np.array(order0)

    order1, beta1 = [], []
    with Pool(processes=MAX_CURRENT_PROCESSES) as pool:
        for beta, result in pool.imap(get_many_order1, target_betas):
            beta1.append(beta)
            order1.append(result)
    order1 = np.array(order1)
    
    order0_err = np.empty((len(target_betas), SAMPLE_SIZE))
    order1_err = np.empty((len(target_betas), SAMPLE_SIZE))
    for i in range(SAMPLE_SIZE):
        order0_err[:, i] = np.std(np.concatenate((order0[:, :i], order0[:, i+1:]), axis=1))
        order1_err[:, i] = np.std(np.concatenate((order1[:, :i], order1[:, i+1:]), axis=1))
        
        
    plt.errorbar(beta0, np.std(order0, axis=1), yerr=np.sqrt(SAMPLE_SIZE - 1) * np.std(order0_err, axis=1), c="r", fmt="o")
    plt.errorbar(beta1, np.std(order1, axis=1), yerr=np.sqrt(SAMPLE_SIZE - 1) * np.std(order0_err, axis=1), c="b", fmt="o")
    plt.legend(["Order0", "Order1"])
    plt.ylim(bottom=0.0)
    plt.ylabel("std(ΔS)")
    plt.xlabel("Beta")
    plt.title(f"({LAT_SIZE}x{LAT_SIZE})")
    plt.savefig(f"Plots/O3_dS_vs_Beta.png", dpi=500)
    plt.show()
    print("\n")


if __name__ == "__main__":
    main()
