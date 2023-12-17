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
BETA = 0.5
DT = 0.25
FINAL_T = 1.0
SEED = 13333337


Order0 = [(0.125, "Psi_1")]
Order1 = [(0.05, "Psi_2"), (-0.025, "Psi_1_1"), (0.00416667, "Psi_1_1f")]
Order2 = [(-0.06204167, "Psi_1"), (0.018125, "Psi_3"), (-0.01375, "Psi_2_1"), (-0.004166667, "Psi_1_2_disc"), \
    (0.002777778, "Psi_1_1_1_branch"), (0.003125, "Psi_1_1_1_chain"), (0.00341666667, "Psi_1_2f"), \
    (-0.0029166667, "Psi_1_1f_1"), (0.00048611, "Psi_1_1f_1f")]
Flow_Action0 = [Order0]
Flow_Action1 = [Order0, Order1]
Flow_Action2 = [Order0, Order1, Order2]

target_lens = [8, 12, 16, 20, 24, 28, 32]#, 36 , 40, 48]

def get_many_order0(target_len):
    print(f"Starting target_len: {target_len}")
    np.random.seed(SEED)
    deltaS = []
    for i in range(SAMPLE_SIZE):
        deltaS.append(get_deltaS(Flow_Action0, DT, target_len, BETA, FINAL_T))
    print(f"Finished target_len: {target_len}")
    return target_len, deltaS
    
def get_many_order1(target_len):
    print(f"Starting target_len: {target_len}")
    np.random.seed(SEED)
    deltaS = []
    for i in range(SAMPLE_SIZE):
        deltaS.append(get_deltaS(Flow_Action1, DT, target_len, BETA, FINAL_T))
    print(f"Finished target_len: {target_len}")
    return target_len, deltaS
    
def get_many_order2(target_len):
    print(f"Starting target_len: {target_len}")
    np.random.seed(SEED)
    deltaS = []
    for i in range(SAMPLE_SIZE):
        deltaS.append(get_deltaS(Flow_Action2, DT, target_len, BETA, FINAL_T))
    print(f"Finished target_len: {target_len}")
    return target_len, deltaS


def main():
    import matplotlib.pyplot as plt
    MAX_CURRENT_PROCESSES = 15


    order0, len0 = [], []
    with Pool(processes=MAX_CURRENT_PROCESSES) as pool:
        for l, result in pool.imap(get_many_order0, target_lens):
            len0.append(l)
            order0.append(result)
    order0 = np.array(order0)

    order1, len1 = [], []
    with Pool(processes=MAX_CURRENT_PROCESSES) as pool:
        for l, result in pool.imap(get_many_order1, target_lens):
            len1.append(l)
            order1.append(result)
    order1 = np.array(order1)
    
    order2, len2 = [], []
    with Pool(processes=MAX_CURRENT_PROCESSES) as pool:
        for l, result in pool.imap(get_many_order2, target_lens):
            len2.append(l)
            order2.append(result)
    order2 = np.array(order2)

    order0_err = np.empty((len(target_lens), SAMPLE_SIZE))
    order1_err = np.empty((len(target_lens), SAMPLE_SIZE))
    order2_err = np.empty((len(target_lens), SAMPLE_SIZE))
    print("order0", order0)
    print("order1", order1)
    print("order2", order2)
    for i in range(SAMPLE_SIZE):
        order0_err[:, i] = np.std(np.concatenate((order0[:, :i], order0[:, i+1:]), axis=1))
        order1_err[:, i] = np.std(np.concatenate((order1[:, :i], order1[:, i+1:]), axis=1))
        order2_err[:, i] = np.std(np.concatenate((order2[:, :i], order2[:, i+1:]), axis=1))
        
        
    plt.errorbar(len0, np.std(order0, axis=1), yerr=np.sqrt(SAMPLE_SIZE - 1) * np.std(order0_err, axis=1), c="r", fmt="o", markersize=4)
    plt.errorbar(len1, np.std(order1, axis=1), yerr=np.sqrt(SAMPLE_SIZE - 1) * np.std(order1_err, axis=1), c="b", fmt="o", markersize=4)
    plt.errorbar(len2, np.std(order2, axis=1), yerr=np.sqrt(SAMPLE_SIZE - 1) * np.std(order2_err, axis=1), c="g", fmt="o", markersize=4)

    plt.legend(["Order0", "Order1", "Order2"])
    plt.ylim(bottom=0.0)
    plt.ylabel("std(ΔS)")
    plt.xlabel("Lattice Size (Length)")
    plt.title(f"beta={BETA}, t={FINAL_T}")
    plt.savefig(f"Plots/O3_dS_vs_Size_2.png", dpi=500, format="png")
    plt.show()
    print("\n")


if __name__ == "__main__":
    main()
