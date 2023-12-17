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


SAMPLE_SIZE = 300
DT = 0.1
BETA = 0.5
LAT_SIZE = 32
SEED = 13333337


Order0 = [(0.125, "Psi_1")]
Order1 = [(0.05, "Psi_2"), (-0.025, "Psi_1_1"), (0.00416667, "Psi_1_1f")]
Order2 = [(-0.06204167, "Psi_1"), (0.018125, "Psi_3"), (-0.01375, "Psi_2_1"), (-0.004166667, "Psi_1_2_disc"), \
    (0.002777778, "Psi_1_1_1_branch"), (0.003125, "Psi_1_1_1_chain"), (0.00341666667, "Psi_1_2f"), \
    (-0.0029166667, "Psi_1_1f_1"), (0.00048611, "Psi_1_1f_1f")]
Flow_Action0 = [Order0]
Flow_Action1 = [Order0, Order1]
Flow_Action2 = [Order0, Order1, Order2]

flow_times = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

def get_many_order0(target_time):
    print(f"Starting target_time: {target_time}")
    np.random.seed(SEED)
    deltaS = []
    for i in range(SAMPLE_SIZE):
        deltaS.append(get_deltaS(Flow_Action0, DT, LAT_SIZE, BETA, target_time))
    print(f"Finished target_time: {target_time}")
    return target_time, deltaS
    

def get_many_order1(target_time):
    print(f"Starting target_time: {target_time}")
    np.random.seed(SEED)
    deltaS = []
    for i in range(SAMPLE_SIZE):
        deltaS.append(get_deltaS(Flow_Action1, DT, LAT_SIZE, BETA, target_time))
    print(f"Finished target_time: {target_time}")
    return target_time, deltaS
    
    
def get_many_order2(target_time):
    print(f"Starting target_time: {target_time}")
    np.random.seed(SEED)
    deltaS = []
    for i in range(SAMPLE_SIZE):
        deltaS.append(get_deltaS(Flow_Action2, DT, LAT_SIZE, BETA, target_time))
    print(f"Finished target_time: {target_time}")
    return target_time, deltaS


def main():
    import matplotlib.pyplot as plt
    MAX_CURRENT_PROCESSES = 15


    order0, beta0 = [], []
    with Pool(processes=MAX_CURRENT_PROCESSES) as pool:
        for beta, result in pool.imap(get_many_order0, flow_times):
            beta0.append(beta)
            order0.append(result)
    order0 = np.array(order0)

    order1, beta1 = [], []
    with Pool(processes=MAX_CURRENT_PROCESSES) as pool:
        for beta, result in pool.imap(get_many_order1, flow_times):
            beta1.append(beta)
            order1.append(result)
    order1 = np.array(order1)
    
    order2, beta2 = [], []
    with Pool(processes=MAX_CURRENT_PROCESSES) as pool:
        for beta, result in pool.imap(get_many_order2, flow_times):
            beta2.append(beta)
            order2.append(result)
    order2 = np.array(order2)
    
    order0_err = np.empty((len(flow_times), SAMPLE_SIZE))
    order1_err = np.empty((len(flow_times), SAMPLE_SIZE))
    order2_err = np.empty((len(flow_times), SAMPLE_SIZE))
    for i in range(SAMPLE_SIZE):
        order0_err[:, i] = np.std(np.concatenate((order0[:, :i], order0[:, i+1:]), axis=1))
        order1_err[:, i] = np.std(np.concatenate((order1[:, :i], order1[:, i+1:]), axis=1))
        order2_err[:, i] = np.std(np.concatenate((order2[:, :i], order2[:, i+1:]), axis=1))
        
        
    plt.errorbar(beta0, np.std(order0, axis=1), yerr=np.sqrt(SAMPLE_SIZE - 1) * np.std(order0_err, axis=1), c="r", fmt="o")
    plt.errorbar(beta1, np.std(order1, axis=1), yerr=np.sqrt(SAMPLE_SIZE - 1) * np.std(order0_err, axis=1), c="b", fmt="o")
    plt.errorbar(beta2, np.std(order2, axis=1), yerr=np.sqrt(SAMPLE_SIZE - 1) * np.std(order2_err, axis=1), c="g", fmt="o")
    plt.legend(["Order0", "Order1", "Order2"])
    plt.ylim(bottom=0.0)
    plt.ylabel("std(ΔS)")
    plt.xlabel("flow time")
    plt.title(f"{LAT_SIZE}x{LAT_SIZE} lattice, beta={BETA}")
    plt.savefig(f"Plots/O3_dS_vs_final_t.png", dpi=500)


if __name__ == "__main__":
    main()
