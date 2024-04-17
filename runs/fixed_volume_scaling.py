import numpy as np
import psi_functions as psi
import flow_functions as flow
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


SAMPLE_SIZE = 10
DT = 0.1
FINAL_T = 1.0
SEED = 13333337

DataPath = "Data"


Order0 = [(0.125, "Psi_1")]
Order1 = [(0.05, "Psi_2"), (-0.025, "Psi_1_1"), (0.00416667, "Psi_1_1f")]
Order2 = [(-0.06204167, "Psi_1"), (0.018125, "Psi_3"), (-0.01375, "Psi_2_1"), (-0.004166667, "Psi_1_2_disc"),
          (0.002777778, "Psi_1_1_1_branch"), (0.003125, "Psi_1_1_1_chain"), (0.00341666667, "Psi_1_2f"),
          (-0.0029166667, "Psi_1_1f_1"), (0.00048611, "Psi_1_1f_1f")]
Flow_Action0 = [Order0]
Flow_Action1 = [Order0, Order1]
Flow_Action2 = [Order0, Order1, Order2]

targets = [(1.263, 24), (1.370, 36), (1.458, 80), (1.535, 120)]


def get_many_order0(target):
    target_beta, target_len = target
    print(f"Starting target_beta: {target_beta}, target_len: {target_len}")
    np.random.seed(SEED)
    deltaS = []
    for i in range(SAMPLE_SIZE):
        deltaS.append(get_deltaS(Flow_Action0, DT, target_len, target_beta, FINAL_T))
    print(f"Finished target_beta: {target_beta}, target_len: {target_len}")
    return target_beta, deltaS


def get_many_order1(target):
    target_beta, target_len = target
    print(f"Starting target_beta: {target_beta}, target_len: {target_len}")
    np.random.seed(SEED)
    deltaS = []
    for i in range(SAMPLE_SIZE):
        deltaS.append(get_deltaS(Flow_Action1, DT, target_len, target_beta, FINAL_T))
    print(f"Finished target_beta: {target_beta}, target_len: {target_len}")
    return target_beta, deltaS


def get_many_order2(target):
    target_beta, target_len = target
    print(f"Starting target_beta: {target_beta}, target_len: {target_len}")
    np.random.seed(SEED)
    deltaS = []
    for i in range(SAMPLE_SIZE):
        deltaS.append(get_deltaS(Flow_Action2, DT, target_len, target_beta, FINAL_T))
    print(f"Finished target_beta: {target_beta}, target_len: {target_len}")
    return target_beta, deltaS


def main():
    MAX_CURRENT_PROCESSES = 4

    print(f"SAMPLE_SIZE: {SAMPLE_SIZE}")
    print(f"DeltaT     : {DT}")
    print(f"SEED       : {SEED}")

    print(f"\nDoing Measurements for targets: {targets}")

    order0, beta0 = [], []
    print("\nDoing Order 0")
    with Pool(processes=MAX_CURRENT_PROCESSES) as pool:
        for beta, result in pool.imap(get_many_order0, targets):
            beta0.append(beta)
            order0.append(result)
    order0 = np.array(order0)

    print("\nDoing Order 1")
    order1, beta1 = [], []
    with Pool(processes=MAX_CURRENT_PROCESSES) as pool:
        for beta, result in pool.imap(get_many_order1, targets):
            beta1.append(beta)
            order1.append(result)
    order1 = np.array(order1)

    print("\nDoing Order 2")
    order2, beta2 = [], []
    with Pool(processes=MAX_CURRENT_PROCESSES) as pool:
        for beta, result in pool.imap(get_many_order2, targets):
            beta2.append(beta)
            order2.append(result)
    order2 = np.array(order2)

    print("\nAll Measurements Finished!")
    print("\nSaving Data...")
    np.save(f"{DataPath}/fixed_s{SAMPLE_SIZE}.npy", targets)
    np.save(f"{DataPath}/order0_s{SAMPLE_SIZE}.npy", order0)
    np.save(f"{DataPath}/order1_s{SAMPLE_SIZE}.npy", order1)
    np.save(f"{DataPath}/order2_s{SAMPLE_SIZE}.npy", order2)
    print(f"Data Saved at: {DataPath}/")


if __name__ == "__main__":
    main()
