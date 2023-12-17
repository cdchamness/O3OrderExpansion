import numpy as np
import matplotlib.pyplot as plt


SAMPLE_SIZE = 3
LAT_SIZE = 36

def main():
    target_betas = np.load(f"Data/beta_s{SAMPLE_SIZE}_l{LAT_SIZE}.npy")
    order0 = np.load(f"Data/order0_s{SAMPLE_SIZE}_l{LAT_SIZE}.npy")
    order1 = np.load(f"Data/order1_s{SAMPLE_SIZE}_l{LAT_SIZE}.npy")
    order2 = np.load(f"Data/order2_s{SAMPLE_SIZE}_l{LAT_SIZE}.npy")
    
    
    order0_err = np.empty((len(target_betas), SAMPLE_SIZE))
    order1_err = np.empty((len(target_betas), SAMPLE_SIZE))
    order2_err = np.empty((len(target_betas), SAMPLE_SIZE))
    for i in range(SAMPLE_SIZE):
        order0_err[:, i] = np.std(np.concatenate((order0[:, :i], order0[:, i+1:]), axis=1))
        order1_err[:, i] = np.std(np.concatenate((order1[:, :i], order1[:, i+1:]), axis=1))
        order2_err[:, i] = np.std(np.concatenate((order2[:, :i], order2[:, i+1:]), axis=1))
    
    
    plt.errorbar(target_betas, np.std(order0, axis=1), yerr=np.sqrt(SAMPLE_SIZE - 1) * np.std(order0_err, axis=1), c="r", fmt="o")
    plt.errorbar(target_betas, np.std(order1, axis=1 ), yerr=np.sqrt(SAMPLE_SIZE - 1) * np.std(order0_err, axis=1), c="b", fmt="o")
    plt.errorbar(target_betas, np.std(order2, axis=1), yerr=np.sqrt(SAMPLE_SIZE - 1) * np.std(order2_err, axis=1), c="g", fmt="o")
    plt.legend(["Order0", "Order1", "Order2"])
    plt.ylim(bottom=0.0)
    plt.ylabel("std(ΔS)")
    plt.xlabel("Beta")
    plt.title(f"{LAT_SIZE}x{LAT_SIZE} lattice")
    #plt.savefig(f"../Plots/O3_dS_vs_Beta_test.png", dpi=500)
    plt.show()
    print("\n")


if __name__ == '__main__':
    main()