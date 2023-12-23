import numpy as np
import matplotlib.pyplot as plt

SAMPLE_SIZE = 400
BETA = 0.5

plt.rcParams['text.usetex'] = True


def main():
    # Load flow results
    target_lens = np.load(f"Data/length_s{SAMPLE_SIZE}_b{BETA}.npy")
    order0 = np.load(f"Data/order0_s{SAMPLE_SIZE}_b{BETA}.npy")
    order1 = np.load(f"Data/order1_s{SAMPLE_SIZE}_b{BETA}.npy")
    order2 = np.load(f"Data/order2_s{SAMPLE_SIZE}_b{BETA}.npy")

    # Do Jackknife estimates of the variances
    order0_err = np.empty((len(target_lens), SAMPLE_SIZE))
    order1_err = np.empty((len(target_lens), SAMPLE_SIZE))
    order2_err = np.empty((len(target_lens), SAMPLE_SIZE))
    for i in range(SAMPLE_SIZE):
        order0_err[:, i] = np.std(np.concatenate((order0[:, :i], order0[:, i + 1:]), axis=1))
        order1_err[:, i] = np.std(np.concatenate((order1[:, :i], order1[:, i + 1:]), axis=1))
        order2_err[:, i] = np.std(np.concatenate((order2[:, :i], order2[:, i + 1:]), axis=1))

    # Do fits
    coef0 = np.polyfit(target_lens, np.std(order0, axis=1), deg=1)
    print(f"order0: {coef0}")
    y0 = np.poly1d(coef0)
    coef1 = np.polyfit(target_lens, np.std(order1, axis=1), deg=1)
    y1 = np.poly1d(coef1)
    print(f"order1: {coef1}")
    coef2 = np.polyfit(target_lens, np.std(order2, axis=1), deg=1)
    y2 = np.poly1d(coef2)
    print(f"order2: {coef2}")

    # Make Plot
    plt.errorbar(target_lens, np.std(order0, axis=1), yerr=np.sqrt(SAMPLE_SIZE - 1) * np.std(order0_err, axis=1), c="r", fmt='x', elinewidth=1)
    plt.errorbar(target_lens, np.std(order1, axis=1), yerr=np.sqrt(SAMPLE_SIZE - 1) * np.std(order0_err, axis=1), c="b", fmt="x", elinewidth=1)
    plt.errorbar(target_lens, np.std(order2, axis=1), yerr=np.sqrt(SAMPLE_SIZE - 1) * np.std(order2_err, axis=1), c="g", fmt="x", elinewidth=1)
    plt.legend(["Order 0", "Order 1", "Order 2"])

    # # Fit Results
    plt.plot(target_lens, y0(target_lens), c='r')
    plt.plot(target_lens, y1(target_lens), c='b')
    plt.plot(target_lens, y2(target_lens), c='g')

    plt.ylim(bottom=0.0)
    plt.ylabel(r'std$(\Delta S)$', fontsize=16)
    plt.xlabel(r'$L / a$', fontsize=16)
    plt.savefig("Plots/O3_dS_vs_length_test.pdf", dpi=500, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()
