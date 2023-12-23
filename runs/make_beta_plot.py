import numpy as np
import matplotlib.pyplot as plt


SAMPLE_SIZE = 400
LAT_SIZE = 36

plt.rcParams['text.usetex'] = True


def main():
    # Load flow results
    target_betas = np.load(f"Data/beta_s{SAMPLE_SIZE}_l{LAT_SIZE}.npy")
    order0 = np.load(f"Data/order0_s{SAMPLE_SIZE}_l{LAT_SIZE}.npy")
    order1 = np.load(f"Data/order1_s{SAMPLE_SIZE}_l{LAT_SIZE}.npy")
    order2 = np.load(f"Data/order2_s{SAMPLE_SIZE}_l{LAT_SIZE}.npy")

    # Do Jackknife estimates of variances
    order0_err = np.empty((len(target_betas), SAMPLE_SIZE))
    order1_err = np.empty((len(target_betas), SAMPLE_SIZE))
    order2_err = np.empty((len(target_betas), SAMPLE_SIZE))
    for i in range(SAMPLE_SIZE):
        order0_err[:, i] = np.std(np.concatenate((order0[:, :i], order0[:, i + 1:]), axis=1))
        order1_err[:, i] = np.std(np.concatenate((order1[:, :i], order1[:, i + 1:]), axis=1))
        order2_err[:, i] = np.std(np.concatenate((order2[:, :i], order2[:, i + 1:]), axis=1))

    # Do fits
    def n_deg_poly(x, a, n):
        return a * (x ** n)

    print(n_deg_poly(target_betas, 10.8, 2))
    print(np.std(order0, axis=1))

    '''
    # scipy version where I can define the function form exactly
    from scipy.optimize import curve_fit
    coef0 = curve_fit(lambda x, a: n_deg_poly(x, a, 2), target_betas, np.std(order0, axis=1))
    print(f"order0: {coef0}")
    y0 = n_deg_poly(target_betas, coef0[0], 2)
    coef1 = curve_fit(lambda x, a: n_deg_poly(x, a, 3), target_betas, np.std(order0, axis=1))
    print(f"order0: {coef1}")
    y1 = n_deg_poly(target_betas, coef1[0], 3)
    coef2 = curve_fit(lambda x, a: n_deg_poly(x, a, 4), target_betas, np.std(order0, axis=1))
    print(f"order0: {coef2}")
    y2 = n_deg_poly(target_betas, coef2[0], 4)

    '''
    # np.polyfit version
    coef0 = np.polyfit(target_betas, np.std(order0, axis=1), deg=2)
    y0 = np.poly1d(coef0)
    print(f"order0: {coef0}")
    coef1 = np.polyfit(target_betas, np.std(order1, axis=1), deg=3)
    y1 = np.poly1d(coef1)
    print(f"order1: {coef1}")
    coef2 = np.polyfit(target_betas, np.std(order2, axis=1), deg=4)
    y2 = np.poly1d(coef2)
    print(f"order2: {coef2}")
    # '''

    # Make Plots
    # # Data Points
    plt.errorbar(target_betas, np.std(order0, axis=1), yerr=np.sqrt(SAMPLE_SIZE - 1) * np.std(order0_err, axis=1), c="r", fmt="x", elinewidth=1)
    plt.errorbar(target_betas, np.std(order1, axis=1), yerr=np.sqrt(SAMPLE_SIZE - 1) * np.std(order0_err, axis=1), c="b", fmt="x", elinewidth=1)
    plt.errorbar(target_betas, np.std(order2, axis=1), yerr=np.sqrt(SAMPLE_SIZE - 1) * np.std(order2_err, axis=1), c="g", fmt="x", elinewidth=1)

    plt.legend(["Order 0", "Order 1", "Order 2"])

    # # Fit Results
    '''
    # These are the one made with scipy
    plt.plot(target_betas, y0, c='r')
    plt.plot(target_betas, y1, c='b')
    plt.plot(target_betas, y2, c='g')
    '''
    # These are the ones made with np.polyfit which must do all exponents
    plt.plot(target_betas, y0(target_betas), c='r')
    plt.plot(target_betas, y1(target_betas), c='b')
    plt.plot(target_betas, y2(target_betas), c='g')
    # '''

    plt.ylim(bottom=0.0)
    plt.ylabel(r'std$(\Delta S)$', fontsize=16)
    plt.xlabel(r'$\beta$', fontsize=16)
    plt.savefig("Plots/O3_dS_vs_Beta_test.pdf", dpi=500, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()
