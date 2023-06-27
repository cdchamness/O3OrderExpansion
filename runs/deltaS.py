import numpy as np
import psi_functions as psi
import flow_functions as flow


def do_flow(SFlow, lattice, dt, beta):
    lap_total = 0.0
    times = np.arange(0.0, 1.0, dt)

    for t in times:
        lattice, lap_next = flow.RK4(SFlow, lattice, t, dt, beta)
        lap_total += dt * lap_next

    return lattice, lap_total


def get_new_sample(SFlow, dt, lat_size, beta, seed=None):
    lat = flow.gen_random_lattice(lat_size, seed)
    return do_flow(SFlow, lat, dt, beta)


def main():
    import ProgressBar as pb

    SAMPLE_SIZE = 1_000
    STEPS = 30
    BETA = 0.92
    LAT_SIZE = 8

    dt = 1.0 / STEPS

    Order0 = [(0.125, "Psi_1")]
    Order1 = [(0.05, "Psi_2"), (-0.025, "Psi_1_1"), (0.00416667, "Psi_1_1f")]
    Flow_Action = [Order0, Order1]

    deltaS = []
    pb1 = pb.ProgressBar(SAMPLE_SIZE)
    for i in range(SAMPLE_SIZE):
        lat, lnDetJ = get_new_sample(Flow_Action, dt, LAT_SIZE, BETA)
        S = -(BETA / 2.0) * psi.Psi_1(lat)
        deltaS.append(S - lnDetJ)
        pb1.print(i)

    print(np.mean(deltaS))
    print(np.var(deltaS))


if __name__ == "__main__":
    main()
