import numpy as np
import sys
from O3_symbolic import InnerProduct, Term
import symbolic_functions as sf



def makeTermFromString(TermString):
    IPList = []
    IP_strings = TermString.split("><")
    for i, currentIP in enumerate(IP_strings):
        
        # the scalar should be 1.0 for all IP's, except the first
        cScalar = 1.0
        
        if i == 0:
            #getting the non-zero scalar
            cScalar, currentIP = currentIP.split("<")
        if i ==len(IP_strings) - 1:
            #last IP will have a trailing < we need to remove
            currentIP = currentIP[:-1]
        
        IP_split = currentIP.split("|")
        if len(IP_split) == 2:
            bra_str = IP_split[0]
            ins_str = ""
            ket_str = IP_split[1]
        elif len(IP_split) == 3:
            bra_str = IP_split[0]
            ins_str = IP_split[1]
            ket_str = IP_split[2]
        else:
            print(TermString, "is not a valid Term string")
            return None 

        cBra = getShiftsFromStr(bra_str)
        cKet = getShiftsFromStr(ket_str)
        IPList.append(InnerProduct(bra=cBra, ket=cKet, index_type="y", inside=ins_str, scalar=cScalar))
    return Term(IPList=IPList)


def getShiftsFromStr(bra_ket_str):
    shift_str = bra_ket_str.split(",")[1:] # this should be the index_type variable, which we will discard
    return [int(i) if i != "__" else 0 for i in shift_str]

def expo(momentum):
    min_float = sys.float_info.min
    Ts = np.array([[[0,0,0],[0,0,-1],[0,1,0]],[[0,0,1],[0,0,0],[-1,0,0]],[[0,-1,0],[1,0,0],[0,0,0]]])
    norm = np.sqrt(np.einsum('ij,ij->i', momentum, momentum))
    sin = np.sin(norm)/(norm + min_float)
    sin2 = 2 * (np.sin(norm/2.0)/(norm+min_float)) ** 2 
    A = np.einsum('ij,jkl->ikl', momentum, Ts)
    AA = np.einsum("ijk,ikl->ijl", A, A)
    return np.eye(3,3) + A * sin[:, None, None] + AA * sin2[:, None, None]

def updateLat(lattice, momentum, dt):
    return np.einsum('ijk,ik->ij', expo(dt * momentum), lattice)



def getTermValue(term, lattice):
    return term.getScalar() * term.getValue(lattice=lattice)


def getFlowValues(SFlow, lattice, time, beta):
    F = 0
    orderCoef = beta
    for OrderFlow in SFlow:
        for term in OrderFlow:
            F += orderCoef * getTermValue(term, lattice)
        orderCoef *= beta*time
    
    return F 

def LeapFrog(SFlow, lattice, mom, time, dt, beta):
    mom += 0.5*dt*getFlowValues(SFlow, lattice, time, beta)
    lattice = updateLat(lattice, mom, dt)
    mom += 0.5*dt*getFlowValues(SFlow, lattice, time, beta)
    return lattice, mom

def RK4(SFlow, lattice, time, dt, beta):
    k1 = getFlowValues(SFlow, lattice, time, beta)
    x1 = updateLat(lattice, k1, 0.5 * dt)
    k2 = getFlowValues(SFlow, x1, time + 0.5 * dt, beta)
    x2 = updateLat(lattice, k2, 0.5 * dt)
    k3 = getFlowValues(SFlow, x2, time + 0.5 * dt, beta)
    x3 = updateLat(lattice, k3, 0.5 * dt)
    k4 = getFlowValues(SFlow, x3, time + dt, beta)

    F = (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
    return updateLat(lattice, F, dt)

def makeSolnTerms(termList):
    Soln = []
    for Order in termList:
        orderList = []
        for term_string in Order:
            orderList.append(makeTermFromString(term_string))
        Soln.append(orderList)
    return Soln 


def getFlowTerms(Soln):
    FlowList = []
    for Order in Soln:
        orderList = []
        for term in Order:
            dTerms = term.partial("x")
            for dT in dTerms:
                orderList.append(dT)
                orderList = sf.ReduceTermList(orderList)
        FlowList.append(orderList)
    return FlowList


def getLappTerms(Soln):
    LappList = []
    for Order in Soln:
        orderList = []
        for term in Order:
            ddTerms = sf.Laplacian(term, "x")
            for ddT in ddTerms:
                orderList.append(ddT)
                orderList = sf.ReduceTermList(orderList)
        LappList.append(orderList)
    return LappList

def main():
    import ProgressBar as pb

    LatSize = 16 # Volume (needs to be a perfect square)
    lattice = np.empty((LatSize,3))
    for i in range(LatSize):
        while True:
            sample = np.random.uniform(-1., 1., (3))
            if np.linalg.norm(sample) <= 1:
                lattice[i,:] = sample / np.linalg.norm(sample)
                break
    print(lattice)
    print(np.einsum("ij,ij->i",lattice, lattice))

    Order0 = ["0.125<y,__|y,+1>"]
    Order1 = ["0.05<y,__,__|y,+1,+1>", "-0.025<y,+1,__|y,__,__><y,__,__|y,__,+1>", "0.00416667<y,__|y,+1><y,__|y,+1>"]
    SolnStrings = [Order0, Order1]

    Soln = makeSolnTerms(SolnStrings)
    Flow = getFlowTerms(Soln)
    Lapp = getLappTerms(Soln)

    print("Soln: ", Soln)
    print("Flow: ", Flow)
    print("Lapp: ", Lapp)

    #''' # This compares results when integrate with different time steps 
    dts = [1, 0.1, 0.01, 0.001, 0.0001, 0.00001]
    lats, lapps = [], []
    for dt in dts:
        clattice = np.array(lattice)
        times = np.arange(0.0, 1.0, dt)
        lapp = 0
        pb1 = pb.ProgressBar(len(times), prefix=dt)
        for i, t in enumerate(times):
            lapp += 0.5 * dt * getFlowValues(Lapp, clattice, t, 1.263)
            clattice = RK4(Flow, clattice, t, dt, 1.263)
            lapp += 0.5 * dt * getFlowValues(Lapp, clattice, t+dt, 1.263)
            pb1.print(i)
        print(clattice)
        print(np.einsum("ij,ij->i", clattice, clattice))
        print(lapp)
        lats.append(clattice)
        lapps.append(lapp)

    for i in range(1, len(lats)):
        print("\n{} - {}:\nlattice:\n".format(i,i-1), lats[i]-lats[i-1])
        print("Laplacian:", lapps[i]-lapps[i-1])

    #'''
    '''
    # This is to determine how to integrate forwards and backwards properly so that x == f-1(f(x))
    # RK4 is not time reversible! If we want this property we must use a different integrator, such as leapfrog
    mom = np.zeros(np.shape(lattice))
    cLat = np.array(lattice)
    dt = 0.1
    times = np.arange(0.0, 1.0, dt)
    for t in times:
        cLat, mom = LeapFrog(Flow, cLat, mom, time=t, dt=dt, beta=1.0)
    fLat = np.array(cLat)
    for t in reversed(times):
        cLat, mom = LeapFrog(Flow, cLat, mom, time=t, dt=-dt, beta=1.0)
    print(lattice-cLat)
    print(mom)

    rk4Lat = np.array(lattice)
    for t in times:
        rk4Lat, _ = RK4(Flow, rk4Lat, t, dt, beta=1.0)

    print(rk4Lat-fLat)
    '''

if __name__ == "__main__":
    main()