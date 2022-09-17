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
    Ts = np.array([[[0,0,0],[0,0,-1],[0,1,0]],[[0,0,1],[0.0,0],[-1,0,0]],[[0,-1,0],[1,0,0],[0,0,0]]])
    norm = np.sqrt(np.einsum('ij,ij->i', momentum, momentum))
    sin = np.sin(norm)/(norm + min_float)
    sin2 = 2 * (np.sin(norm/2.0)/(norm+min_float)) ** 2 
    A = np.einsum('ij,jkl->ikl', momentum, Ts)
    AA = np.einsum("ijk,ikl->ijl", A, A)
    return np.eye(3,3) + A * sin[:, None, None] + AA * sin2[:, None, None]

def updateLat(lattice, momentum, dt):
    return np.einsum('ijk,ik->ij', expo(dt * momentum), lattice)



def getTermValue(term, lattice):
    return term.scalar() * term.getValue(lattice=lattice)


def getFlowValues(SFlow, lattice, time, beta):
    F = 0
    orderCoef = beta
    for OrderFlow in SFlow:
        for term in OrderFlow:
            F += orderCoef * getTermValue(term, lattice)
        orderCoef *= beta*T
    
    return F 


def RK4(SFlow, lattice, time, dt, beta, LappFlow = None):
    L = 0 # Default Value to return in LappFlow is None
    k1 = getFlowValues(SFlow, lattice, time, beta)
    x1 = updateLat(lattice, k1, 0.5 * dt)
    k2 = getFlowValues(SFlow, x1, time + 0.5 * dt, beta)
    x2 = updateLat(lattice, k2, 0.5 * dt)
    k3 = getFlowValues(SFlow, x2, time + 0.5 * dt, beta)
    x3 = updateLat(lattice, k3, 0.5 * dt)
    k4 = getFlowValues(SFlow, x3, time + dt, beta)

    F = (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
    out = updateLat(lattice, F, dt)

    if LappFlow is not None:
        l1 = getFlowValues(LappFlow, lattice, time, beta)
        l2 = getFlowValues(LappFlow, x1, time + 0.5 * dt, beta)
        l3 = getFlowValues(LappFlow, x2, time + 0.5 * dt, beta)
        l4 = getFlowValues(LappFlow, x3, time + dt, beta)

        L = (l1 + 2 * l2 + 2 * l3 + l4) / 6.0

    return out, L 

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
    LatSize = 16 # Volume (needs to be a perfect square)
    lattice = np.random.normal(0.0, 1.0, (LatSize, 3))

    Order0 = ["0.125<y,__|y,+1>"]
    Order1 = ["0.05<y,__,__|y,+1,+1>", "-0.025<y,+1,__|y,__,__><y,__,__|y,__,+1>", "0.00416667<y,__|y,+1><y,__|y,+1>"]
    SolnStrings = [Order0, Order1]

    Soln = makeSolnTerms(SolnStrings)
    Flow = getFlowTerms(Soln)
    Lapp = getLappTerms(Soln)

    print("Soln: ", Soln)
    print("Flow: ", Flow)
    print("Lapp: ", Lapp)

if __name__ == "__main__":
    main()