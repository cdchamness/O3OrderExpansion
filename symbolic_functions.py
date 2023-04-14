import numpy as np
from O3_symbolic import InnerProduct, Term


def Laplacian(term, partialIndexType="x", tIndex="a"):
    if isinstance(term, Term):

        #print("Doing Laplacian on: ", term)

        cp = term.copy()
        cp.IPList[0].scalar *= -1
        dT_List = cp.partial(partialIndexType=partialIndexType, tIndex=tIndex)

        #print("Result of 1st derivative: ", dT_List)

        ddTList = []
        for dT in dT_List:

            #print("\nDoing 2nd Derivative on: ", dT)

            ddTs = dT.partial(partialIndexType=partialIndexType, tIndex=tIndex)
            for ddT in ddTs:
                ddTList.append(ddT)
                ddTList = ReduceTermList(ddTList)
        #print("Final Result: ", ddTList)
        return ddTList

    else:
        print("Laplacian must be called on a 'Term' object. Not type: {}".format(type(term)))

def GradiantProduct(T1, T2, partialIndexType='x', tIndex="a"):
    if isinstance(T1, Term) and isinstance(T2, Term):
        outList = []
        dT1_List = T1.partial(partialIndexType=partialIndexType, tIndex=tIndex)
        dT2_List = T2.partial(partialIndexType=partialIndexType, tIndex=tIndex)

        for dT1 in dT1_List:
            for dT2 in dT2_List:
                out = dT1.copy() * dT2.copy()
                for term in out:
                    outList.append(term)
                    outList = ReduceTermList(outList)
        return outList
    
    if isinstance(T1, Term) and isinstance(T2, InnerProduct):
        T2 = Term([T2])
        return GradiantProduct(T1, T2, partialIndexType=partialIndexType, tIndex=tIndex) 
        
    if isinstance(T1, InnerProduct) and isinstance(T2, Term):
        T1 = Term([T1])
        return GradiantProduct(T1, T2, partialIndexType=partialIndexType, tIndex=tIndex) 

    if isinstance(T1, InnerProduct) and isinstance(T2, InnerProduct):
        T1 = Term([T1])
        T2 = Term([T2])
        return GradiantProduct(T1, T2, partialIndexType=partialIndexType, tIndex=tIndex) 

    print("One of T1 (type: {}) or T2 (type: {}) could not be converted into Type: {}".format(type(T1), type(T2), type(Term)))
    return None

def ReduceTermList(TermList):
    # Checks to see if a newest term added to outlist can be reduced given the terms already in TermList
    copyList = list(TermList)
    newTerm = copyList.pop()
    newTerm.fullReduce()
    for i, oldTerm in enumerate(copyList):
        result = oldTerm + newTerm
        if result == None:
            print("ReduceTermList: failure")
            raise(ValueError("ReduceTermList Error"))
        elif len(result) == 1:
            copyList[i] = result[0]
            return copyList
        elif len(result) == 2:
            pass
        else:
            raise(ValueError("ReduceTermList Error"))
    return list(TermList)


def NextOrderGP(CurrentOrderGPList):
    GPTerms = []
    # Extend every IP in every Term for the new mu-type sum
    for T in CurrentOrderGPList:
        Tcp = T.copy()
        for IP in Tcp.IPList:
            IP.extend(1)
            IP.set_index_type("y")

        # Create new IP/Term that has shift in new direction
        S = InnerProduct(bra=[0], ket=[0], index_type="y", scalar=1.0/2.0)
        S.extendToLength(len(Tcp.IPList[0].bra))
        S.ket[-1] = 1
        newTerm = Term([S.copy()])

        nextGPTerms = GradiantProduct(Tcp, newTerm)
        for tt in nextGPTerms:
            GPTerms.append(tt)
            GPTerms = ReduceTermList(GPTerms)
    return GPTerms


def GetClosedLaplacian(TermList):
    # Returns 4 items:
    #   1st: List containing the Minimal Set of Terms Closed under the Laplacian that include TermList
    #   2nd: A dictionary with key = Term.Val, Value = matrix index
    #   3rd: The Laplacian Matrix, with terms in their corresponding matrix locations
    #   4th: The Array corresponding to the TermList wrt the ValDict

    constVal = np.array([1.0])
    closedList = list(TermList)
    ValDict = {str(T.Val):i for i, T in enumerate(closedList)} # Populates current values into Dict
    GPArray = [T.getScalar() for T in closedList]
    LapLists = []
    
    
    # Loops threw closedList, applies Laplacian, 
    #       if term not already in closedList then it is added 
    #       A 0 is also added to GPArray so that it is compatable length
    # Saves result from each Laplacian in LapRes which get added to LapLists
    # Filters out constant value results ( <x+|x+> => Val = [1.0] = const Val)
    i = 0
    while i < len(closedList):
        LapRes = []
        T = closedList[i].copy()
        T.IPList[0].scalar = 1.0
        T.set_index_type("y")
        LTs = Laplacian(T, partialIndexType="x")
        for lt in LTs:
            if lt.Val != constVal:
                if str(lt.Val) not in ValDict.keys():
                    closedList.append(lt)
                    GPArray.append(0)
                    ValDict[str(lt.Val)] = len(closedList) - 1 # Lists start at 0...
                LapRes.append(lt.copy())
        LapLists.append(LapRes)
        i += 1
    
    # Loops threw LapLists to get each LapRes
    # Each term in LapRes finds where that type is in ValDict
    # then creates LapMat (matrix) which is the Laplacian Matrix
    LapMat = np.zeros((len(closedList),len(closedList)))
    for i, LapRes in enumerate(LapLists):
        for term in LapRes:
            j = ValDict[str(term.Val)]
            LapMat[j,i] = term.getScalar()

    return closedList, ValDict, LapMat, np.array(GPArray)


def doNextOrder(prevOrder):
    GPList = NextOrderGP(prevOrder)
    nextOrderList, _, LapMat, GPArray = GetClosedLaplacian(GPList)
    resultArray = np.dot(np.linalg.inv(LapMat), GPArray)
    result = []
    for i, term in enumerate(nextOrderList):
        tcp = term.copy()
        tcp.IPList[0].scalar = resultArray[i]
        result.append(tcp)

    return result


if __name__ == "__main__":
    ip1 = InnerProduct([0], [1], scalar=1.0/8.0) # 0th order Solution (as it is unique and easy to do)
    t1 = Term([ip1])
    FinalResult = [[t1]]
    print("Order 0:\n\t", t1, "\n")
    for order in range(1,6):
        result = doNextOrder(FinalResult[-1])
        print("Order {}:".format(order))
        for term in result:
            print("\t", term, term.Val)
        print("\n")
        FinalResult.append(result)

    




