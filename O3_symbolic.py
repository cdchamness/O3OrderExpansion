from itertools import product
import numpy as np

class InnerProduct(object):
    def __init__(self, bra=[], ket=[], index_type="y", inside="", scalar=1.0, xKdelta=[], partialIndexType='', muKdeltas=[]):
        if len(bra) != len(ket): 
            raise TypeError("bra and ket are not the same size! bra: {} with length: {} vs ket: {} with length: {}".format(bra, len(list(bra)), ket, len(list(ket))))
        self.bra = list(bra)
        self.ket = list(ket)
        self.index_type = str(index_type)
        self.inside = str(inside)
        self.scalar = float(scalar)
        self.xKdelta = list(xKdelta)
        self.partialIndexType = str(partialIndexType)
        self.muKdeltas = list(muKdeltas)
        self.reduce() 

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        myStr=""
        self.reduce()
        if self.scalar != 1.0:
            myStr += str(round(self.scalar, 8)) + '<' + self.index_type
        else:
            myStr += '<' + self.index_type
        for i in self.bra:
            myStr += ","
            if i == 0:
                myStr += "__"
            else:
                if i > 0:
                    myStr += "+"
                myStr += str(i)
        myStr += "|" + self.inside
        if len(self.inside) != 0:
            myStr += "|"
        myStr += self.index_type   
        for i in self.ket:
            myStr += ","
            if i == 0:
                myStr += "__"
            else:
                if i > 0:
                    myStr += "+"
                myStr += str(i)
        myStr += ">"
        if self.xKdelta != []:
            myStr += "d{" + self.partialIndexType + ";" + self.index_type
            if self.xKdelta != [False] * len(self.xKdelta):
                for i in self.xKdelta:
                    myStr += ","
                    if i == 0:
                        myStr += "__"
                    else:
                        if i > 0:
                            myStr += "+"
                        myStr += str(i)
            myStr += "}"
        if self.muKdeltas != []:
            for muKdelta in self.muKdeltas:
                myStr += "d{" + self.index_type + ";" + self.index_type
                for i in muKdelta:
                    myStr += ","
                    if i == 0:
                        myStr += "__"
                    else:
                        if i > 0:
                            myStr += "+"
                        myStr += str(i)
                myStr += "}"
        return myStr

    def __mul__(self, other):
        cp = self.copy() 
        if isinstance(other, int) or isinstance(other, float): # scalar multiplication, returns InnerProduct
            cp.scalar *= float(other)
            return cp 

        if isinstance(other, list): # multiply each element in the list, returns List of InnerProducts or Terms
            newList = []
            for val in other:
                newList += cp * val
            return newList

        if isinstance(other, Term): # Returns a List of Terms, as it could produce multiple Terms
            if cp.inside == "":
                newIPList = list(other.IPList) + [cp]
                return [Term(newIPList)]
            elif cp.inside == "T":
                newIPList1, newIPList2 = [], []
                for IP in other.IPList:
                    if IP.inside == "T":
                        Term1, Term2 = cp * IP
                        newIPList1 += Term1.IPList
                        newIPList2 += Term2.IPList
                    elif IP.inside == "":
                        newIPList1 += [IP]
                        newIPList2 += [IP]
                    else:
                        raise TypeError("This should be inaccessible. If you are seeing this something went wrong way before here")
                    
                if newIPList1 == newIPList2: # This should only be the case if 'other' had no IPs with inside == "T". So just add cp to newIPList1 and return the Term.
                    newIPList1 += [cp]
                    return [Term(newIPList1)]
                else: # This should only be the case if there was an InnerProduct that had inside == "T" which should result in 2 terms.
                    return [Term(newIPList1), Term(newIPList2)]

        if isinstance(other, InnerProduct): #This will return a list of Terms as it could result in 2 Terms.
            if cp.index_type == other.index_type:
                if cp.inside == "T" and other.inside == "T":
                    # This is where the magic happens
                    IP1 = InnerProduct(cp.bra, other.bra, index_type=cp.index_type, scalar=cp.scalar*other.scalar, xKdelta=cp.xKdelta, partialIndexType=cp.partialIndexType, muKdeltas=cp.muKdeltas+other.muKdeltas)
                    IP2 = InnerProduct(cp.ket, other.ket, index_type=cp.index_type, scalar=1.0, xKdelta=other.xKdelta, partialIndexType=other.partialIndexType)
                    IP3 = InnerProduct(cp.ket, other.bra, index_type=cp.index_type, scalar=-1.0*cp.scalar*other.scalar, xKdelta=cp.xKdelta, partialIndexType=cp.partialIndexType, muKdeltas=cp.muKdeltas+other.muKdeltas)
                    IP4 = InnerProduct(cp.bra, other.ket, index_type=cp.index_type, scalar=1.0, xKdelta=other.xKdelta, partialIndexType=other.partialIndexType)
                    return [Term(list([IP1,IP2])), Term(list([IP3,IP4]))]
                else:
                    return [Term([cp, other])]
            else:
                raise TypeError("IndexType Error: index_type1: {} and index_type2: {} do not match")



    def copy(self):
        return InnerProduct(bra=self.bra, ket=self.ket, index_type=self.index_type, inside=self.inside, scalar=self.scalar, xKdelta=self.xKdelta, partialIndexType=self.partialIndexType, muKdeltas=self.muKdeltas)

    def set_index_type(self, newIndexType):
        self.index_type = newIndexType

    def extend(self, amount=1):
        extender = [False] * amount
        self.bra += extender
        self.ket += extender

    def extendToLength(self, targetLength):
        currentLength = len(self.bra)
        diff = targetLength - currentLength
        if diff < 0:
            print("Cannot extend {} to {} as {} is smaller than {}".format(currentLength, targetLength, currentLength, targetLength))
        else:
            self.extend(amount=diff)

    def reduce(self):
        if self.inside == "TT":
            self.inside = ""
            self.scalar *= -2.0

        if len(self.muKdeltas) > 0:
            for muKdelta in list(self.muKdeltas):
                if len(muKdelta) > 0 and muKdelta == [0] * len(muKdelta):
                    self.muKdeltas.remove(muKdelta)
        
        return self

    def partial(self, partialIndexType="x"):
        cp = self.copy()
        if cp.bra == cp.ket:
            return []

        if partialIndexType != cp.index_type:
            IP1 = InnerProduct(bra=cp.bra, ket=cp.ket, index_type=cp.index_type, inside=cp.inside + "T", scalar = -cp.scalar, xKdelta=cp.bra, partialIndexType=partialIndexType, muKdeltas=cp.muKdeltas)
            IP2 = InnerProduct(bra=cp.bra, ket=cp.ket, index_type=cp.index_type, inside=cp.inside + "T", scalar = cp.scalar, xKdelta=cp.ket, partialIndexType=partialIndexType, muKdeltas=cp.muKdeltas)
            return [IP1, IP2]

        elif partialIndexType == cp.index_type:
            d_IPs= []

            # apply the partial on the bra
            bra_shifts = self.getShifts(cp.bra)
            bra_shift_count = len(bra_shifts)
            
            if bra_shift_count == 0:
                # This is the derivative hitting the same place again, which is non-zero, but creates no muKdeltas
                d_IPs.append(InnerProduct(bra=cp.bra, ket=cp.ket, index_type=cp.index_type, inside=cp.inside + "T", scalar = -cp.scalar, partialIndexType=""))

            elif bra_shift_count % 2  == 0:
                AllmuKdeltas = self.getAllmuKdeltas(cp.bra)
                for next_muKdeltas in AllmuKdeltas:                    
                    # make a new IP for each possible pairing
                    d_IPs.append(InnerProduct(bra=cp.bra, ket=cp.ket, index_type=cp.index_type, inside=cp.inside + "T", scalar = -len(AllmuKdeltas) * cp.scalar, partialIndexType="", muKdeltas=next_muKdeltas))

            # Finished the partial on bra, now do the same for the kets
            ket_shifts = self.getShifts(cp.ket)
            ket_shift_count = len(ket_shifts)

            if ket_shift_count == 0:
                # Same as bra_shift_count == 0, but you dont get the extra minus sign
                 d_IPs.append(InnerProduct(bra=cp.bra, ket=cp.ket, index_type=cp.index_type, inside=cp.inside + "T", scalar = cp.scalar, partialIndexType=""))

            elif ket_shift_count % 2 == 0:
                AllmuKdeltas = self.getAllmuKdeltas(cp.ket)
                for next_muKdeltas in AllmuKdeltas:
                    # make a new IP for each possible pairing
                    d_IPs.append(InnerProduct(bra=cp.bra, ket=cp.ket, index_type=cp.index_type, inside=cp.inside + "T", scalar = len(AllmuKdeltas) * cp.scalar, partialIndexType="", muKdeltas=next_muKdeltas))
            return d_IPs

    def doubleFactorial(self, num: int) -> int:
        if num == 0 or num == 1:
            return 1
        else:
            return num * self.doubleFactorial(num-2)

    def getPairings(self, index_list):
        list_len = len(index_list)
        if list_len % 2 != 0:
            print("`index_list` must be even, argument given: {} has {} elements".format(index_list, list_len))
            return None

        if list_len == 2:
            return frozenset({index_list[0], index_list[1]})

        out = []
        for i in range(1, list_len):
            pair = frozenset({index_list[0], index_list[i]})
            other_pairs = self.getPairings(index_list[1:i] + index_list[i+1:])
            if isinstance(other_pairs, frozenset):
                new_set = {pair, other_pairs}
                if new_set not in out:
                    out.append(new_set)
                
            else:
                for partial_pairing in other_pairs:
                    new_set = set()
                    new_set.add(pair)
                    for frozen_pair in partial_pairing:
                        new_set.add(frozen_pair)
                    if new_set not in out:
                        out.append(new_set)
        
        num = 0
        while num < len(out):
            remove_this_set = False
            pairing = out[num]
            for a_set in pairing:
                if len(a_set) == 1:
                    remove_this_set = True

            if remove_this_set:
                out.remove(pairing)
            else:
                num += 1

        return out

    def getShifts(self, l: list):
        out = []
        for index, value in enumerate(l):
            v = abs(value)
            while v > 0:
                out.append(index)
                v -= 1
        return out

    def getAllmuKdeltas(self, state: list):
        AllmuKdeltas = []
        shifts = self.getShifts(state)
        pairings = self.getPairings(shifts)
        
        if isinstance(pairings, list):
            for pairing in pairings:
                muKdeltas = []
                for pair in pairing:
                    muKdelta = [0] * len(state)
                    for val in pair:
                        muKdelta[val] = 1 if state[val] > 0 else -1 
                    muKdeltas.append(muKdelta)
                AllmuKdeltas.append(muKdeltas)
        
        elif isinstance(pairings, frozenset):
            muKdelta = [0] * len(state)
            for val in pairings:
                muKdelta[val] = 1 if state[val] > 0 else -1
            AllmuKdeltas.append([muKdelta])
        
        else:
            raise TypeError("This should be unreachable")
        
        return AllmuKdeltas


class Term(object):

    def __init__(self, IPList=[]):
        for elem in IPList:
            if not isinstance(elem, InnerProduct):
                raise TypeError("Element of IPList is not an InnerProduct. {} has Type: {}".format(elem, type(elem)))
        self.IPList = [ip.copy() for ip in IPList]
        self.Tindex = [i for i, ip in enumerate(self.IPList) if ip.inside == "T"]
        self.simple_reduce()
        self.Val = self.getValue()

    def copy(self):
        newList = []
        for ip in self.IPList:
            newList += [ip.copy()]
        return Term(newList)

    def getScalar(self):
        return self.IPList[0].scalar

    def __str__(self):
        termString = ""
        for ip in self.IPList:
            termString += str(ip)
        return termString

    def __repr__(self):
        return str(self)

    def set_index_type(self, newIndexType):
        for ip in self.IPList:
            ip.set_index_type(newIndexType)

    def simple_reduce(self): 
        # Extend IPs with lower bra_ket lengths so that all have the same length
        max_len = max([len(ip.bra) for ip in self.IPList])
        for i, ip in enumerate(self.IPList):
            ip.extendToLength(max_len)
            # This puts the prod(scalar) on the first IP
            if i != 0:
                self.IPList[0] *= ip.scalar
                ip.scalar = 1.0
        
        # Enforce the xKdelta, there should be no issues by doing this here in the order expansion calculation
        count=0 # counter to make sure we only get in this if statement once 
        for ip in self.IPList:
            if ip.xKdelta != []:
                # We should only be able to reach here once per "Term", if 2 IP have xKdelta's something went wrong
                shift = ip.xKdelta
                newIndex = ip.partialIndexType
                count += 1
        if count > 1: # Ensures we didnt enforce xKdelta's multiple times, shouldn't ever be an issue
            raise TypeError("Multiple runs of xKdelta!! count: {}".format(count))
        if count == 1: # Only update InnerProducts if we encounter an xKdelta
            for ip2 in self.IPList:
                ip2.bra = [x - y for x, y in zip(ip2.bra, shift)]
                ip2.ket = [x - y for x, y in zip(ip2.ket, shift)]  
                ip2.set_index_type(newIndex)
                ip2.partialIndexType = ''
                ip2.xKdelta =[]

    def fullReduce(self):
        self.muReduce()
        self.CollapseSums()
        self.Val = self.getValue()

    def muReduce(self):
        collapse_index_list = []
        for ip in self.IPList:
            while len(ip.muKdeltas) > 0:
                pop_muKdelta = ip.muKdeltas.pop()
                # As the muKdelta should have no overlap in indicies it is fine to do these all here and Collapse the sums after
                final_index = [index for index, item in enumerate(pop_muKdelta) if item != 0][-1] #len(muKdelta) - muKdelta[::-1].index(True) - 1
                collapse_index_list.append(final_index)
                for ip2 in self.IPList:
                    if ip2.bra[final_index] == pop_muKdelta[final_index]:
                        ip2.bra = [x - y for x, y in zip(ip2.bra, pop_muKdelta)]
                    if ip2.ket[final_index] == pop_muKdelta[final_index]:
                        ip2.ket = [x - y for x, y in zip(ip2.ket, pop_muKdelta)]

        # Collapse sums over the indicies that were killed by the muKdeltas they should be all zeros anyway given our method above
        # Either they were on -> which were then turn off by the xor, or they were off and thus skipped the xor in either case they are off
        for ip3 in self.IPList:
            ip3.bra = [val for (i, val) in enumerate(ip3.bra) if i not in collapse_index_list]
            ip3.ket = [val for (i, val) in enumerate(ip3.ket) if i not in collapse_index_list]
        
        # This removes terms that are pointless <x_|x_> == 1, <x|x> == 1, etc.
        i = 0
        carry_scalar = 1.0
        while i < len(self.IPList):
            ip = self.IPList[i]
            ip.scalar *= carry_scalar
            if ip.bra == ip.ket and len(self.IPList) != 1 and ip.inside == "":
                carry_scalar = float(ip.scalar)
                del self.IPList[i]
            else:
                i+=1


    def partial(self, partialIndexType="x"):
        out = []
        for i, ip in enumerate(self.IPList):
            newIPs = ip.partial(partialIndexType=partialIndexType)
            otherIPs = self.IPList[:i] + self.IPList[i+1:]
            if len(otherIPs) == 0:
                for newIP in newIPs:
                    out.append(Term([newIP]))
            else:
                for newIP in newIPs:
                    res = newIP * Term(otherIPs)
                    for newTerm in res:
                        out.append(newTerm)
        return out


    def CollapseSums(self):
        IP_count = len(self.IPList)
        i = 0
        while i < len(self.IPList[0].bra):
            counter = 0
            val = self.IPList[0].bra[i]
            for IP in self.IPList:
                if IP.bra[i] == val and IP.ket[i] == val:
                    counter += 1
            
            if counter == IP_count:
                for IP2 in self.IPList:
                    IP2.bra = IP2.bra[:i] + IP2.bra[(i+1):]
                    IP2.ket = IP2.ket[:i] + IP2.ket[(i+1):]
                self.IPList[0].scalar *= 4.0
            
            else:
                i += 1

    def getValue(self, lattice=None):
        if type(lattice) == type(None):
            length = 6 #(len(self.IPList[0].bra) + 1)
            lattice = np.empty((length*length,3))
            np.random.seed(222)
            for i in range(length*length):
                while True:
                    sample = np.random.uniform(-1., 1., (3))
                    if np.linalg.norm(sample) <= 1:
                        lattice[i,:] = sample / np.linalg.norm(sample)
                        break
                    else: 
                        length = int(np.sqrt(np.shape(lattice)[0]))

        copies = 2 * len(self.IPList)
        lat_shape = np.shape(lattice)
        T = np.array(
            [[[0, 0, 0], [0, 0,-1], [ 0, 1, 0]], 
            [[0, 0, 1], [0, 0, 0], [-1, 0, 0]],
            [[0,-1, 0], [1, 0, 0], [ 0, 0, 0]]])


        shift_shape = [copies] + [x for x in lat_shape]
        bra_ket_lats = np.empty(shift_shape)


        steps = len(self.IPList[0].bra)
        if steps == 0:
            return np.array([1.0])
        neighbors = [(0,1), (0,-1), (1,1), (1,-1)]
        shifts = product(neighbors, repeat=steps)

        vect = False
        for IP in self.IPList:
            if IP.inside == 'T':
                vect = True


        if vect == False:
            inter_shape = [length*length]
            result_shape = 1
        elif vect == True:
            inter_shape = [length*length,3]
            result_shape = [length*length, 3]


        result = np.zeros(result_shape)


        for shift in shifts:
            inter = np.ones(inter_shape)
            for i in range(copies):
                bra_ket_lats[i,:,:] = np.array(lattice)
                IPNum = i // 2
                braket_type = i % 2

                if braket_type == 0:
                    instruct = self.IPList[IPNum].bra
                elif braket_type == 1:
                    instruct = self.IPList[IPNum].ket
                else:
                    print('Error on bra/ket retrevial')
                    return None

                for mu in range(steps):
                    if instruct[mu] == 0:
                        pass
                    elif isinstance(instruct[mu], int):
                        shifter = np.reshape(bra_ket_lats[i,:,:], (length,length,3))
                        shifter = np.roll(shifter, axis=shift[mu][0], shift=shift[mu][1]*instruct[mu])
                        bra_ket_lats[i,:,:] = np.reshape(shifter, (length*length,3))
                    else:
                        print('Error on getting shift from instruct: instruct[mu] = {}'.format(instruct[mu]))
                        return None

            for bra_index in range(0,copies,2):
                ket_index = bra_index + 1
                IP_index = bra_index // 2
                if self.IPList[IP_index].inside == '':
                    # do inner product
                    a = np.einsum('ij,ij->i', bra_ket_lats[bra_index,:,:], bra_ket_lats[ket_index,:,:])
                    #a = self.IPList[IP_index].scalar * np.einsum('ij,ij->i', bra_ket_lats[bra_index,:,:], bra_ket_lats[ket_index,:,:])

                    if vect == False:
                        inter *= a
                    elif vect == True:
                        inter = np.einsum('i,ij->ij', a, inter)

                elif self.IPList[IP_index].inside == 'T':
                    # do inner product with T inserted
                    # Removed part of 'a' calculation that multiplied by self.IPList[IP_index].scalar 
                    # This may cause issues in certain unknown cases. IDK. This is what is done on IP
                    # Without the 'T' inside. So maybe it is what I should have done as I did above
                    # a = self.IPList[IP_index].scalar * np.einsum('ij,jkl,ik->il',bra_ket_lats[bra_index,:,:], T, bra_ket_lats[ket_index,:,:])
                    a = np.einsum('ij,jkl,ik->il',bra_ket_lats[bra_index,:,:], T, bra_ket_lats[ket_index,:,:])
                    inter *= a
                else:
                    print('Error on getting the inside')
                    return None
            if vect == False:
                result += np.sum(inter,axis=0)
            elif vect == True:
                result += inter
        return result

            

    def __mul__(self, other): # Returns a List of Terms, as it could produce multiple Terms, see InnerProduct Implementation for more details
        if isinstance(other, InnerProduct):
            return other * self

        if isinstance(other, Term): # Returns a list of Terms as it is possible to make 2 Terms
            if self.Tindex != [] and other.Tindex != []:
                T1, T2 = self.IPList[self.Tindex[0]] * other.IPList[other.Tindex[0]]
                extra_IPs = self.IPList[:self.Tindex[0]] + self.IPList[self.Tindex[0]+1:] + other.IPList[:other.Tindex[0]] + other.IPList[other.Tindex[0]+1:]
                return [Term(extra_IPs + T1.IPList), Term(extra_IPs + T2.IPList)]
            return [Term(self.IPList + other.IPList)]

        if isinstance(other, float) or isinstance(other, int): # Returns a Term
            newIPZero = self.IPList[0].copy() * other
            cp = self.copy()
            cp.IPList[0] = newIPZero
            return cp 

    def __add__(self, other): 
        cp = self.copy()
        if isinstance(other, Term):
            if np.abs(np.sum(cp.Val - other.Val)) < 1e-10:
                cp.IPList[0].scalar += other.IPList[0].scalar
                return [Term(list(cp.IPList))]
            return list([cp, other.copy()])
        if isinstance(other, list):
            if len(other) == 1:
                # If there is one element in the list, add the element with self
                return self + other[0]
            else:
                # If there is multiple elements in the list, add the element as a new item to the list
                return other + [self]

        print("Cannot add type: {} with type: {}".format(type(self), type(other)))
        return None

    def __radd__(self, other):
        return self.__add__(other)


