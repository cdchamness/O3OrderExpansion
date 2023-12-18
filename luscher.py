# compact fields
# it solves the equation
# Y' = Z(Y) Y
#
# Finds value of y for a given x using step size h 
# and initial value y0 at x0.
# the following functions are needed.
# Z is the generator of the flow in the ODE (see above)
# note that this algorith assumes no time dependence for the flow generator
# evoY evolves the Y fields (which may have a complicated evolution)
#
def rungeKuttaCompact(x0, y0, x, Nint,Z,evoY): 
    # Count number of iterations using step size or 
    # step height h 
    h = (x - x0)/Nint 
    # Iterate for number of iterations 
    Y = y0 
    for i in range(0, Nint):
        #print(i)
        "Apply Runge Kutta Formulas to find next value of y"
        #W0 = y
        Z0 = Z(Y)
        W1=evoY(h,0.25*Z0,Y)
        Z1 = Z(W1)
        W2=evoY(h,8.0/9.0*Z1- 17.0/36.0*Z0,W1)
        Z2 = Z(W2)
        Y = evoY(h,3.0/4.0*Z2 - 8.0/9.0*Z1 + 17.0/36.0*Z0,W2)
        # Update next value of x 
        x0 = x0 + h
        # not needed because Z does not depend on x
    
    return Y
