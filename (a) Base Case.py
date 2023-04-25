# Project 2: Diffusion-Limited Aggregation
# Category: Base Case
# ----------------------------- Basic Info ------------------------------
# Lattice site:                 Square site
# Dimensions:                   2
# Initial cluster:              A point at the origin
# Prob. to move at directions:  Same, all as 1/4
# Starting point:               Randomly on a circle
# Remove when it's too far:     Yes
# Periodicity of the lattice:   None

# ----------------------------- Parameters -----------------------------
# Lattice side length:          2 * latticeRange, [-latticeRange, latticeRange]
# Total num. of particle:       numOfParticles
# Sticking probability at distance i: P_nn[i]
# P_nn[1] = 1, P_nn[i] = 0,     i >= 2 or i == 0
# Particle starting radius:     R_start = R_current_max + delta (delta is set to 5)
# Largest clustering radius:    R_max
# Particle removal radius:      R_remove
# Progress of the calculation:  step

import math
import numpy as np
import matplotlib.pyplot as plt

#######################################################################################
################################ Parameter Declaration ################################
#######################################################################################

############# Region 1: You can freely change the parameters below #############
# Lattice index: [-latticeRange, latticeRange]
latticeRange = 110
# Number of particles to be added
numOfParticles = 5000
# Maximum simulation range, R_max should be less than R_remove
R_max = 100
# R_remove should be slightly less than latticeRange
R_remove = 105
# Change the variable "step" to modify the display
step = 10

################# Region 2: Do not modify the parameters below #################
latticeSide = 2 * latticeRange
# Lattice type
lattice = np.zeros((latticeSide, latticeSide))
# Starting radius displacement w.r.t. current maximum
delta = 5
# Current farthest point, use to determine the next starting radius
R_current_max = 1

#######################################################################################
################################# Function Definition #################################
#######################################################################################

# Construct the lattice site
# Not useful, could be deleted
def constructLatticeSite(latticeSide):
    return np.zeros((latticeSide, latticeSide))

# Set the initial cluster
# The function modifies the 2-D lattice array directly
# The initial cluster is set to be one point at the origin
def setInitialCluster():
    ######## You can change the initial cluster here ########
    lattice[latticeSide//2][latticeSide//2] = 1
    global R_current_max
    indices = np.argwhere(lattice != 0)
    max_index = 1
    for i, j in indices:
        dist = i + j - latticeSide
        if dist > max_index:
            R_current_max = dist
    return

# Randomly generate a point at the radius specified by "radius"
# The function modifies the 2-D lattice array directly
def randGenerate(lattice, radius):
    theta = 2*np.pi*np.random.random()
    x = int(latticeSide//2 + radius*np.cos(theta))
    y = int(latticeSide//2 + radius*np.sin(theta))
    lattice[x][y] = 1
    return x, y

# Check whether the given point sticks to the cluster
# May be deterministic or probabilistic
# ----- Return values -----
# return 1:     Sticks to the cluster
# return 0:     Not sticks to the cluster
def sticksToCluster(x, y):
    if(lattice[x-1][y] == 1 or lattice[x+1][y] == 1 or lattice[x][y-1] == 1 or lattice[x][y+1] == 1):
        return 1
    else:
        return 0

# Conduct the simulation of a single particle
# ----- Input parameters -----
# x, y: Coordinates of the current simulating point
# ----- Return values -----
# return 2:     Sticks to the cluster exceeding R_max
# return 1:     Sticks successfully within R_max limit
# return 0:     Moves further than R_remove
def simulation(x, y):
    while(True):
        temp = sticksToCluster(x, y)
        curDist = pow(x-latticeSide//2, 2)+pow(y-latticeSide//2, 2)
        if(temp == 1):
            if(curDist > pow(R_max, 2)):
                return 2
            else:
                # Modify the next starting radius
                global R_current_max
                if(math.sqrt(curDist) > R_current_max):
                    R_current_max = int(math.sqrt(curDist))
                return 1
        elif(curDist > pow(R_remove, 2)):
            lattice[x][y] = 0
            return 0
        else:
            rand = int(4 * np.random.random())
            if(rand == 0):
                lattice[x][y] = 0
                lattice[x-1][y] = 1
                x, y = x-1, y
            elif(rand == 1):
                lattice[x][y] = 0
                lattice[x+1][y] = 1
                x, y = x+1, y
            elif(rand == 2):
                lattice[x][y] = 0
                lattice[x][y-1] = 1
                x, y = x, y-1
            else:
                lattice[x][y] = 0
                lattice[x][y+1] = 1
                x, y = x, y+1
            # print(x, y)
            # print(curDist)
            # print(pow(R_remove, 2))

# Display the calculation progress
temp_step = step
def progressDisplay(count, numOfParticles):
    global step
    progress = int(count / numOfParticles * 100)
    if(progress >= step):
        temp = int(step/temp_step)
        print("Progress:", "-" * temp, step, end="")
        print("%")
        step += temp_step

def main():
    # lattice = constructLatticeSite(latticeSide)
    setInitialCluster()
    successCount = 0
    unsuccessCount = 0
    for count in range(numOfParticles):
        progressDisplay(count, numOfParticles)
        x, y = randGenerate(lattice, R_current_max + delta)
        simuResult = simulation(x, y)
        # Handle the return value of the simuResult
        if(simuResult == 2):
            print("Sticks to the cluster exceeding R_max")
        elif(simuResult == 1):
            successCount += 1
        elif(simuResult == 0):
            unsuccessCount += 1
    print("-------------------- Detailed Info --------------------")
    print("Number of points successfully clustered:", successCount)
    print("Number of points removed:", unsuccessCount)
    print("Successfully clustering rate:", "{:.2f}%".format(successCount/numOfParticles*100))
    print("-------------------- End Info --------------------")

def plotFigure():
    plt.imshow(lattice, cmap='gray')
    plt.colorbar()
    plt.show()

main()
plotFigure()