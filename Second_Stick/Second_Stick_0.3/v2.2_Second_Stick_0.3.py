# Project 2: Diffusion-Limited Aggregation
# Category: Second Stick 0.3
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
# P_nn[1] = 0.3, P_nn[2] = 0.15, P_nn[i] = 0, i >= 3 or i == 0
# Particle starting radius:     R_start = R_current_max + delta (delta is set to 5)
# Largest clustering radius:    R_max
# Progress of the calculation:  step

import math
import numpy as np
import matplotlib.pyplot as plt
import time

import os
# Change the working directory to the current file directory
file_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(file_dir)

start_time = time.time()

#######################################################################################
################################ Parameter Declaration ################################
#######################################################################################

############# Region 1: You can freely change the parameters below #############
# Number of particles to be added
numOfParticles = 10000
# Maximum simulation range, R_max should be less than ~latticeRange/3
R_max = 250
# Lattice index: [0, 2 * latticeRange]
latticeRange = R_max*3 + 10
# Change the variable "step" to modify the display
step = 1

################# Region 2: Do not modify the parameters below #################
BackgroundColorPara = -0.1
latticeSide = 2 * latticeRange
# Lattice type
lattice = np.zeros((latticeSide, latticeSide))
lattice.fill(BackgroundColorPara)
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
    lattice[latticeSide//2][latticeSide//2] = 1/numOfParticles
    global R_current_max
    indices = np.argwhere(lattice > 0)
    max_index = 1
    for i, j in indices:
        dist = i + j - latticeSide
        if dist > max_index:
            R_current_max = dist
    return

# Randomly generate a point at the radius specified by "radius"
# The function modifies the 2-D lattice array directly
def randGenerate(lattice, radius, progress):
    theta = 2*np.pi*np.random.random()
    x = int(latticeSide//2 + radius*np.cos(theta))
    y = int(latticeSide//2 + radius*np.sin(theta))
    lattice[x][y] = progress
    return x, y

# Check whether the given point sticks to the cluster
# May be deterministic or probabilistic
# ----- Return values -----
# return 1:     Sticks to the cluster
# return 0:     Not sticks to the cluster
def sticksToCluster(x, y):
    curDist = (x-latticeSide//2)*(x-latticeSide//2) + (y-latticeSide//2)*(y-latticeSide//2)
    if(curDist > (R_current_max + 2)*(R_current_max + 2)):
        return 0
    elif(lattice[x-1][y] > 0 or lattice[x+1][y] > 0 or lattice[x][y-1] > 0 or lattice[x][y+1] > 0):
        randomNumber = np.random.random()
        if(randomNumber < 0.3):
            return 1
    elif(lattice[x-2][y] > 0 or lattice[x+2][y] > 0 or lattice[x][y-2] > 0 or lattice[x][y+2] > 0 or\
        lattice[x-1][y-1] > 0 or lattice[x+1][y-1] > 0 or lattice[x-1][y+1] > 0 or lattice[x+1][y+1] > 0):
        randomNumber = np.random.random()
        if(randomNumber < 0.15):
            return 1
    return 0

# Conduct the simulation of a single particle
# ----- Input parameters -----
# x, y: Coordinates of the current simulating point
# ----- Return values -----
# return 2:     Sticks to the cluster exceeding R_max
# return 1:     Sticks successfully within R_max limit
# return 0:     Removed
def simulation(x, y, progress):
    while(True):
        temp = sticksToCluster(x, y)
        curDist = (x-latticeSide//2)*(x-latticeSide//2) + (y-latticeSide//2)*(y-latticeSide//2)
        if(temp == 1):
            if(curDist > R_max*R_max):
                return 2
            else:
                # Modify the next starting radius
                global R_current_max
                if(math.sqrt(curDist) > R_current_max):
                    R_current_max = int(math.sqrt(curDist))
                return 1
        elif(curDist > (min(max(3*R_current_max, R_current_max + delta), latticeRange-1)) * (min(max(3*R_current_max, R_current_max + delta), latticeRange-1))):
            lattice[x][y] = BackgroundColorPara
            return 0
        else:
            candidates = []
            if lattice[x-1][y] < 0:
                candidates.append((x-1, y))
            if lattice[x+1][y] < 0:
                candidates.append((x+1, y))
            if lattice[x][y-1] < 0:
                candidates.append((x, y-1))
            if lattice[x][y+1] < 0:
                candidates.append((x, y+1))

            if(len(candidates) > 0):
                idx = np.random.randint(len(candidates))
                i, j = candidates[idx]
                lattice[x][y], lattice[i][j] = lattice[i][j], lattice[x][y]
                x, y = i, j

# Display the calculation progress
temp_step = step
def progressDisplay(count, numOfParticles):
    global step
    progress = int(count/numOfParticles*100)
    if(progress >= step):
        temp = int(step/temp_step)
        print("Progress:", "-" * temp, step, end="")
        print("%")
        step += temp_step

def main():
    # lattice = constructLatticeSite(latticeSide)
    setInitialCluster()
    count = 1
    successCount = 0
    unsuccessCount = 0
    while count <= numOfParticles:
        progressDisplay(count, numOfParticles)
        x, y = randGenerate(lattice, R_current_max + delta, float(count)/numOfParticles)
        simuResult = simulation(x, y, float(count)/numOfParticles)
        # Handle the return value of the simuResult
        if(simuResult == 2):
            print("Sticks to the cluster exceeding R_max")
            count += 1
        elif(simuResult == 1):
            successCount += 1
            count += 1
        elif(simuResult == 0):
            unsuccessCount += 1
        
    print("-------------------- Detailed Info --------------------")
    print("Number of points successfully clustered:", successCount)
    print("Number of points removed:", unsuccessCount)
    print("Successfully clustering rate:", "{:.2f}%".format(successCount/(successCount+unsuccessCount)*100))
    print("-------------------- End Info --------------------")

    return successCount/(successCount+unsuccessCount)

def plotFigure(SuccessfullyClusteringRate):
    cmap = plt.cm.gist_rainbow
    cmap.set_under('w')
    plt.imshow(lattice, cmap=cmap, vmin=0, vmax=1)
    plt.colorbar()
    plt.xlim([(latticeRange - R_current_max)*0.9, (latticeRange + R_current_max)*1.1])
    plt.ylim([(latticeRange - R_current_max)*0.9, (latticeRange + R_current_max)*1.1])

    plotname = f'Clustering_rate={"{:.2f}%".format(SuccessfullyClusteringRate*100)}.png'
    plt.savefig(plotname, dpi=500)
    plt.show()

SuccessfullyClusteringRate = main()
print('Time:', time.time()-start_time)
plotFigure(SuccessfullyClusteringRate)
filename = f'Clustering_rate={"{:.2f}%".format(SuccessfullyClusteringRate*100)}.txt'
np.savetxt(filename, lattice, fmt='%.4f')