import math
import numpy as np
import matplotlib.pyplot as plt
import time
import numba

start_time = time.time()

@numba.jit(nopython=True)
def main():
    ############# Region 1: You can freely change the parameters below #############
    # Lattice index: [-latticeRange, latticeRange]
    latticeRange = 500
    # Number of particles to be added
    numOfParticles = 500
    # Maximum simulation range, R_max should be less than ~latticeRange/3
    R_max = 300
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

    # lattice = constructLatticeSite(latticeSide)
    
    # Set the initial cluster
    # The function modifies the 2-D lattice array directly
    # The initial cluster is set to be one point at the origin

    ######## You can change the initial cluster here ########
    lattice[latticeSide//2][latticeSide//2] = 1/numOfParticles
    indices = np.argwhere(lattice > 0)
    max_index = 1
    for i, j in indices:
        dist = i + j - latticeSide
        if dist > max_index:
            R_current_max = dist

    count = 1
    successCount = 0
    unsuccessCount = 0
    while count <= numOfParticles:

        # Display the calculation progress
        temp_step = step
        progressDisplay = int(count/numOfParticles*100)
        if(progressDisplay >= step):
            temp = int(step/temp_step)
            '''
            print("Progress:", "-" * temp, step, end="")
            print("%")
            '''
            step += temp_step
        
        # Randomly generate a point at the radius specified by "radius"
        # The function modifies the 2-D lattice array directly
        theta = 2*np.pi*np.random.random()
        radius = R_current_max + delta
        progress = float(count)/numOfParticles
        x = int(latticeSide//2 + radius*np.cos(theta))
        y = int(latticeSide//2 + radius*np.sin(theta))
        lattice[x][y] = progress

        # Conduct the simulation of a single particle
        # ----- Input parameters -----
        # x, y: Coordinates of the current simulating point
        # ----- Return values -----
        # return 2:     Sticks to the cluster exceeding R_max
        # return 1:     Sticks successfully within R_max limit
        # return 0:     Removed
        simuResult = 0
        while(True):
            temp = 0
            # Check whether the given point sticks to the cluster
            # May be deterministic or probabilistic
            # ----- Return values -----
            # return 1:     Sticks to the cluster
            # return 0:     Not sticks to the cluster
            curDist = (x-latticeSide//2)*(x-latticeSide//2) + (y-latticeSide//2)*(y-latticeSide//2)
            if(curDist > (R_current_max + 2)*(R_current_max + 2)):
                temp = 0
            elif(lattice[x-1][y] > 0 or lattice[x+1][y] > 0 or lattice[x][y-1] > 0 or lattice[x][y+1] > 0):
                temp = 1
            temp = 0

            if(temp == 1):
                if(curDist > R_max*R_max):
                    simuResult = 2
                    break
                else:
                    # Modify the next starting radius
                    if(math.sqrt(curDist) > R_current_max):
                        R_current_max = int(math.sqrt(curDist))
                    simuResult = 1
                    break
            elif(curDist > (min(max(3*R_current_max, R_current_max + delta), latticeRange-1)) * (min(max(3*R_current_max, R_current_max + delta), latticeRange-1))):
                lattice[x][y] = BackgroundColorPara
                simuResult = 0
                break
            else:
                rand = int(4 * np.random.random())
                if(rand == 0):
                    lattice[x][y] = BackgroundColorPara
                    lattice[x-1][y] = progress
                    x, y = x-1, y
                elif(rand == 1):
                    lattice[x][y] = BackgroundColorPara
                    lattice[x+1][y] = progress
                    x, y = x+1, y
                elif(rand == 2):
                    lattice[x][y] = BackgroundColorPara
                    lattice[x][y-1] = progress
                    x, y = x, y-1
                else:
                    lattice[x][y] = BackgroundColorPara
                    lattice[x][y+1] = progress
                    x, y = x, y+1
        
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
    '''
    print("Successfully clustering rate:", "{:.2f}%".format(successCount/(successCount+unsuccessCount)*100))
    '''
    print("-------------------- End Info --------------------")

'''
def plotFigure():
    cmap = plt.cm.gist_rainbow
    cmap.set_under('w')
    plt.imshow(lattice, cmap=cmap, vmin=0, vmax=1)
    plt.colorbar()
    plt.show()
'''
main()
print('Time:', time.time()-start_time)
'''
plotFigure()
'''