import functools

from multiprocessing import Pool, Value, Lock, Manager

from sympy import *
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import matplotlib as mpl



def solution_rects(solution):
    T2 = np.array((T(solution[0], solution[1]) * R(solution[2]))).astype(np.float64)
    T3 = np.array((T(solution[3], solution[4]) * R(solution[5]))).astype(np.float64)

    W1_bl = [-1.5, -0.5]
    W1_br = [1.5, -0.5]
    W1_tl = [-1.5, 0.5]
    W1_tr = [1.5, 0.5]

    B1_bl = [-0.5, 0.5]
    B1_br = [0.5, 0.5]
    B1_tl = [-0.5, 2.5]
    B1_tr = [0.5, 2.5]

    B2_bl = T2.dot(np.array([-1.5, -0.5, 1])).tolist()[0:2]
    B2_br = T2.dot(np.array([1.5, -0.5, 1])).tolist()[0:2]
    B2_tl = T2.dot(np.array([-1.5, 0.5, 1])).tolist()[0:2]
    B2_tr = T2.dot(np.array([1.5, 0.5, 1])).tolist()[0:2]

    W2_bl = T2.dot(np.array([1.5, -0.5, 1])).tolist()[0:2]
    W2_br = T2.dot(np.array([3.5, -0.5, 1])).tolist()[0:2]
    W2_tl = T2.dot(np.array([1.5, 0.5, 1])).tolist()[0:2]
    W2_tr = T2.dot(np.array([3.5, 0.5, 1])).tolist()[0:2]

    B3_bl = T3.dot(np.array([-1.0, -0.5, 1])).tolist()[0:2]
    B3_br = T3.dot(np.array([1.0, -0.5, 1])).tolist()[0:2]
    B3_tl = T3.dot(np.array([-1.0, 0.5, 1])).tolist()[0:2]
    B3_tr = T3.dot(np.array([1.0, 0.5, 1])).tolist()[0:2]

    W3_bl = T3.dot(np.array([1.0, -0.5, 1])).tolist()[0:2]
    W3_br = T3.dot(np.array([3.0, -0.5, 1])).tolist()[0:2]
    W3_tl = T3.dot(np.array([1.0, 0.5, 1])).tolist()[0:2]
    W3_tr = T3.dot(np.array([3.0, 0.5, 1])).tolist()[0:2]

    return (
        (W1_bl, W1_br, W1_tl, W1_tr),
        (B1_bl, B1_br, B1_tl, B1_tr),
        (B2_bl, B2_br, B2_tl, B2_tr),
        (W2_bl, W2_br, W2_tl, W2_tr),
        (B3_bl, B3_br, B3_tl, B3_tr),
        (W3_bl, W3_br, W3_tl, W3_tr),
    )





def plot_solution(solution, rects, name):
    rect_W1 = Rectangle(rects[0][0], 3, 1, color="yellow")
    rect_B1 = Rectangle(rects[1][0], 1, 2, color="brown")

    rect_B2 = Rectangle(rects[2][0], 3, 1, deg(solution[2]), color="brown")
    rect_W2 = Rectangle(rects[3][0], 2, 1, deg(solution[2]), color="yellow")

    rect_B3 = Rectangle(rects[4][0], 2, 1, deg(solution[5]), color="brown")
    rect_W3 = Rectangle(rects[5][0], 2, 1, deg(solution[5]), color="yellow")

    fig = plt.figure()

    ax = fig.add_subplot(111)
    ax.add_patch(rect_W1)
    ax.add_patch(rect_B1)
    ax.add_patch(rect_W2)
    ax.add_patch(rect_B2)
    ax.add_patch(rect_W3)
    ax.add_patch(rect_B3)

    ax.grid(True)
    ax.autoscale(True)

    ax.axis('equal')

    #plt.show()
    fig.savefig("out/"+str(name)+".png")
    fig.clf()
    plt.close(fig)
    
def T(x, y):
    """
    Transform
    """
    return Matrix([ [1, 0, x], [0, 1, y], [0, 0, 1] ])

def R(t):
    """
    Rotate
    """
    return Matrix([ [cos(t), -sin(t), 0], [sin(t), cos(t), 0], [0, 0, 1] ])

def RL(T0, angle):
    """
    Rotate local
    """
    return T0 * R(angle) * T0**-1

def try_solve(s1, s2, h2, h3):
    """
    Try to solve the final system with heuristic angle h2, h3
    """
    try: 
        eqs = []

        for e in s1:
            eqs.append(e)

        for e in s2:
            eqs.append(e)

        #for e in s3:
        #    eqs.append(e)

        eqs = [e for e in eqs if e != 0]

        for e in eqs:
            print(e)


        solution = nsolve(eqs, [x2, y2, t2, x3, y3, t3], [-10., -5., h2, 2., 7., h3] )
        #solution = nonlinsolve(eqs, [x2, y2, t2, x3, y3, t3])
        #solution = solve(eqs, [x2, y2, t2, x3, y3, t3])
        return solution
    except Exception:
        return None

def solve_OAB(O, A, B, h2, h3):
    s1 = W2 - (O**-1) * A
    s2 = W3 - (O**-1) * B
    return try_solve(s1, s2, h2, h3)


def solve_print(O, A, B, h2, h3, name):
    solution = solve_OAB(O, A, B, h2, h3)
    if (solution):
        pprint(solution)
        print(solution)
        rects = solution_rects(solution)
        plot_solution(solution, rects, name)
        return 1
    return 0

def solve_print_permutation(lock, idx, param):
    (O, A, B, h2, h3) = param
    with lock:
        i = idx.value
        idx.value += 1
    solve_print(O, A, B, h2, h3, i)
    with lock:
        i = idx.value
        idx.value += 1
    solve_print(O, B, A, h2, h3, i)

x2, y2, t2, x3, y3, t3 = symbols("x2 y2 t2 x3 y3 t3")

init_printing(use_unicode = True)
# Fixed at origin, no rotation
W1 = eye(3)
B1 = T(0, 1.5) * R(pi/2)

# Black origin
B2 = T(x2, y2) * R(t2)
W2 = B2 * T(2.5, 0)

B3 = T(x3, y3) * R(t3)
W3 = B3 * T(2, 0)

# W2, W3 transform relative to W1 should be same as B1, B3 relative to B2 
# Note W1 and B2 are both 3-long. 
# Must try manually for solutions

# Black origin reference (L=3)
BO = [B2, B2 * R(pi)]
# Target A (L=2)
BA = [B1, B1 * R(pi)]
# Target B (L=2)
BB = [B3, B3 * R(pi)]

# Heuristic angles
hs = [0, 0.5235987755982988, 0.7853981633974483, 1.5707963267948966, 2.0943951023931953, 3.1415926535897, 3.665191429188092, 3.9269908169872414, 4.1887902047863905, 4.71238898038469, 5.235987755982989, 5.497787143782138, 5.759586531581287]


params = [(O, A, B, h2, h3) for O in BO for A in BA for B in BB for h2 in hs for h3 in hs]

manager = Manager()
idx_lock = manager.Lock()
idx = manager.Value('i', 0)
with Pool(15) as p:
    p.map(functools.partial(solve_print_permutation, idx_lock, idx), params)

#for (O, A, B) in [(O, A, B) for O in BO for A in BA for B in BB]:
#    for (h2, h3) in [(h2, h3) for h2 in hs for h3 in hs]:
#for O in BO:
#    for A in BA:
#        for B in BB:
#            for (h2, h3) in [(h2, h3) for h2 in hs for h3 in hs]:
#        di1 = solve_print(O, A, B, h2, h3, i)
#        i += di1
#        di2 = solve_print(O, B, A, h2, h3, i)
#        i += di2
#
