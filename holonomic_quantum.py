import vpython as vp
import numpy as np
import scipy as sc
from scipy.special import binom
from scipy.stats import unitary_group
from itertools import combinations, permutations
from functools import reduce 
from qutip.utilities import clebsch

import jax
import jax.numpy as jp
from jax.config import config
config.update("jax_enable_x64", True)

np.set_printoptions(precision=4, suppress=True)



def basis(d, i):
    v = np.zeros(d)
    v[i] = 1
    return v

def rand_ket(j):
    d = int(2*j + 1)
    v = np.random.randn(d) + 1j*np.random.randn(d) 
    return v/np.linalg.norm(v)

def spin_basis(j, m):
    v = np.zeros(int(2*j+1))
    v[int(j-m)] = 1
    return v

def sigma_plus(j):
    return np.array([[np.sqrt(j*(j+1) - m*n) \
                          if m == n+1 else 0 \
                              for n in np.arange(j, -j-1, -1)]\
                                 for m in np.arange(j, -j-1, -1)])

def sigma_minus(j):
    return sigma_plus(j).conj().T

def sigma_z(j):
    return np.diag(np.array([m for m in np.arange(j, -j-1, -1)]))

def sigma_y(j):
    return (sigma_plus(j) - sigma_minus(j))/(2j)

def sigma_x(j):
    return (sigma_plus(j) + sigma_minus(j))/2




def majorana_polynomial(ket):
    j = (len(ket)-1)/2
    return np.polynomial.Polynomial(\
                    [(-1)**(j-m)*\
                     np.sqrt(binom(2*j, j-m))*\
                     spin_basis(j, m).conj() @ ket\
                         for m in np.arange(-j, j+1)])

def projective_roots(n, poly):
    r = poly.roots()
    return np.concatenate([r, np.repeat(np.inf, n-len(r))])

def majorana_roots(ket):
    return projective_roots(len(ket)-1, majorana_polynomial(ket))

def plane_to_sphere(z):
    if z == np.inf:
        return np.array([0,0,-1])
    else:
        return np.array([2*z.real, 2*z.imag, 1-z.real**2 - z.imag**2])/\
                    (1+z.real**2+z.imag**2)
    
def projective_stars(n, poly):
    return np.array([plane_to_sphere(r) for r in projective_roots(n, poly)])

def majorana_stars(ket):
    return projective_stars(len(ket)-1, majorana_polynomial(ket))





def viz_projective_stars(n, poly):
    scene = vp.canvas(background=vp.color.white)
    vp.sphere(color=vp.color.blue, opacity=0.2)
    [vp.sphere(radius=0.1, pos=vp.vector(*star)) for star in projective_stars(n, poly)]
    
def viz_majorana_stars(ket):
    viz_projective_stars(len(ket)-1, majorana_polynomial(ket))




def random_grassmannian(k, n):
    return unitary_group.rvs(n)[:k]

def standard_grassmannian_form(G):
    return np.linalg.inv(G[:,:2]) @ G

def plucker_coordinate(I, G):
    return np.linalg.det(G[:, I])

def plucker_indices(k, n):
    return list(combinations(list(range(n)), k))

def plucker_coordinates(G):
    return np.array([plucker_coordinate(i, G) for i in plucker_indices(*G.shape)])


def __antisymmetrize__(a, b):
    return np.kron(a, b) - np.kron(b, a)

def antisymmetrize(*V):
    return reduce(__antisymmetrize__, V)




def plucker_basis(k, n):
    return np.array([antisymmetrize(*[basis(n, i) for i in I]) for I in plucker_indices(k, n)])

def plucker_inner_product(v, w):
    return np.linalg.det(v.conj() @ w.T)

def kplane_inner_product(v, w):
    return abs(plucker_inner_product(v,w))/\
                (np.sqrt(plucker_inner_product(v,v))*\
                 np.sqrt(plucker_inner_product(w,w)))

G = random_grassmannian(2, 4)
print(G)
print(G @ G.conj().T)
print(standard_grassmannian_form(G))
c = plucker_coordinates(G);
print(c)
b = plucker_basis(2,4)
print(b)
print(sum([c[i]*b[i] for i in range(len(c))]))
print(antisymmetrize(*G))






def spin_operator_to_plucker(O, k):
    n = len(O)
    P = plucker_basis(k, n)
    return P @ sum([reduce(np.kron, \
                               [O if i == j else np.eye(n) \
                                   for j in range(k)])\
                                       for i in range(k)]) @ P.T/k

def lower_until_zero(j, k, highest=None):
    N = int(binom(int(2*j+1), k))
    highest = highest if type(highest) == np.ndarray else basis(N, 0)
    rungs = [highest]
    L = spin_operator_to_plucker(sigma_minus(j), k)
    while not np.allclose(rungs[-1], np.zeros(N)):
        rungs[-1] = rungs[-1]/np.linalg.norm(rungs[-1])
        rungs.append(L @ rungs[-1])
    return np.array(rungs[:-1])


def spherical_tensor(j, l, m):
    return np.array([[(-1.)**(j-m1-m)*clebsch(j, j, l, m1, -m2, m) \
                   for m1 in np.arange(j, -j-1, -1)]\
                          for m2 in np.arange(j, -j-1, -1)])

def multipole_expansion(O):
    j = (len(O)-1)/2
    return np.array([[(spherical_tensor(j, l, m).conj().T @ O).trace() for m in np.arange(-l, l+1)] for l in np.arange(0, 2*j+1)])





def expect_xyz(ket):
    j = (len(ket)-1)/2
    return np.array([ket.conj() @ sigma_x(j) @ ket,\
                     ket.conj() @ sigma_y(j) @ ket,\
                     ket.conj() @ sigma_z(j) @ ket]).real

def rotate(j, theta, phi):
    return sc.linalg.expm((theta/2) * \
                          (np.exp(-1j*phi)*sigma_plus(j) -\
                           np.exp(1j*phi)*sigma_minus(j)))

def cartesian_to_spherical(xyz):
    x, y, z = xyz
    r = np.linalg.norm(xyz)
    return np.array([r,np.arccos(z/r),np.arctan2(y,x)])


def first_nonzero_in_multipole_expansion(ket):
    j = (len(ket)-1)/2
    MP = multipole_expansion(np.outer(ket, ket.conj()))
    for l in np.arange(0, 2*j+1):
        for m in np.arange(-l, l+1):
            c = MP[int(l)][int(m+l)]
            if not np.isclose(c,0) and m != 0:
                return [c, l, m]
            
def spectator_component(ket):
    if len(ket) == 1:
        return ket[0]
    j = (len(ket)-1)/2
    r, theta, phi = cartesian_to_spherical(expect_xyz(ket))
    ket_ = rotate(j, theta, phi) @ ket
    c, l, m = first_nonzero_in_multipole_expansion(ket_)
    ket__ = sc.linalg.expm(-1j*np.angle(c)*sigma_z(j)/m) @ ket_
    c2 = ket__[np.where(np.logical_not(np.isclose(ket__, 0)))[0][0]]
    return np.sqrt(ket.conj() @ ket)*np.exp(1j*np.angle(c2))

expect_xyz(spin3)
r, theta, phi = cartesian_to_spherical(expect_xyz(spin3)); theta, phi
spin3_ = rotate(3, theta, phi) @ spin3
multipole_expansion(np.outer(spin3_, spin3_.conj()))
spin3_
np.sqrt(spin3.conj() @ spin3)




def check_1anticoherence(G):
    k, n = G.shape
    j = (n-1)/2
    return np.allclose(np.array([G @ sigma_x(j) @ G.conj().T,\
                                 G @ sigma_y(j) @ G.conj().T,\
                                 G @ sigma_z(j) @ G.conj().T]),0)



def qmatrix(f, t):
    k = len(f(0))
    return np.array([[f(0)[i].conj() @ f(t)[j] for j in range(k)] for i in range(k)])

def holonomy(f, t):
    U, D, V = np.linalg.svd(qmatrix(f, t))
    return U @ V

def rot_holonomy(G, R):
    k = G.shape[0]
    G_ = G @ R
    Q = np.array([[G[i].conj() @ G_[j] for j in range(k)] for i in range(k)])
    U, D, V = np.linalg.svd(Q)
    return U @ V




pi_cnot = np.array([(spin_basis(5,5) + 1j*np.sqrt(2)*spin_basis(5,0) + spin_basis(5,-5))/2,\
                    (spin_basis(5,5) - 1j*np.sqrt(2)*spin_basis(5,0) + spin_basis(5,-5))/2,\
                    (np.sqrt(2)*spin_basis(5,3) + 1j*np.sqrt(3)*spin_basis(5,-2) )/np.sqrt(5),\
                    (1j*np.sqrt(3)*spin_basis(5,2) + np.sqrt(2)*spin_basis(5,-3))/np.sqrt(5)])


check_1anticoherence(pi_cnot)
pi_cnot_holonomy1 = rot_holonomy(pi_cnot, sc.linalg.expm(-1j*sigma_x(5)*np.pi)); pi_cnot_holonomy1
np.allclose(pi_cnot_holonomy1 @ pi_cnot, pi_cnot @  sc.linalg.expm(-1j*sigma_x(5)*np.pi))

pi_cnot_holonomy2 = rot_holonomy(pi_cnot, sc.linalg.expm(-1j*sigma_z(5)*(2*np.pi/5))); pi_cnot_holonomy2
np.allclose(pi_cnot_holonomy2 @ pi_cnot, pi_cnot @  sc.linalg.expm(-1j*sigma_z(5)*(2*np.pi/5)))




def sign(lst):
    parity = 1
    for i in range(0,len(lst)-1):
        if lst[i] != i:
            parity *= -1
            mn = min(range(i,len(lst)), key=lst.__getitem__)
            lst[i],lst[mn] = lst[mn],lst[i]
    return parity    

def principle_constellation_polynomial(G):
    k, n = G.shape
    P = [majorana_polynomial(g) for g in G]
    W = np.array([[P[i].deriv(m=j) for j in range(k)] for i in range(k)])
    return sum([sign(list(perm))*np.prod([W[i,p] for i, p in enumerate(perm)]) for perm in permutations(list(range(k)))]) 

def number_of_principal_stars(j, k):
    return 2*np.sum(np.arange(j-k+1, j+1))



def spin_coherent_state(j, z):
    if z == np.inf:
        return spin_coherent_state(j, 0)[::-1]
    return (1+z*z.conjugate())**(-j) * \
            sum([np.sqrt(binom(2*j, j-m))*z**(j-m)*spin_basis(j,m)\
                     for m in np.arange(-j, j+1)])



z = 1.1
spin_coherent_state(3/2, z)



cs = sc.linalg.expm(z*sigma_minus(3/2)) @ spin_basis(3/2,3/2)
cs = cs/np.linalg.norm(cs); cs

def integer(x):
    return np.equal(np.mod(x, 1), 0)

def spin_coherent_representation(ket):
    j = (len(ket)-1)/2
    m = majorana_polynomial(ket)
    def scs_rep(z):
        if z == np.inf:
            return ket[0]
        elif np.isclose(z, 0):
            return ket[-1]
        return ((z.conjugate()/z)/(1+z*z.conjugate()))**j * m(z)
    return scs_rep






j = 1/2
ket = rand_ket(j)
scs_rep1 = spin_coherent_representation(ket)
scs_rep2 = lambda z: spin_coherent_state(j, np.inf if np.isclose(z, 0) else -1/z.conjugate()).conj() @ ket

scs_rep1(1), scs_rep2(1)
scs_rep1(0), scs_rep2(0)
scs_rep1(np.inf), scs_rep2(np.inf)



def coherent_plane(j, k):
    d = int(2*j+1)
    return lambda z: np.eye(d)[:k,:]@sc.linalg.expm(-z*sigma_plus(j))



def find_1anticoherent_subspace(j, k, max_iter=1000):
    d = int(2*j + 1)
    X, Y, Z = sigma_x(j), sigma_y(j), sigma_z(j)

    @jax.jit
    def one_anticoherence(V):
        R = (V[0:d*k] + 1j*V[d*k:]).reshape(k, d)
        return (jp.linalg.norm(R @ X @ R.conj().T) + \
                jp.linalg.norm(R @ Y @ R.conj().T) + \
                jp.linalg.norm(R @ Z @ R.conj().T)).real

    @jax.jit
    def orthogonality(V):
        R = (V[0:d*k] + 1j*V[d*k:]).reshape(k, d)
        return jp.linalg.norm((R @ R.conj().T) - jp.eye(k)).real
    
    for t in range(max_iter):
        try:
            V = np.random.randn(2*d*k)
            result = sc.optimize.minimize(one_anticoherence, V,\
                                          jac=jax.jit(jax.jacrev(one_anticoherence)),\
                                          tol=1e-23,\
                                          constraints=[{"type": "eq",\
                                                        "fun": orthogonality,\
                                                        "jac": jax.jit(jax.jacrev(orthogonality))}],\
                                          options={"disp": True,\
                                                   "maxiter": 5000},
                                          method="trust-constr")
            R = (result.x[0:d*k] + 1j*result.x[d*k:]).reshape(k, d)
            return R
        except:
            continue



j, k = 2, 2
R = find_1anticoherent_subspace(j, k)

R @ R.conj().T
R @ sigma_x(j) @ R.conj().T
R @ sigma_y(j) @ R.conj().T
R @ sigma_z(j) @ R.conj().T
bd = U_2_2 @ plucker_coordinates(R)
j3 = bd[:7]
j1 = bd[7:]
(j3, j1)
viz_majorana_stars(j3)
    






