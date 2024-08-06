from aspire.basis.basis_utils import lgwt
from aspire.utils.matlab_compat import m_reshape
from aspire.basis.basis_utils import sph_bessel

import numpy as np
import scipy.linalg as sl
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.special import legendre,chebyc
from tqdm import tqdm


# ===================================================
# ============ [a_{1,l},a_{2,l},...,a_{K,l}] ========
# ===================================================

def vec_alpha(K,ell):
    ''' ell: the order of the Auto-Correlation to calculate. The maximum should be around 10.
    K: the image size / 2, we are making alpha for any 1<k<K. '''
    
    s = 1
#     s = 2*ell+1
    alpha = np.zeros((K+1,))
    P = legendre(ell)
    
    for k in range(0,K+1):
        f = lambda phi: np.cos(k*phi) * s * P(np.cos(phi)) * np.sin(phi)

        alpha_k = integrate.fixed_quad( f , 0 , np.pi,n=K+10)[0]
        if np.abs(alpha_k)>1e-8:
            alpha[k] = alpha_k
    return alpha

# ==========================================================================
# ============ Help function for aspire.basis.FFBBasis2D.get_radial ========
# ==========================================================================

def find_idx_J(basis):
    ''' Get a list of indexes of the k-th Bessel function for any k < ell_max as it generates in ASPIRE
    return:
    a list with the indexes of the k-th Bessel function in place k '''
    
    img_size = basis.n_r
    ell_max = basis.ell_max    # same as K = img_size//2
    
    List = []
    ind = 0
    idx = ind + np.arange(basis.k_max[0], dtype=int)
    List.append(idx)
    ind = ind + np.size(idx)

    for ell in range(1,ell_max + 1):
        k_max_ell = basis.k_max[ell]
        idx = ind + np.arange(k_max_ell, dtype=int)
        List.append(idx)
        ind = ind + np.size(idx)
    return List

# =======================================================================================================
# ========== This function is the 3D equivalent of aspire.basis.FFBBasis2D.get_radial ===================
# ====== generates [j_ls(r)] for l,s, and r for the default range given by the basis (basis3d)===========
# =======================================================================================================

def Get_Radial(img_size,basis3d):
    ''' Get the spherical Bessel function with aspire'''

    ell_max = basis3d.ell_max
        
    r, wt_r = lgwt(img_size, 0.0, basis3d.kcut, dtype=basis3d.dtype)
    r = m_reshape(r, (img_size, 1))


    radial = np.zeros(
        shape=(img_size, np.max(basis3d.k_max), ell_max + 1), dtype=basis3d.dtype
    )

    for ell in range(0, ell_max + 1):
        k_max_ell = basis3d.k_max[ell]
        rmat = r * basis3d.r0[ell][0:k_max_ell].T / basis3d.kcut
        radial_ell = np.zeros_like(rmat)
        for ik in range(0, k_max_ell):
            radial_ell[:, ik] = sph_bessel(ell, rmat[:, ik])
        nrm = np.abs(sph_bessel(ell + 1, basis3d.r0[ell][0:k_max_ell].T) / 4)
        radial_ell = radial_ell / nrm
        radial[:, 0:k_max_ell, ell] = radial_ell
    return radial



# ===============================================================================================
# ============ B_kl is a p(k) by p(l) matrix holds beta_kq,ls in the q,s entery. ================
# ================== beta_kq,ls = integral over r { Jkq(r) * jls(r) * r**2 } ====================
# ===============================================================================================

def B_kl( k , ell, basis, basis3d):
    ''' Matrix size p(k) X p(ell),
    depends on the Bessel functions and spherical Bessel functions- therefore depends on the two bases'''

    img_size = basis.n_r

    radial = basis._precomp["radial"] # we have samples of the wanted Bessel normalized function in n_r (=64) nodes.
    sph_radial = Get_Radial(img_size,basis3d) #   jls = sph_radial[:, s-1, l]

    r, w = lgwt(img_size, 0.0, basis.kcut, dtype=basis.dtype)
            
    idx = find_idx_J(basis)

    p_k = basis.k_max[k]
    p_l = basis3d.k_max[ell]
    B = np.zeros((p_k,p_l))

    for q in range(1,p_k):
        for s in range(1,p_l):
            Jkq = radial[idx[k]][q-1]
            jls = sph_radial[:, s-1, ell ]
            beta = np.sum( Jkq * jls * r**2 * w , dtype='float64')
            B[q-1,s-1] = beta
    return B.T

def all_Bkl(img_size,ell,basis,basis3d,printt=True):
    ''' Bkl for given ell and all k in [ell,K] were K is the image size/2
        Note: Bkl is of the size p(l) X p(k)'''
    Bkl = {}
    K = img_size//2
#     if printt==True:
#         print(f'In process: making Bkl for l = {ell} and any k')
    for k in tqdm(range(ell,K+1)):
    # for k in range(ell,K+1):
        Bkl[k] = B_kl( k , ell,basis,basis3d)
    return Bkl


# ==========================================================================================
# ============ All the needed B_kl for a data with image_size as given =====================
# ==========================================================================================

def B_dict(img_size,ell_max,basis,basis3d):
    B = {}
    for ell in range(1,ell_max+1):
        B[ell] = all_Bkl(img_size,ell,basis,basis3d)
    return B



