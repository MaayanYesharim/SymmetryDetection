from aspire.basis import FFBBasis2D
from new_ffb_3d import NewFFBBasis3D as FFBBasis3D

from aspire.volume import Volume
from aspire.source.simulation import Simulation
import aspire.volume.volume_synthesis as vsynth
from aspire.utils import Rotation
from aspire.utils.matlab_compat import m_reshape

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy.linalg as sl
import os

import image_ac_auxiliary as aux

# ==================================================================== 
# ===== NOTE: You have to run Blk_matrices_maker.py first
# ====================================================================

try:
    matricesB = np.load(os.path.join(os.path.dirname(__file__), 'saved_dictionary_matricesB.pkl'),allow_pickle='TRUE') #.item ## Use for .npy file
except FileNotFoundError:
    print("================!!!!!!!==================== ")
    print("You have to run Blk_matrices_maker.py first ")
    print("================!!!!!!!==================== ")




# ==============================================
# =========== Default Setting ==================
# ==============================================

img_size = 300
K = img_size//2 #ell_max for 2D basis
ell_max = 10
n_imgs = 1000

# =========== Define Basis ==================
# Fast method for Fourier-Bessel basis in 2D and 3D
vol_basis = FFBBasis3D(size=img_size, ell_max=K, dtype=np.float64)
img_basis = FFBBasis2D(size=img_size, ell_max=K, dtype=np.float64)

# =========== define Rank Table ==================
RANK_TABLE = {
    "Asymmetric":   np.array([1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]),
    "C2":           np.array([1, 1, 3, 3, 5, 5,  7,  7,  9,  9,  11]),
    "C4":           np.array([1, 1, 1, 1, 3, 3,  3,  3,  5,  5,  5]),
    "D2":           np.array([1, 0, 2, 1, 3, 2,  4,  3,  5,  4,  6]),
    "D4":           np.array([1, 0, 1, 0, 2, 1,  2,  1,  3,  2,  3]),
    "T12":          np.array([1, 0, 0, 1, 1, 0,  2,  1,  1,  2,  2]),
    "O24":          np.array([1, 0, 0, 0, 1, 0,  1,  0,  1,  1,  1]),
    "I60":          np.array([1, 0, 0, 0, 0, 0,  1,  0,  0,  0,  1])
}





# ================================================================
# =========== Volume Correlation of Order ell ====================
# ================================================================

def volume_auto_correlation(vol_coeff , ell , vol_basis=vol_basis):
    ''' Correlation of order ell, given certain coefficients. volume-basis needs to be using big "ell_max" '''
    
    k_max_ell = vol_basis.k_max[ell]  # p(ell) number of roots of j_ell
    C_ell = np.zeros((k_max_ell, k_max_ell), dtype=vol_coeff.dtype)
    ind = vol_basis._indices["ells"] == ell  # ind where
    coeff_ell = vol_coeff[:, ind]
    
    for m in range(2 * ell + 1):
        Alm = coeff_ell[:, m * k_max_ell:(m + 1) * k_max_ell]
        Clm = Alm.T.conj() @ Alm
        C_ell += Clm
    v_ell = m_reshape(C_ell, (k_max_ell * k_max_ell, 1))
    
    return C_ell

# =============================================================
# =========== Image Correlation of Order ell ==================
# =============================================================

def image_auto_correlation(covar_coeff,ell, img_basis, vol_basis, matricesB_ell=None):
    img_size = img_basis.n_r
    K = img_size//2     # K in the thesis, same as ffbbasis.ell_max
    
    alpha = aux.vec_alpha(K,ell)
    
    if matricesB_ell is None:
        matricesB_ell = aux.all_Bkl(img_size,ell,basis=img_basis,basis3d=vol_basis)
    elif img_size<300:
        matricesB_ell = aux.all_Bkl(img_size,ell,basis=img_basis,basis3d=vol_basis)
        
        
    K_list =[]
    
        
    start = max(1,ell)
    for k in range(start,K+1): # compute the k'th component: p(l)Xp(l) matrix
        B = matricesB_ell[k]
        C_FB = covar_coeff[2*k-1] 
        M_k =  alpha[k] * B @ C_FB @ B.T
        K_list.append(M_k)

    C_ell = sum(K_list)
    
    return C_ell




def covariance_coefficients(coeff,basis=img_basis):
    ''' This function is for testing if the function of ASPIRE "get_covar" calculates what we wish. '''
    n_img = coeff.shape[0]
    complex_coeff = basis.to_complex(coeff)
    complex_coeff = complex_coeff.asnumpy()    
    dict_cFB = {}      
    ind = 0
    block_size = basis.k_max[ind]
    idx = np.arange(block_size, dtype=int)
    ind += block_size

    coeff_k = complex_coeff[:, idx].T
    cFB_k = np.real(coeff_k @ np.conj(coeff_k).T)/n_img

    dict_cFB[0] = cFB_k

    for k in range(1,basis.ell_max+1):
        
        block_size = basis.k_max[k]
        idx = ind + np.arange(block_size, dtype=int)

        coeff_k = complex_coeff[:, idx].T
        cFB_k = np.real(coeff_k @ np.conj(coeff_k).T)/n_img

        dict_cFB[2*k-1] = cFB_k
        
        ind += block_size
    
    return dict_cFB



def calculateAutocorreltation(vol_coeff, img_covar_coeff, 
                                     vol_basis=vol_basis, img_basis=img_basis,
                                     matricesB=matricesB,
                                     ell_max = ell_max):
    vol_ac = {}
    img_ac = {}
    img_size = img_basis.n_r
    
    for ell in range(1, ell_max+1):
        # Calcualte volume auto-correlation
        vol_ac_ell = volume_auto_correlation(vol_coeff, ell,vol_basis).real
        vol_ac[ell-1] = vol_ac_ell

        if img_size==300:
            matricesB_ell=matricesB[ell]
        else:
            matricesB_ell=None
        # Calculate image auto-correlation
        img_ac_ell = image_auto_correlation(img_covar_coeff, ell, 
                                      img_basis=img_basis, vol_basis=vol_basis,
                                      matricesB_ell=matricesB_ell).real
        img_ac[ell-1] = img_ac_ell
        
        
    
    # Fing linear scaling parameter to fit image-Corr_l and vol-Corr_l:
    vol_ac_summary = []
    img_ac_summary = []

    sumfunc = lambda x: np.median(x)

    for ell in range(1, ell_max+1):
        vol_ac_summary.append(sumfunc(vol_ac[ell-1]))
        img_ac_summary.append(sumfunc(img_ac[ell-1]))
    p = np.polyfit(img_ac_summary, vol_ac_summary, 1)
    
    for ell in range(1,ell_max+1):
        img_ac[ell-1] = p[0] * img_ac[ell-1]
    
    return vol_ac, img_ac, p[0]
