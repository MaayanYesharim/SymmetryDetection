from aspire.basis import FFBBasis2D
from new_ffb_3d import NewFFBBasis3D as FFBBasis3D
from image_ac_auxiliary import B_dict
import numpy as np
import pickle
import os
import requests

# =============================================================================================
# =========== Calculate the Bkl matrices beforehand for default parameters ====================
# =============================================================================================

# Prameters
img_size = 300
K = int(np.floor(np.pi*img_size//2))

# Generate basis
vol_basis = FFBBasis3D(img_size, ell_max = 10 , dtype=np.float64) 
img_basis = FFBBasis2D( img_size ,K , dtype=np.float64)

# Calculation
B2save =  B_dict(img_size,10,img_basis,vol_basis)

# Save the dictionary
with open('saved_dictionary_matricesB.pkl', 'wb') as fp:
    pickle.dump(B2save, fp)
    print('dictionary of matrices_B_kl saved successfully to file')

# =============================================================================================
# =========== Download Real Volumes From https://rcsb.org  ====================================
# =============================================================================================

folder_name = 'vols'

# Create the folder if it does not exist
if not os.path.exists(folder_name):
    os.makedirs(folder_name)
    print(f"Folder '{folder_name}' created.")

urls = ['https://files.rcsb.org/pub/emdb/structures/EMD-0781/map/emd_0781.map.gz',
        'https://files.rcsb.org/pub/emdb/structures/EMD-9944/map/emd_9944.map.gz',
        'https://files.rcsb.org/pub/emdb/structures/EMD-30067/map/emd_30067.map.gz',
        'https://files.rcsb.org/pub/emdb/structures/EMD-20414/map/emd_20414.map.gz',
        'https://files.rcsb.org/pub/emdb/structures/EMD-20687/map/emd_20687.map.gz',
        'https://files.rcsb.org/pub/emdb/structures/EMD-0742/map/emd_0742.map.gz',
        'https://files.rcsb.org/pub/emdb/structures/EMD-0108/map/emd_0108.map.gz',
        'https://files.rcsb.org/pub/emdb/structures/EMD-14194/map/emd_14194.map.gz']

# Download each file
for url in urls:
    file_name = os.path.join(folder_name, os.path.basename(url))
    response = requests.get(url)
    with open(file_name, "wb") as f:
        f.write(response.content)
    print(f"Downloaded: {file_name}")

print("All files downloaded.")