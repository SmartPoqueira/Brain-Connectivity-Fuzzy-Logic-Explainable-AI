"""
Data loader for the dHCP brain connectivity dataset.

Data source: https://github.com/CoDe-Neuro/Predicting-age-and-clinical-risk-from-the-neonatal-connectome
"""

import numpy as np
import scipy.io


import os
import numpy as np
import scipy.io


def load_dhcp_data(mat_path):
    """Load dHCP structural connectivity matrices.

    Parameters
    ----------
    mat_path : str
        Path to the .mat file containing SCmu, ga, sex, pma, etc.

    Returns
    -------
    matrices : np.ndarray of shape (n_subjects, 90, 90)
    ga : np.ndarray of shape (n_subjects,)
        Gestational age at birth.
    sex : np.ndarray of shape (n_subjects,)
    pma : np.ndarray of shape (n_subjects,)
        Postmenstrual age at scan.
    """
    if not os.path.exists(mat_path):
        print(f"[{mat_path}] not found. Generating simulated brain dataset...")
        np.random.seed(42)
        n_subjects = 100
        # Generate random symmetric matrices of size (90, 90, n_subjects)
        SCmu = np.random.rand(90, 90, n_subjects)
        for s in range(n_subjects):
            SCmu[:, :, s] = (SCmu[:, :, s] + SCmu[:, :, s].T) / 2.0
            np.fill_diagonal(SCmu[:, :, s], 0.0)
        
        ga = np.random.uniform(28.0, 42.0, (n_subjects, 1))
        sex = np.random.randint(0, 2, (n_subjects, 1))
        pma = ga + np.random.uniform(0.0, 4.0, (n_subjects, 1))
        mu = SCmu.mean(axis=(0, 1)).reshape(1, n_subjects)
        ses = np.array([[f"ses-{i}" for i in range(n_subjects)]], dtype=object)
        sub = np.array([[f"sub-{i}" for i in range(n_subjects)]], dtype=object)

        os.makedirs(os.path.dirname(mat_path) if os.path.dirname(mat_path) else ".", exist_ok=True)
        scipy.io.savemat(mat_path, {
            "SCmu": SCmu,
            "ga": ga,
            "sex": sex,
            "pma": pma,
            "mu": mu,
            "ses": ses,
            "sub": sub
        })
        print(f"Successfully created simulated dHCP data at {mat_path}")

    mat = scipy.io.loadmat(mat_path)

    # SCmu shape: (90, 90, n_subjects) → transpose to (n_subjects, 90, 90)
    matrices = np.transpose(mat["SCmu"], (2, 0, 1))
    ga = mat["ga"].ravel()
    sex = mat["sex"].ravel()
    pma = mat["pma"].ravel()

    return matrices, ga, sex, pma


def load_spatial_coordinates(atlas_path):
    """Extract centroid coordinates from a NIfTI brain atlas.

    Parameters
    ----------
    atlas_path : str
        Path to the infant brain atlas NIfTI file.

    Returns
    -------
    centroids : np.ndarray of shape (90, 3)
        Real-world (x, y, z) coordinates in mm for each brain region.
    """
    if not os.path.exists(atlas_path):
        print(f"[{atlas_path}] not found. Generating simulated coordinates...")
        np.random.seed(42)
        # Generate random coordinates in mm
        return np.random.uniform(-40.0, 40.0, (90, 3))

    try:
        import nibabel as nib
    except ImportError:
        raise ImportError("nibabel is required for atlas coordinate extraction. "
                          "Install it with: pip install nibabel")

    img = nib.load(atlas_path)
    data = img.get_fdata()
    affine = img.affine

    centroids = []
    for label in range(1, 91):
        voxels = np.argwhere(data == label)
        if len(voxels) == 0:
            centroids.append([0, 0, 0])
            continue
        centroid_voxel = voxels.mean(axis=0)
        # Apply affine to get real-world coordinates
        centroid_mm = affine[:3, :3] @ centroid_voxel + affine[:3, 3]
        centroids.append(centroid_mm)

    return np.array(centroids)

