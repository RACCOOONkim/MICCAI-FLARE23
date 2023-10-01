import os
import nibabel as nib
import numpy as np
import datetime
from concurrent.futures import ProcessPoolExecutor

def get_files(directory):
    files = sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.nii.gz')])
    return files

def load_data(file):
    if file is not None:
        img = nib.load(file)
        return img.get_fdata()
    else:
        return None

def merge_labels_1800(pseudo_a_file, pseudo_b_file, pseudo_s1_file, pseudo_s2_file, pseudo_s3_file):
    # Load the data
    pseudo_a, pseudo_b, pseudo_s1, pseudo_s2, pseudo_s3 = map(load_data, [pseudo_a_file, pseudo_b_file, pseudo_s1_file, pseudo_s2_file, pseudo_s3_file])

    # Ensure the images have the same shape
    shapes = [data.shape for data in [pseudo_a, pseudo_b, pseudo_s1, pseudo_s2, pseudo_s3]]
    assert all(shape == pseudo_a.shape for shape in shapes), "The shapes of the images are not the same."

    # Extract the bounding box
    nonzero_coords = np.nonzero(pseudo_a)
    min_coords = np.min(nonzero_coords, axis=1)
    max_coords = np.max(nonzero_coords, axis=1) + 1

    output = np.zeros(pseudo_a.shape, dtype=pseudo_a.dtype)
    for i in range(min_coords[0], max_coords[0]):
        for j in range(min_coords[1], max_coords[1]):
            for k in range(min_coords[2], max_coords[2]):
                if pseudo_a[i, j, k] != pseudo_b[i, j, k]:
                    if pseudo_a[i, j, k] == 0:
                        output[i, j, k] = pseudo_b[i, j, k]
                    elif pseudo_b[i, j, k] == 0:
                        output[i, j, k] = pseudo_a[i, j, k]
                    elif pseudo_s1[i, j, k] in {pseudo_a[i, j, k], pseudo_b[i, j, k]}:
                        output[i, j, k] = pseudo_s1[i, j, k]
                    else:
                        output[i, j, k] = 0
                else:
                    output[i, j, k] = pseudo_a[i, j, k]
                if sum([d[i, j, k] == 14 for d in [pseudo_s1, pseudo_s2, pseudo_s3]]) >= 2:
                    output[i, j, k] = 14  
                    
    return nib.Nifti1Image(output, nib.load(pseudo_a_file).affine)

def process_file(files, output_1800_dir):
    pseudo_a_file, pseudo_b_file, pseudo_s1_file, pseudo_s2_file, pseudo_s3_file = files
    output_path = os.path.join(output_1800_dir, os.path.basename(pseudo_a_file))
    if os.path.exists(output_path):
        return
    new_img = merge_labels_1800(pseudo_a_file, pseudo_b_file, pseudo_s1_file, pseudo_s2_file, pseudo_s3_file)
    nib.save(new_img, output_path)

def process_all(pseudo_a_1800_dir, pseudo_b_1800_dir, pseudo_s1_1800_dir, pseudo_s2_1800_dir, pseudo_s3_1800_dir, output_1800_dir, num_workers=40):
    files_list = [get_files(dir) for dir in [pseudo_a_1800_dir, pseudo_b_1800_dir, pseudo_s1_1800_dir, pseudo_s2_1800_dir, pseudo_s3_1800_dir]]

    if not all(len(files_list[0]) == len(files) for files in files_list):
        raise ValueError("Directories do not have the same number of files")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        executor.map(process_file, zip(*files_list), [output_1800_dir] * len(files_list[0]))

process_all('./Pseudo_A_1800', './Pseudo_B_1800', './Pseudo_S1_1800', './Pseudo_S2_1800', './Pseudo_S3_1800', './Output_1800')
