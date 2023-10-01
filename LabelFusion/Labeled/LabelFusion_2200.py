import os
import nibabel as nib
import numpy as np
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
    
def merge_labels_2200(pseudo_a_file, pseudo_b_file, partially_file):
    # Load the data
    pseudo_a, pseudo_b, partially = map(load_data, [pseudo_a_file, pseudo_b_file, partially_file])

    # Ensure the images have the same shape
    assert pseudo_a.shape == pseudo_b.shape == partially.shape, "The shapes of the images are not the same."

    # Extract the bounding box
    nonzero_coords = np.nonzero(pseudo_a)
    min_coords = np.min(nonzero_coords, axis=1)
    max_coords = np.max(nonzero_coords, axis=1) + 1

    output = np.zeros(pseudo_a.shape, dtype=pseudo_a.dtype)
    # Work only within the bounding box of non-zero coordinates
    for i in range(min_coords[0], max_coords[0]):
        for j in range(min_coords[1], max_coords[1]):
            for k in range(min_coords[2], max_coords[2]):
                if pseudo_a[i, j, k] != pseudo_b[i, j, k]:
                    if pseudo_a[i, j, k] == 0:
                        output[i, j, k] = pseudo_b[i, j, k]
                    elif pseudo_b[i, j, k] == 0:
                        output[i, j, k] = pseudo_a[i, j, k]
                    elif partially[i, j, k] in {pseudo_a[i, j, k], pseudo_b[i, j, k]}:
                        output[i, j, k] = partially[i, j, k]
                    else:
                        output[i, j, k] = 0
                else:
                    output[i, j, k] = pseudo_a[i, j, k]
                if partially[i, j, k] == 14:
                    output[i, j, k] = 14

    return nib.Nifti1Image(output, nib.load(pseudo_a_file).affine)

def process_file(files, output_2200_dir):
    pseudo_a_file, pseudo_b_file, partially_file = files
    output_path = os.path.join(output_2200_dir, os.path.basename(pseudo_a_file))
    if os.path.exists(output_path):
        return
    new_img = merge_labels_2200(pseudo_a_file, pseudo_b_file, partially_file)
    nib.save(new_img, output_path)

def process_all(pseudo_a_2200_dir, pseudo_b_2200_dir, partially_2200_dir, output_2200_dir, num_workers=40):
    files_list = [get_files(dir) for dir in [pseudo_a_2200_dir, pseudo_b_2200_dir, partially_2200_dir]]

    if not all(len(files_list[0]) == len(files) for files in files_list):
        raise ValueError("Directories do not have the same number of files")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        executor.map(process_file, zip(*files_list), [output_2200_dir] * len(files_list[0]))

process_all('./Pseudo_A_2200', './Pseudo_B_2200', './Partially_2200', './Output_2200')
