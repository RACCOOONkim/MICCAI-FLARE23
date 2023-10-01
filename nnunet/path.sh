mkdir -p $PWD/dataset/nnUNet_raw_data_base/nnUNet_raw_data
mkdir -p $PWD/dataset/nnUNet_preprocessed

mkdir -p $PWD/LabelFusion/Labeled/{Pseudo_A_2200,Pseudo_B_2200,Partial_2200,Output_2200}
mkdir -p $PWD/LabelFusion/Unlabeled/{Pseudo_A_1800,Pseudo_B_1800,Pseudo_S1_1800,Pseudo_S2_1800,Pseudo_S3_1800,Output_1800}


export RESULTS_FOLDER="$PWD/dataset/nnUNet_results"
export nnUNet_raw_data_base="$PWD/dataset/nnUNet_raw_data_base"
export nnUNet_preprocessed="$PWD/dataset/nnUNet_preprocessed"