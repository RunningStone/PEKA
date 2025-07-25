"""
in peka, we define following datasets:
    - breast_in_heast :: HEST1k sub database only include breast and homo sapiens
        - breast_xenium_100k :: Xenium platform from breast_in_heast
        - breast_visium_26k :: Visium platform from breast_in_heast
        - breast_visium_st_70k :: ST platform from breast_in_heast

"""
import argparse
args = argparse.ArgumentParser()
args.add_argument("--project_root", type=str, default='./') #project folder
args.add_argument("--database_root", type=str, default='./PEKA/DATA/' ) # processed database storage folder
args.add_argument('--datasets_predefine', type=str, default="peka_breast_datasets.csv" )


print("🤖 running 2_peka_dataset_generator.py")
# 配置项目路径
import sys
import os

if __name__ == "__main__":
    args = args.parse_args()
    proj_path = os.path.abspath(args.project_root)

    sys.path.append(proj_path)
    sys.path.append(proj_path + "/PEKA/")
    # add external models path
    external_module_path = str(proj_path) + "/PEKA/External_models/"
    sys.path.append(external_module_path)
    print(f" ⭐️ external_module_path: {external_module_path}")
    sys.path.append(external_module_path + "/HEST/src/")

    from dotenv import load_dotenv
    env_path = str(proj_path) + "/PEKA/.env"
    load_dotenv(dotenv_path=env_path)
    # get environment variables
    hest_storage_path = os.getenv("HEST1K_STORAGE_PATH")
    data_root = args.database_root
    from peka.Data.hest1k_helper import construct_sub_dataset_index, HEST1K_sub_database,\
                                                extract_patches_from_hest, load_subdataset,\
                                                gene_name_alignment
    from peka.Data.database_helper import check_database_status




    dataset_predefine = args.datasets_predefine

    datasets_info = check_database_status(data_root, hest_storage_path, 
                                          dataset_predefine = dataset_predefine)

    for dataset_info in datasets_info:
        dataset_name = list(dataset_info.keys())[0]
        data_para = list(dataset_info.values())[0]
        index_loc = construct_sub_dataset_index(data_para=data_para, with_explore=True)

        # load hest data for current subdataset
        hest_data, len_hest_data, subdataset_folder = load_subdataset(hest_loc=hest_storage_path,
                                datasets_folder=data_root,
                                subdataset_name=dataset_name,
                                )
        # extract patches
        extract_patches_from_hest(hest_data=hest_data,
                                len_hest_data=len_hest_data,
                                subdataset_folder=data_para.dataset_storage_folder+"/"+data_para.dataset_name,
                                patch_size=data_para.patch_size,
                                pixel_size=data_para.pixel_size,
                                )
        # align gene name
        gene_name_alignment(hest_data=hest_data,
                            subdataset_folder=subdataset_folder,
                            )
