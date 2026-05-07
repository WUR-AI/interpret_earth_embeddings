from overlap_utils import get_list_dims

if __name__ == "__main__":
    parent_folder = '/lustre/backup/SHARED/AIN/embed_interpret/data/'
    sample_type = 'lc_stratified_sample'
    modality = 'tessera'
    save_results = True
    dir_save = 'outputs/'
    
    df_dim = get_list_dims(parent_folder=parent_folder, sample_type=sample_type,
                           modality=modality, save_results=save_results, dir_save=dir_save)
    