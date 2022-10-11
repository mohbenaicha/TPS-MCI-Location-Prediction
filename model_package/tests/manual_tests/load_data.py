import pandas as pd
import yaml


with open("config.yaml", 'r') as stream:
    config = yaml.safe_load(stream)
    stream.close()

def load_dataset(*, file_name: str, training: bool = True) -> pd.DataFrame:
    data = pd.read_csv(file_name)
    
    if training:
        inference_features_to_add = config.get('inference_features_to_add')
        for feat in inference_features_to_add:
            data[feat] = 0
    try:
        data = drop_invalid_data(data=data)
        print('dropping invalid complete')
    except Exception as error:
        print(error)
        

    return data[config.get('train_features')+inference_features_to_add+config.get('targets')]

def drop_invalid_data(data,
                      drop_duplicate=True,
                      drop_na_and_zeroes=True,
                      drop_nsa=True):
        
    if drop_duplicate:
        data.drop_duplicates(
            subset=[str(config.get('duplicate_record_key'))], 
            inplace=True
        )

    # MCI data will have recent reports with Lat/Long
    # with coordinate of 0, 0 respectively whereas normally
    # so they should be dropped for training purposes

    if drop_na_and_zeroes:

        data.dropna(subset=config.get('features_na_not_allowed'), inplace=True
        )
        
        trgt_idx = (list(data[data[config.get('targets')[0]] == 0].index) + 
                     list(data[data[config.get('targets')[1]] == 0].index))
        data.drop(axis=0, index = trgt_idx, inplace=True)

    if drop_nsa:
        for feat in config.get('NSA_features'):
            print('Dropping NSA in: ', feat)
            NSA_idx = data[data[feat] == 'NSA'].index
            print('NSA index: ', NSA_idx)
            data.drop(index=NSA_idx, inplace=True)
    
#     data.reset_index(inplace=True)    

    return data