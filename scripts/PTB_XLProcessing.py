import ast
import os
import logging as lg
import numpy as np
import pandas as pd
import wfdb
from wfdb.io import rdrecord, Record
import multiprocessing as mp
import wfdb.processing as wd
from FeatureExtraction import featureExtract
from denoising import denoise
from wfdb.io import wrsamp
import uuid

'''
This script processes the signals from the ptb-xl dataset to be used for feature extraction. Given the raw signal and comments it gets the condition name abbreviations,
removes noise and stores the resulting signals in a directory. 
'''


path="ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/" #path to dataset

def load_raw_data(df, sampling_rate, path):
        if sampling_rate == 100:
            data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
        else:
            data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
        data = np.array([signal for signal, meta in data])
        return data

def preprocess():
    '''
    This function initialises the script. Takes in the records parses the conditions and runs the denoising steps storing the result in a directory
    '''
    
    lg.info('Starting preprocessing...')
    Y = pd.read_csv(path + 'ptbxl_database.csv', index_col='ecg_id')
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

    HC=[]
    AFIB=[]
    MI=[]
    for index, row in Y.iterrows():
        for key, val in row.scp_codes.items():  
            if key == 'NORM' and val == 100:
                HC.append(path+row.filename_hr)  
            elif key == 'AFIB' and val == 100:
                AFIB.append(path+row.filename_hr)
            elif key in['IMI','ASMI','ILMI',"AMI",'ALMI','LMI','IPLMI','IPMI','PMI'] and val == 100:
                MI.append(path+row.filename_hr)
    num_workers = mp.cpu_count()

    lg.info(f'Starting processing with {num_workers} workers for MI...')
    with mp.Pool(num_workers) as pool:
        pool.starmap(process_record, zip(MI[:50], ['MI'] * 50))
        lg.info(f'Starting processing with {num_workers} workers for MI...')
    with mp.Pool(num_workers) as pool:
        pool.starmap(process_record, zip(HC[:50], ['HC'] * 50))
        lg.info(f'Starting processing with {num_workers} workers for MI...')
    with mp.Pool(num_workers) as pool:
        pool.starmap(process_record, zip(AFIB[:50], ['AFIB'] * 50))



    lg.info('Preprocessing completed.')
        







def process_record(path, name):
    'This function processes a single record and writes to path given the path to the signal and condition name '
    try:
        lg.info(f'Processing record {path}...')
        record = rdrecord(path)
        record.p_signal = wd.normalize_bound(record.p_signal)
        signal = record.to_dataframe()
        signal.columns = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        sf = record.fs
        signal.fillna(0, inplace=True)
        signal = denoise(signal, sf)
        
        output_name = str(uuid.uuid4())
        lg.info(f'Writing processed signal to file {output_name}...')
        
        wrsamp(
            record_name=output_name,
            fs=sf,
            units=record.units,
            sig_name=record.sig_name,
            p_signal=signal.to_numpy(),  
            fmt=record.fmt,
            comments=[name],
            write_dir='processed7'
        )
        lg.info(f'Finished processing record {path}.')
    except Exception as e:
        lg.error(f'ERROR: {e}')  

if __name__ == '__main__':
    featureExtract()
