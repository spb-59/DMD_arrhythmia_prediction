import os
import logging as lg
from wfdb.io import rdrecord, Record
import multiprocessing as mp
import wfdb.processing as wd
from FeatureExtraction import featureExtract
from denoising import denoise
from wfdb.io import wrsamp
import uuid

'''
This script processes the signals from the ptb dataset to be used for feature extraction. Given the raw signal and comments it gets the condition name abbreviations,
removes noise and stores the resulting signals in a directory. 
'''

recordPath = 'ptb-diagnostic-ecg-database-1.0.0/ptb-diagnostic-ecg-database-1.0.0/'

def preprocess():
    '''
    This function initialises the script. Takes in the records parses the conditions and runs the denoising steps storing the result in a directory
    '''
    
    lg.info('Starting preprocessing...')
    file_paths = []
    with open(recordPath + "RECORDS", 'r') as file:
        file_paths = [line.strip() for line in file.readlines()]
   
    lg.info('Extracted file names from RECORDS')

    paths = [recordPath + path for path in file_paths]
    paths=paths
    
    MI = []
    HC = []
    DR=[]
    BBB=[]
    MH=[]
    VHD=[]
    MY=[]
    MSC=[]
    for path in paths:
        record = rdrecord(path)
        cond = parseConditions(record)

        if cond == 'HC':
            HC.append(path)
            lg.info(f'Classified {path} as Healthy Control')
        elif cond == 'MI':
            MI.append(path)
            lg.info(f'Classified {path} as Myocardial Infarction')
        elif cond == 'DR':
            DR.append(path)
            lg.info(f'Classified {path} as Dysrhythmia')
        elif cond == 'BBB':
            BBB.append(path)
            lg.info(f'Classified {path} as Bundle Branch Block')
        elif cond == 'MH':
            MH.append(path)
            lg.info(f'Classified {path} as Myocardial Hypertrophy')
        elif cond == 'VHD':
            VHD.append(path)
            lg.info(f'Classified {path} as Valvular Heart Disease')
        elif cond == 'MY':
            MY.append(path)
            lg.info(f'Classified {path} as Myocarditis')
        else:
            MSC.append(path)
            lg.info(f'Classified {path} as MSC')

    num_workers = mp.cpu_count()

    conditions = [
        ('MI',MI),
        ('HC',HC),
        ('DR', DR),
        ('BBB', BBB),
        ('MH', MH),
        ('VHD', VHD),
        ('MY', MY),
        ('MSC', MSC)
    ]


    for cond_name, cond_list in conditions:
        lg.info(f'Starting processing with {num_workers} workers for {cond_name}...')
        with mp.Pool(num_workers) as pool:
            pool.starmap(process_record, zip(cond_list, [cond_name] * len(cond_list)))

        lg.info(f'Finished processing for {cond_name}.')

    lg.info('Preprocessing completed.')

def process_record(path, name):
    'This function processes a single record and writes to path given the path to the signal and condition name '
    try:
        lg.info(f'Processing record {path}...')
        record = rdrecord(path)
        record.p_signal = wd.normalize_bound(record.p_signal)
        signal = record.to_dataframe()
        signal.columns = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'vx', 'vy', 'vz']
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
            p_signal=signal.to_numpy(),  # Convert DataFrame to values
            fmt=record.fmt,
            comments=[name],
            write_dir='processedFullPaper'
        )
        lg.info(f'Finished processing record {path}.')
    except Exception as e:
        lg.error(f'ERROR: {e}')  # Fixed logging error message

def parseConditions(record: Record):
    comment = record.comments
    
    for c in comment:
        
        if 'Myocardial infarction' in c:
            return 'MI'
        if 'Healthy control' in c:
            return 'HC'
        if 'Dysrhythmia' in c:
            return 'DR'
        if 'Bundle branch block' in c:
            return 'BBB'
        if 'Myocardial hypertrophy' in c:
            return 'MH'
        if 'Valvular heart disease' in c:
            return 'VHD'
        if 'Myocarditis' in c:
            return 'MY'
    return 'MSC'
        


if __name__ == '__main__':
    preprocess()
    featureExtract()
