
from collections import Counter
import os
from matplotlib import pyplot as plt
from wfdb.io import rdrecord
from wfdb import Record
import wfdb.processing as wd
import pandas as pd
from wfdb.io import wrsamp
from denoising import denoise
import logging as lg
import time
import multiprocessing as mp

'''
This script is for preprocessing the Large scale arrhythmia Dataset. Given the raw signal and comments it gets the condition name abbreviations,
removes noise and stores the resulting signals in a directory.
'''


basePath="physionet-data/a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0/" # path to signals

SNOMED = pd.read_csv('physionet-data/a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0/ConditionNames_SNOMED-CT.csv') # path to condition map


mapping = pd.Series(SNOMED['Acronym Name'].values, index=SNOMED['Snomed_CT'].astype(str)).to_dict()


def parseConditions(comments):


    for data in comments:
        if data.startswith("Dx:"):
            dx_codes = data.split(": ")[1].split(",")

            mapped = [mapping.get(dx, f"Unknown Dx: {dx}") for dx in dx_codes]
            return mapped



def process_record(record_path, record_name):
    '''
    This function takes in a path to  a signal , parses comments, denoises and writes to directroy.
    '''
    try:
        lg.info(f"Starting processing for {record_name}")
        lg.info(record_path)

        record = rdrecord(record_path)

        lg.info("Record extracted")

        record.p_signal = wd.normalize_bound(record.p_signal)
        signal = record.to_dataframe()
        sf = record.fs
        signal.fillna(0,inplace=True)

        lg.info("Denoising starting")

        signal = denoise(signal, sf)
        comments = parseConditions(record.comments)

        signal = signal.to_numpy()

        lg.info("Denoising finished for signal, creating entry to directory")

        wrsamp(
            record_name=record_name,
            fs=sf,
            units=record.units,
            sig_name=record.sig_name,
            p_signal=signal,
            fmt=record.fmt,
            comments=comments,
            write_dir='processed1'
        )
        lg.info("Entry created, starting next record")

    except Exception as e:
        lg.error(f"Error filtering record {record_name}: {e}")

def Denoise():
    '''
    This function initiates the script and gets all records from the directory, then it parses the comments and denoises the signal, after which 
    it is written to  a directory 
    '''
    start = time.perf_counter()
    lg.info('Starting extraction')
    
    lg.info("Opening Records File")
    
    with open(basePath + "RECORDS", 'r') as file:
        file_paths = [line.strip() for line in file.readlines()]

    lg.info("File paths extracted", file_paths)
    recordPath = []
    names = []

    lg.info('Starting File Name extraction')
    for path in file_paths:
        for root, dirs, files in os.walk(basePath + path):
            for file in files:
                if file.endswith(".mat"):
                    record_name = os.path.splitext(file)[0]
                    record_path = os.path.join(root, record_name)  # Use file here, not record_name

                    names.append(record_name)
                    recordPath.append(record_path)
    
    lg.info(f'File name info extracted, {len(names)} records found')

    lg.info("Starting signal processing")

    num_workers = mp.cpu_count()
    with mp.Pool(num_workers) as pool:
        pool.starmap(process_record, zip(recordPath, names))

    lg.info("Processing Complete")
    end = time.perf_counter()
    lg.info(f'Signal processing complete for {len(names)} records in {end - start:.3f} seconds')

if __name__=="__main__":
    Denoise()
