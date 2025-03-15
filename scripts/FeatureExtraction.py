
import os
import logging as lg
import numpy as np
from wfdb.io import rdrecord
from wfdb import Record
import pydmd as dmd
import multiprocessing as mp

'''
This script is for performing feature extraction using DMD. This assumes that the data has been preprocessed and is in the specified format. 
This assumes preprocessing has converted dataset comments into abbreviation for specific conditions.
'''


path='processedFullPaper' #this is the directory of the processed signals
writeDir='featuresFullPaper' # this is the directory to write the features to
    

def getHeader(M_s, P_s, M_u, P_u):
    '''Get the header for the csv file'''
    m_s_str = ', '.join([f'M_s{i+1}' for i in range(len(M_s))])
    p_s_str = ', '.join([f'P_s{i+1}' for i in range(len(P_s))])
    m_u_str = ', '.join([f'M_u{i+1}' for i in range(len(M_u))])
    p_u_str = ', '.join([f'P_u{i+1}' for i in range(len(P_u))])
    return f"R_N,R_L,R_M,R_P,Lam_min,Lam_max,{m_s_str},{p_s_str},{m_u_str},{p_u_str}\n"



def format_array(array):
    """Convert a numpy array or list to a space-separated string."""
    return  ', '.join(map(str, array))
  

def format_csv_row(R_N, R_L,R_M, R_P, Lam_min, Lam_max, M_s, P_s, M_u, P_u):
    """Format the row into a CSV-compatible string."""
    return (
        f"{R_N},{R_L},{R_M},{R_P},{Lam_min},{Lam_max},"+f"{format_array(M_s)},"+f"{format_array(P_s)},"+f"{format_array(M_u)},"+f"{format_array(P_u)}\n"
    )

def featureExtract():
    """
    This function initiates the script gets all paths of processed records , and processes them for feature extraction 
    """

    recordPath=[]
    names=[]

    lg.info('Starting File Name extraction')

    for root, dirs, files in os.walk(path):

        for file in files:
            if file.endswith(".dat"):

                record_name = os.path.splitext(file)[0]
                

                record_path = os.path.join(root, record_name)

                names.append(record_name)

                recordPath.append(record_path)
    num_workers = 4
    with mp.Pool(num_workers) as pool:
        pool.map(process_record, recordPath)

def process_record(record_path):
    '''
    This function takes in a path to  a single record , extracts features using DMD and writes the result to  a file.
    '''
    try:
        recordSig = rdrecord(record_path)
        lg.info("Starting DMD for record: %s", record_path)


        signals=[recordSig.p_signal[i:i + 4000,:12] for i in range(0, len(recordSig.p_signal), 2000)]
        for signal in signals:
         features, header = extract(signal)
        
         writeFile(features, header,recordSig.comments)
        
        lg.info("Finished extracting for record: %s", record_path)
    except KeyError as e:
        lg.error("Error key not found %s",e)
    except Exception as e:
        lg.error("Error processing record %s: %s", record_path, e)


def extract(signal:np.ndarray):

    '''
    This function extracts the DMD features given  a signal
    '''
    # Get the signals augumented
    signal=AugMat(signal.T,200)


    #fit the DMD model
    DMD=dmd.DMD()
    DMD.fit(signal)

    #get the eigenvalue and vectors
    eigs=DMD.eigs
    modes=DMD.modes

    #restack the modes to match the 12 leads
    restacked= modes.reshape(12, 200, -1).mean(axis=1)

    #get lambda U for unstable S for stable
    Lambda_ind_u = np.where(np.abs(eigs) > 1)
    Lambda_ind_s = np.where(np.abs(eigs) < 1)

    Lambda_u = eigs[Lambda_ind_u] #unstable eigen values
    Lambda_s = eigs[Lambda_ind_s] #stable eigen values

    # Get the eigenvectors off the eigenvalue indexes
    Pho_u = restacked[:,Lambda_ind_u].reshape((12,Lambda_u.shape[0])) #unstable modes
    Pho_s = restacked[:,Lambda_ind_s].reshape((12,Lambda_s.shape[0])) #stable modes

    #number of Stable and unstable modes
    numS=Lambda_s.shape[0]
    numU=Lambda_u.shape[0]

    # unstable to stable DM ratio
    R_N=(numU)/(numU+numS)

    #unstable lambda to all
    R_L=np.sum(np.abs(Lambda_u))/(np.sum(np.abs(Lambda_s))+np.sum(np.abs(Lambda_u)))

    R_M = np.sum(np.sum(np.abs(Pho_u), axis = 1))/(np.sum(np.sum(np.abs(Pho_u), axis = 1)) + np.sum(np.sum(np.abs(Pho_s), axis = 1)))
    R_P = np.sum(np.sum(np.angle(Pho_u), axis = 1))/(np.sum(np.sum(np.angle(Pho_u), axis = 1)) +  np.sum(np.sum(np.angle(Pho_s), axis = 1)))


    Lam_min = np.min(np.abs(Lambda_s),initial = -1 )
    Lam_max = np.max(np.abs(Lambda_u), initial = -1 )

    #for unstble modes
    M_u = np.mean(np.abs(Pho_u), axis=1)
    P_u = np.mean(np.angle(Pho_u-np.angle(Pho_u[0])), axis=1)


    #for stable modes
    M_s = np.mean(np.abs(Pho_s), axis=1)
    P_s = np.mean(np.angle(Pho_s-np.angle(Pho_s[0])), axis=1)



    return  format_csv_row(R_N,R_L, R_M, R_P, Lam_min, Lam_max, M_s, P_s, M_u, P_u),getHeader(M_s,P_s,M_u,P_u)



def writeFile(features,header,comments):
    '''
    This function writes the extracted features to the specified directory given the header and condition name

    '''
    for comment in comments:
        file_path = os.path.join(writeDir, f"{comment}.csv")
        

        
        # Determine if the file exists
        file_exists = os.path.exists(file_path)
        
        # Open the file in append mode
        with open(file_path, "a") as f:
            if not file_exists:
                # Write header if the file does not exist
                f.write(header)
            
            # Write features
            f.write(features)


def AugMat(signal: np.ndarray, h: int):
    '''
    This function augments the matrix for applying DMD by a augmentation factor of input h
    '''
    n, m = signal.shape
    aug = []
    for i in range(n):
        for x in range(h):
            row = signal[i][x:m-h+x]
            aug.append(row)
    return np.vstack(aug)

if __name__=="__main__":
    print(format_array([1,2,3]))
    







    





    