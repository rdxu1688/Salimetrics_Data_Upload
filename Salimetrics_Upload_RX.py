import pandas as pd
import numpy as np
import os
import re
import json
import redcap as rc
import requests
import glob

def is_data(row):
    """
    Check if the first element of the row matches the Salimetrics ID pattern (e.g., '12345-6789').

    Parameters:
    row (list) a of the csv file

    Returns:
    bool: True if matches pattern False otherwise
    """
    pattern = r'^\d{5}-\d{4}$'
    if len(row) > 0 and re.match(pattern, row[0]):
        return True
    else:
        return False

# Method reads file as input to clean it and return as a dataframe
def read_file(filepath):
    """
    Reads a hormone data file, cleans it, and returns the data as a dataframe

    Parameters:
    filepath (str): The path to the input file.

    Returns:
    pd.DataFrame: A DataFrame containing the cleaned data.
    """
    #get the file
    file_name = os.path.basename(filepath)
    #print(file_name)
    
    title_name = "NA"
    if "DHEA" in file_name:
        title_name = "dhea"
    elif "ERT" in file_name:
        title_name = "ert"
    elif "HSE" in file_name:
        title_name = "hse"
    if title_name == "NA":
        raise ValueError("Not valid file")
        
    titles = ["hormone_sal_bc_y", 
              "id_redcap", 
              "hormone_scr_"+title_name+"_rep1", 
              "hormone_scr_"+title_name+"_rep1_qns",
              "hormone_scr_"+title_name+"_rep1_nd",
              "hormone_scr_"+title_name+"_rep1_ll",
              "hormone_scr_"+title_name+"_rep2",
              "hormone_scr_"+title_name+"_rep2_qns",
              "hormone_scr_"+title_name+"_rep2_nd",
              "hormone_scr_"+title_name+"_rep2_ll",
              "hormone_scr_"+title_name+"_mean",
              "hormone_saliva_salimatric_scores_daic_use_only_complete",
              "comments"]
    data_df = pd.DataFrame(columns=titles)
    
    with open(file_name, 'r') as file:
        # Skip first 6 to make it slightly faster haha
        for i in range(6):
            next(file)

        # Loop through each row
        for row in file:
            # Split the row into columns
            temp = row.strip().split(',')
            
            # Check row is data
            if is_data(temp):
                # Remove the Salimetrics ID
                temp.pop(0) 
                
                if len(temp) < 6:
                    temp.append('')
                
                qns_rep1 = 1 if "qns" in temp[2].lower() else 0
                qns_rep2 = 1 if "qns" in temp[3].lower() else 0
                nd_rep1 = 1 if "nd" in temp[2].lower() else 0
                nd_rep2 = 1 if "nd" in temp[3].lower() else 0
                bls_rep1 = 1 if '*' in temp[2] else 0
                bls_rep2 = 1 if '*' in temp[3] else 0
                
                if(bls_rep1 == 1):
                    temp[2] = temp[2][1:]
                if(bls_rep2 == 1):
                    temp[3] = temp[3][1:]
                if(qns_rep1 or nd_rep1):
                    temp[2] = ""
                if(qns_rep2 or nd_rep2):
                    temp[3] = ""
                temp[4] = temp[4][1:] if (len(temp[4]) > 0 and temp[4][0] == "*") else temp[4]

                new_row = {"hormone_sal_bc_y" : temp[0],
                           "id_redcap" : temp[1],
                           "hormone_scr_"+title_name+"_rep1" : temp[2].replace(">", ''),
                           "hormone_scr_"+title_name+"_rep1_qns" : qns_rep1,
                           "hormone_scr_"+title_name+"_rep1_nd" : nd_rep1,
                           "hormone_scr_"+title_name+"_rep1_ll" : bls_rep1,
                           "hormone_scr_"+title_name+"_rep2" : temp[3].replace(">", ''),
                           "hormone_scr_"+title_name+"_rep2_qns" : qns_rep2,
                           "hormone_scr_"+title_name+"_rep2_nd" : nd_rep2,
                           "hormone_scr_"+title_name+"_rep2_ll" : bls_rep2,
                           "hormone_scr_"+title_name+"_mean" : temp[4].replace(">", ''),
                           "hormone_saliva_salimatric_scores_daic_use_only_complete" : "2",
                           "comments" : temp[5]}
                
                
                if new_row['id_redcap'] is None or new_row['id_redcap'] == '':
                    print(temp[1] + "No pGUID")
                new_row_df = pd.DataFrame([new_row], columns=titles)
                data_df = pd.concat([data_df, new_row_df], ignore_index=True)

    # Remove WHITESPACE CHARACTERS and " in barcode and pGUID column
    data_df['hormone_sal_bc_y'] = data_df['hormone_sal_bc_y'].replace(r'\s', '', regex=True)
    data_df['id_redcap'] = data_df['id_redcap'].replace(r'\s', '', regex=True)
    data_df['hormone_sal_bc_y'] = data_df['hormone_sal_bc_y'].replace('"', '', regex=True)
    data_df['id_redcap'] = data_df['id_redcap'].replace('"', '', regex=True)
    return data_df

def add_event(df):
    """
    Add redcap_event_name to dataframe based on pGUID column

    Parameters:
    dataframe lacking redcap_event_name

    Returns:
    dataframe with redcap_event_name
    """
    conditions = [df['hormone_sal_bc_y'].str.contains('Y07'),
                  df['hormone_sal_bc_y'].str.contains('Y06'),
                  df['hormone_sal_bc_y'].str.contains('Y05'),
                  df['hormone_sal_bc_y'].str.contains('Y04'),]
    choices = ['7_year_follow_up_y_arm_1',
               '6_year_follow_up_y_arm_1',
               '5_year_follow_up_y_arm_1',
               '4_year_follow_up_y_arm_1',]
    df['redcap_event_name'] = np.select(conditions, choices, default='')

    return df

# initialize access token
site = 'DAIRC'
REDCAP_URL = 'https://abcd-rc.ucsd.edu/redcap/api/'
CURRENT_DIR = os.getcwd()
with open(os.path.join(CURRENT_DIR, './secure/tokens.json')) as data_file:
    redcap_tokens = json.load(data_file)
    redcap_tokens = pd.DataFrame.from_dict(redcap_tokens, orient='index', columns=['token'])
redcap_tokens

# Get device list from main Redcap project
try:
    rc_token = redcap_tokens.loc[site, 'token']
except KeyError:
    log.error('%s: Redcap token ID is not available!', site)
    #continue

# initialize access to REDCap
rc_api = rc.Project(REDCAP_URL, rc_token)

# get data from REDCap (Y_Pubertal Hormone Saliva)
rc_barcode_fields = ['enroll_total', 'asnt_timestamp', 'hormone_sal_end_y','hormone_sal_bc_y', 'hormone_sal_sex', 
                     'hormone_saliva_salimatric_scores_daic_use_only_complete']
REDCAP_EVENT = ['baseline_year_1_arm_1','1_year_follow_up_y_arm_1','2_year_follow_up_y_arm_1', '3_year_follow_up_y_arm_1',
               '4_year_follow_up_y_arm_1','5_year_follow_up_y_arm_1', '6_year_follow_up_y_arm_1']

# Query redcap data 
# Uncomment forms for DHEA/ERT/HSE numerical values
rc_devices = rc_api.export_records(
    fields = rc_barcode_fields + [rc_api.def_field],
    # forms = ['hormone_saliva_salimatric_scores_daic_use_only'], 
    # records = ['NDAR_INVT185D4UD'],  # ['4_year_follow_up_y_arm_1','5_year_follow_up_y_arm_1', '6_year_follow_up_y_arm_1',
    events = ['4_year_follow_up_y_arm_1','5_year_follow_up_y_arm_1', '6_year_follow_up_y_arm_1', '7_year_follow_up_y_arm_1'],
    # REDCAP_EVENT,
    export_data_access_groups=True,
    format ='df')


# set up lists and columns to drop for later
filtered_rows = []
path_pattern = CURRENT_DIR+'/*.csv'
drop_col = ['redcap_repeat_instrument', 'redcap_repeat_instance', 'redcap_data_access_group', 'asnt_timestamp',
            'hormone_sal_sex', 'hormone_sal_end_y', 'enroll_total___1']
drop_col2 = ['id_redcap_x', 'hormone_sal_bc_y', 'hormone_saliva_salimatric_scores_daic_use_only_complete_y','comments']
drop_col3 = ['hormone_sal_bc_y_x', 'hormone_saliva_salimatric_scores_daic_use_only_complete_y', 'hormone_sal_bc_y_y','comments']

# process and clean the queried data 
rc_devices = rc_devices.reset_index()
rc_devices = rc_devices[(rc_devices['hormone_sal_bc_y'].notna()) & (rc_devices['hormone_sal_bc_y'].str.match(r'Y0[1-7]-PS\d{2}-\d{3}'))]
rc_devices = rc_devices[(rc_devices['id_redcap'].notna())]
rc_devices_clean_barcode = rc_devices.drop_duplicates(subset='hormone_sal_bc_y', keep=False)
rc_devices_clean_barcode = rc_devices_clean_barcode.drop(columns = drop_col)

# loop through all csv data files in current directory to process and upload to redcap
for filepath in glob.glob(path_pattern):
    tempname = filepath
    print('reading ' + path_pattern)
    # print("Processing:", filepath)
    
    try:
        monthlydata = read_file(filepath)
    except ValueError as e:
        print("Error processing file:", filepath)
        print(e)
        
    # match csv data with queried data to ensure a match
    barcode_upload = pd.merge(monthlydata, rc_devices_clean_barcode, on='hormone_sal_bc_y', how='inner')
    barcode_bad = monthlydata[~monthlydata['hormone_sal_bc_y'].isin(rc_devices_clean_barcode['hormone_sal_bc_y'])]
    barcode_bad = add_event(barcode_bad)
    pGUID_upload = pd.merge(barcode_bad, rc_devices_clean_barcode, on=['id_redcap', 'redcap_event_name'], how='inner')
    
    # drop excess columns
    barcode_upload = barcode_upload.drop(columns = drop_col2)
    barcode_upload.rename(columns={'id_redcap_y': 'id_redcap', 
                                    'hormone_saliva_salimatric_scores_daic_use_only_complete_x': 'hormone_saliva_salimatric_scores_daic_use_only_complete'}, inplace=True)
    pGUID_upload = pGUID_upload.drop(columns = drop_col3)
    pGUID_upload.rename(columns={'hormone_saliva_salimatric_scores_daic_use_only_complete_x': 'hormone_saliva_salimatric_scores_daic_use_only_complete'}, inplace=True)
    
    # combine
    upload = pd.concat([barcode_upload, pGUID_upload])
    
    # upload row by row and save unuploaded to filtered_rows
    for index, row in upload.iterrows():
        # convert to dataframe
        upload_row_df = pd.DataFrame(row).T 
        upload_row_df.set_index(['id_redcap','redcap_event_name'],  inplace=True)
        
        try:
            out = rc_api.import_records(upload_row_df, overwrite='overwrite', return_content='ids')
            print('%s: Successfully updated Redcap ksads records for %s', site, out)
        except requests.RequestException as e:
            print('%s: Error occurred during upload of %d records. exception is %s', site, upload_row_df.shape[0], e)
            filtered_rows.append(upload_row_df)

# uncomment if needed to record unuploaded data
# filtered_rows.to_csv('unmapped.csv', index=False)