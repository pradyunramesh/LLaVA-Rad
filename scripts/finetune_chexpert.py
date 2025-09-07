import pandas as pd 
import json
import sys
import os

from ChexpertPreprocessing import CheXpertPreprocessing

##Utility Methods
def create_labels_dict(row):
    """Convert CheXpert labels to dictionary for a single row"""
    return {label: row[label] for label in CHEXPERT_LABELS if label in row}

def create_conversations_list(row):
    conversations = []
    output = ""
    if row['section_findings'] and row['section_findings'] != 'nan':
        output = output + str(row['section_findings']).replace("\n", "").strip() + " "
    if row['section_impression'] and row['section_impression'] != 'nan':
        output = output + str(row['section_impression']).replace("\n", "").strip()
    dict_findings = {"from": "gpt", "value": output}
    if row['section_indication'] is not None:
        reason = row['section_indication'].replace('\n', '')
        dict_reason = {"from": "human", "value": f"<image>\nProvide a description of the findings in the radiology image given the following indication: {reason}"}
    else:
        dict_reason = {"from": "human", "value": "<image>\nProvide a description of the findings in the radiology image."}
    conversations.append(dict_reason)
    conversations.append(dict_findings)
    return conversations

#Set paths and constants
input_json_file = "/home/vj2292@mc.cumc.columbia.edu/radiologist/final.jsonl"
path = '/data/raw_data/chexpert/chexpertplus/df_chexpert_plus_240401.csv'
preprocessed_path = "/home/pr2762@mc.cumc.columbia.edu/CXR_Attribution/splits/preprocessed_chexpert.csv"
REQUIRED_COLUMNS = {'Examination', 'Indication', 'Findings', 'Impression'}
CHEXPERT_LABELS = ['No Finding','Enlarged Cardiomediastinum','Cardiomegaly','Lung Opacity',
    'Lung Lesion','Edema','Consolidation','Pneumonia','Atelectasis','Pneumothorax',
    'Pleural Effusion','Pleural Other','Fracture','Support Devices']

#Load and process the input Chexpert data to fit the Llava-Rad format
splits = pd.read_csv(preprocessed_path)
chexpert_obj = CheXpertPreprocessing(path, splits)
df = chexpert_obj.read_data()
df['section_indication'] = (df['section_history'].fillna('') + " " + df['section_clinical_history'].fillna(''))
df['chexpert_labels'] = df.apply(create_labels_dict, axis=1)
df['conversations'] = df.apply(create_conversations_list, axis=1)

# Printing out some useful metrics
print(df['path_to_image'].head())
print("Number of null findings: ", df['section_findings'].isnull().sum())
print("Number of null impressions: ", df['section_impression'].isnull().sum())
print("Length of chexpert data after preprocessing: ", len(df))
print(df['split'].value_counts())

def process_input_json_file(df_row):
    '''Method to process the input data file for the chexpert generate_method'''
    input_row = {
        'reason': df_row['section_indication'],
        'findings': df_row['section_findings'],
        'impressions': df_row['section_impression'],
        'image': df_row['path_to_image'],
        'generate_method': 'chexpert',
        'chexpert_labels': df_row['chexpert_labels'],
        'split': df_row['split'],
        'conversations': df_row['conversations']
    }
    return input_row

def process_gpt_file():
    '''Method to process the input data file for the gpt generate_method'''
    records = []
    counter = 0
    with open(input_json_file, "r") as f:
        for line in f:
            counter += 1
            line = line.strip()
            if not line:
                continue
            try:
                line = line.strip().replace('\\n', '').replace('\\', '')
                line = line.strip('"')
                record = json.loads(line)
                if isinstance(record, dict) and REQUIRED_COLUMNS.issubset(record.keys()):
                    records.append(record)
                else:
                    print(f"Skipping line {counter}: Record is not a dictionary")
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON on line {counter}: {e}")
                continue
    input_json_df = pd.DataFrame(records)
    length_input_json = len(input_json_df)
    print(f"Length of input json file: {length_input_json}")
    print(f"Number of json files processed: {counter}")
    print(input_json_df.head())

#Delete existing data.jsonl file if it exists - Append mode is used to write the processed data
with open("data.jsonl", "a") as f:
    for index, row in df.iterrows():
        prompt = process_input_json_file(row)
        f.write(json.dumps(prompt) + "\n")
