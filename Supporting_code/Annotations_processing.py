"""
Script for processing and cleaning the google drive annotations.
The output is a simple .csv file with the acronyms and expansions per document.
Date: 17-05-2022
"""
# dependencies
import os
import pandas as pd
import re

"""
train --> "acronym": 20, "expansion": "secrecy rate", "id": "TR-0", "tokens":[]
test --> "acronym": 34, "id": "TS-0", "tokens": ["Experiment", "2", ":"

To do's: create test and train sets 
"""

rootdir = '/Users/jesher/Desktop/Master data science UvA/Semester 2/Thesis/google_annotation/json_data/'
df = pd.DataFrame(columns=['acronym', 'expansion', 'language', 'type'])


def cleaning_raw_annotation(df, rootdir):
    # Creating the file directories
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            abs_path = os.path.join(subdir, file)
            individual_json = pd.read_json(abs_path)

            # Extracting the annotators
            exception_mails = ['jesher_a@hotmail.com']

            # filer for mails with an underscore
            for i in exception_mails:
                if bool(re.search(i, abs_path.split('/')[-1])):
                    annotators = i
                else:
                    annotators = abs_path.split('_')[-1]
                    annotators = annotators[:-5]

            # Extracting the document ID's
            doc_id = abs_path.split('/')[-1]
            doc_id = doc_id.replace(annotators, '')[:-6]

            # Transforming the json  in a proper format
            individual_json = individual_json.transpose()
            individual_json.reset_index(inplace=True)
            individual_json = individual_json.rename(columns={'index':'acronym'})
            individual_json['doc_id'] = doc_id
            individual_json['annotator'] = annotators

            # adding everything together in a df
            df = pd.concat([df, individual_json], ignore_index=True)

            # fill the missing values in the language column
            df['language'].replace('', "other language", inplace=True)

    print(df.to_string())

if __name__ == "__main__":
    cleaning_raw_annotation(df, rootdir)