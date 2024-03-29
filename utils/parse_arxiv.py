import json
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from collections import Counter

def save_json(data: dict, path: str, verbose: bool = True):
    # dump the config file 
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"Saved to {path}")

def load_json(path: str, verbose: bool = True):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if verbose:
        print(f"Loaded from {path}")
    return data

def get_metadta_str(df_row, metadata_cols):
    '''
    This function gets the metadata string from the dataframe row

    args:
        - df_row: dataframe row
        - metadata_cols: list of metadata columns
    return: metadata string
    '''
    metadata_str = ""
    for col in metadata_cols:
        metadata_str += "{} {}".format(col, str(df_row[col])) + " "
    return metadata_str

def sort_dict(dict_data, byval=True, reverse=True):
    '''
    This function sorts a dictionary by value or key

    args:
        - dict_data: dictionary to be sorted
        - byval: sort by value or key
        - reverse: sort in ascending or descending order
    return: sorted dictionary
    '''
    if byval: idx = 1
    else: idx = 0
    return dict(sorted(dict_data.items(), key=lambda item: item[idx], reverse=reverse))

def parse_authors(author_str):
    '''
    This function parses the authors string into a list of authors

    args: 
        - author_str: string of authors
    return: list of authors
    '''
    authors = []
    # remove noises
    author_str = author_str.replace(" ", "")
    author_str = author_str.replace("\n", "")
    parse_by_comma = author_str.split(',')
    for a in parse_by_comma:
        # parse by 'and'
        authors.extend(a.split(' and '))
    return authors

def parse_title(title_str):
    '''
    replace new line with space
    '''
    # replace multple space into one
    title_str = ' '.join(title_str.split())
    return title_str.replace("\n", " ")

def parse_abstract(abstract_str):
    '''
    replace new line with space
    '''
    # replace multple space into one
    abstract_str = ' '.join(abstract_str.split())
    return abstract_str.replace("\n", " ")

def parse_journal(journal_str):
    '''
    replace new line with space
    '''
    # replace multple space into one
    journal_str = ' '.join(journal_str.split())
    return journal_str.replace("\n", " ")

def parse_categories(cat_str):
    '''
    This function parses the categories string into a list of categories

    args:
        - cat_str: string of categories
    return: list of categories
    '''
    # split by space
    cat_list = cat_str.split(' ')
    return cat_list

def parse_year(date_str):
    '''
    This function parses the date string into a year

    args:
        - date_str: string of date
    return: year
    '''
    return date_str.split('-')[0]

def count_name_frequencies(name_list2d):
    '''
    This function counts the frequency of each name in a list of lists

    args:
        - name_list2d: list (paper) of lists of names
    return: dictionary of name frequencies
    '''
    # Flatten the list of lists into a single list of names
    all_names = [name for sublist in name_list2d for name in sublist]
    # Use Counter to count the frequency of each name
    name_frequencies = Counter(all_names)
    return dict(name_frequencies)

def merge_add_dict(a, b):
    '''
    This function merges two dictionaries by adding values of common keys
    '''
    return {key: a.get(key, 0) + b.get(key, 0) for key in set(a) | set(b)}

def gather_stats(df):
    '''
    This function gathers statistics from the dataframe

    args:
        - df: dataframe
    return: dictionary of statistics
    '''
    intermediate_submitter_stats = dict(df['submitter'].value_counts())
    intermediate_authors_stats = count_name_frequencies(list(map(parse_authors, df['authors'])))
    # intermediate_authors_stats = merge_add_dict(intermediate_submitter_stats, intermediate_authors_stats)
    intermediate_cat_stats = dict(df['categories'].value_counts())
    intermediate_jou_stats = dict(df['journal-ref'].value_counts())
    intermediate_year_stats = dict(df['update_date'].value_counts())
    return intermediate_authors_stats, intermediate_cat_stats, intermediate_jou_stats, intermediate_year_stats

def make_keyword_id(category, value):
    '''
    This function makes a keyword id from category and value

    args:
        -  category: category of the keyword
        -  value: value of the keyword
    return: keyword id
    '''
    return category + ":" + value
