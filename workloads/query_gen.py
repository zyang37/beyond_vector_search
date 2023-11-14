'''
This script should take in a dataframe and generate a list of queries (query text, k value)

A workload should be a dataframe, and will be write to a csv file. 
'''

import sys
import random
import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from pprint import pprint

sys.path.append("../")

from utils.parse_arxiv import *
from keyword_extractor import *

# random.seed(0)
# np.random.seed(0)

class QueryTemplate:
    def __init__(self):
        self.title = None
        self.author =[]
        self.year = None
        self.categories = None
        self.keywords = []
        self.journal = None
        self.abstract = None

        self.infor_prob = {
            'author': 0.5,
            'year': 0.5,
            'categories': 0.5,
            'keywords': 0.5,
            'journal': 0.5
        }

    def generate_query(self, title=False):
        start_text = "find papers"
        query = start_text
        if title:
            query += self.title_query()
        else:
            # if not random add infor to the query, use append_info()
            query += self.append_info()
            if query == start_text:
                # if no infor added, add title
                query += self.title_query()
        return query

    def append_info(self):
        # add info to the query based on the probability
        infor = ""
        if random.random() < self.infor_prob['author']:
            infor += self.author_query()
        if random.random() < self.infor_prob['year']:
            infor += self.year_query()
        if random.random() < self.infor_prob['categories']:
            infor += self.categories_query()
        if random.random() < self.infor_prob['keywords']:
            infor += self.keywords_query()
        if random.random() < self.infor_prob['journal']:
            infor += self.journal_query()
        return infor

    def update_prob(self, prob):
        # update the probability of adding info to the query
        for key in prob.keys():
            self.infor_prob[key] = prob[key]

    def parse_info(self, info: dict):
        # set the attributes of the class
        self.author = parse_authors(info['authors'])
        self.title = parse_title(info['title'])
        self.year = int(info['update_date'].split("-")[0])
        self.categories = parse_categories(info['categories'])

        # currently only use the top 30 keywords
        self.abstract = parse_abstract(info['abstract'])
        self.keywords = remove_noise_from_keywords(extract_keywords(self.abstract, score=False)[:30])
        self.journal = parse_journal(info['journal-ref'])
        
    def title_query(self):
        return " titled < {} >".format(self.title)
    
    def author_query(self):
        random_author = random.choice(self.author)
        return " written by < {} >".format(random_author)
    
    def year_query(self):
        return " from year {}".format(self.year)
    
    def categories_query(self):
        random_category = random.choice(self.categories)
        return " on < {} >".format(random_category)
    
    def keywords_query(self):
        random_keyword = random.choice(self.keywords)
        return " about < {} >".format(random_keyword)
    
    def journal_query(self):
        return " published at < {} >".format(self.journal)
    
    def print_info(self):
        print("title       :{}".format(self.title))
        print("abstract    :{}".format(self.abstract))
        print("author      :{}".format(self.author))
        print("year        :{}".format(self.year))
        print("categories  :{}".format(self.categories))
        print("keywords    :{}".format(self.keywords))
        print("journal     :{}".format(self.journal))
    

def generate_one():
    # generate a query
    ret_query = {'query':None, 'k':None, 'type': None}
    return ret_query

def generate_many():
    # generate a list of queries
    pass


if __name__ == '__main__':
    # get a number from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num', type=int, default=10, help='number of queries to generate')
    args = parser.parse_args()
    numbers_of_queries = args.num

    file = open('../data/filtered_data.pickle', 'rb')
    data = pickle.load(file)
    file.close()
    data.reset_index(drop=True, inplace=True)
    
    # sample 1 row from the data, print it out
    sample = data.sample(1).iloc[0]
    sample = dict(sample)
    query_template = QueryTemplate()
    query_template.parse_info(sample)

    print()
    print("<<< Selected paper >>>")
    # pprint(sample)
    query_template.print_info()

    # generate 10 queries
    print()
    print("<<< Randomly generated queries >>>")
    for i in range(numbers_of_queries):
        print("{}.".format(i+1), query_template.generate_query())
