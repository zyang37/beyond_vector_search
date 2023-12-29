import sys
import json
import pickle
import random
import argparse
import numpy as np
import pandas as pd
from pprint import pprint
from tqdm.auto import tqdm

sys.path.append("../")
from workloads.keyword_extractor import *
from build_graph import graph_extend_node_edge
from vector_graph.bipartite_graph_dict import BipartiteGraphDict

# fix the random seed
random.seed(0)
np.random.seed(0)

class CNN_NewsQueryTemplate:
    def __init__(self, prob_cfg: dict = None):
        self.cnn_news_parser = CnnNewsParser(None)
        self.prompt = "find news"
        if prob_cfg is not None:
            self.infor_prob = prob_cfg
        else:
            self.infor_prob = {
                'Keywords': 0.5,
                'Author': 0.5,
                'Date published': 0.5,
                'Section': 0.5,
                'Category': 0.5,
                'Article text': 1
            }

        self.query_funcs = {
            'Keywords': self.keywords_query,
            'Author': self.author_query,
            'Date published': self.date_query, 
            'Section': self.section_query,
            'Category': self.categories_query, 
            'Article text': self.article_text_query
        }
        self.cnn_news_parser.parse_func_dict['Article text'] = self.parse_article_text

        # initialize the infor_dict, which stores the parsed information later
        self.infor_dict = {}
        for key in self.query_funcs.keys():
            self.infor_dict[key] = []

    def generate_queries(self, title=False, num=1):
        # generate a list of queries
        queries = []
        for i in range(num):
            queries.append(self.generate_one_query(title=title))
        return queries

    def generate_one_query(self, title=False):
        start_text = self.prompt
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
        for key in self.infor_prob.keys():
            if random.random() < self.infor_prob[key]:
                infor += self.query_funcs[key]()
        return infor

    def update_prob(self, prob):
        # update the probability of adding info to the query
        for key in prob.keys():
            self.infor_prob[key] = prob[key]

    def parse_info(self, info: dict):
        # set the infor_dict
        self.title = info['Headline']
        for key in self.infor_dict.keys():
            parsed_infor = self.cnn_news_parser.parse_func_dict[key](info[key])
            self.infor_dict[key] = parsed_infor

        # keyword_scores: [(s1, k1), (s2, k2), ...]
        self.keyword_scores_dict = extract_keywords(self.infor_dict['Article text'], score=True)[:20]
        # get list of score
        self.keyword_weights = [s for s, k in self.keyword_scores_dict]
        # get list of clean keywords, note: this might reduce the number of keywords
        self.keywords = remove_noise_from_keywords([k for s, k in self.keyword_scores_dict])
        if len(self.keywords) == 0:
            '''
            Note: some paper's abstract contains lots of math, and the keyword extractor cannot extract any keywords
            '''
            # if no keywords, use the original keywords
            self.keywords = [k for s, k in self.keyword_scores_dict]

        # adjust the weights of keywords
        self.keyword_weights = self.keyword_weights[: len(self.keywords)]
        self.keyword_scores_dict = self.keyword_scores_dict[: len(self.keywords)]

    def parse_article_text(self, article_text):
        '''
        replace new line with space
        '''
        # replace multple space into one
        abstract_str = ' '.join(article_text.split())
        return abstract_str.replace("\n", " ")

    def article_text_query(self, max_num=3):
        # random_keyword = random.choice(self.keywords)
        # Randomly add more keywords (max 5), return str like "k1 and k2". no repeated keywords
        num = random.randint(1, max_num)
        if num >= len(self.keywords):
            '''
            Note: some paper have very few keywords, 
            so we need to make sure the number of keywords is not larger than the number of keywords in the paper
            '''
            num = len(self.keywords)
        try:
            keywords = random.choices(self.keywords, k=num, weights=self.keyword_weights)
        except:
            print(self.keywords, self.infor_dict['Article text'])
        keywords = list(set(keywords))
        random_keyword_list = "{}".format(" and ".join(keywords))
        return " about < {} >".format(random_keyword_list)

    def keywords_query(self):
        # random select a keyword from the keywords list
        random_keyword = random.choice(self.infor_dict['Keywords'])
        return " related to < {} >".format(random_keyword)

    def title_query(self):
        return " with headline < {} >".format(self.title)

    def author_query(self, max_num=3):
        max_num = min(max_num, len(self.infor_dict['Author']))
        num = random.randint(1, max_num)
        authors = random.sample(self.infor_dict['Author'], num)
        random_author_list = "{}".format(" and ".join(authors))
        return " written by < {} >".format(random_author_list)

    def date_query(self):
        return " published on < {} >".format(self.infor_dict['Date published'])

    def categories_query(self):
        return " on < {} >".format(self.infor_dict['Category'])

    def section_query(self):
        return " around the < {} >".format(self.infor_dict['Section'])

    def print_info(self):
        print("Title: {}".format(self.title))
        for key in self.infor_dict.keys():
            print("{}: {}".format(key, self.infor_dict[key]))


class CnnNewsParser:
    def __init__(self, df=None, id_col='Url'):
        self.df = df
        self.unique_ks = []
        self.exclude_authors = ['cnn', 'CNN']

        self.parse_func_dict = {
            'Keywords': self.parse_keywords,
            'Author': self.parse_authors,
            'Date published': self.parse_date, 
            'Section': self.parse_section,
            'Category': self.parse_categories,
        }
        
        if self.df is not None:
            self.id_col = id_col
            self.df[self.id_col] = self.df[self.id_col].astype("string")
            # drop duplicates
            self.df.drop_duplicates(subset=[self.id_col], inplace=True)
            self.unique_ks = np.unique(df['Section'])
            # check if all columns are in the df
            for k in self.parse_func_dict.keys():
                assert k in self.df.columns, f"{k} is not in the df"
            self.build_graph()
        else:
            # print("No df is provided.")
            pass

    def build_graph(self):
        self.G = BipartiteGraphDict()
        Knodes_dict = {}
        edges_dict = {}
        parse_infor_dict = {}
        for k in self.parse_func_dict.keys():
            Knodes_dict[k] = []
            edges_dict[k] = []
            parse_infor_dict[k] = self.df[k].map(self.parse_func_dict[k])

        for idx in tqdm(range(self.df.shape[0])):
            document_id = self.df[self.id_col].iloc[idx]
            for k in self.parse_func_dict.keys():
                Knodes_dict[k], edges_dict[k] = graph_extend_node_edge(idx, parse_infor_dict[k], k, Knodes_dict[k], document_id, edges_dict[k])

        self.G.add_data_nodes(set(self.df[self.id_col].tolist()))
        for k in self.parse_func_dict.keys():
            Knodes_dict[k] = set(Knodes_dict[k])
            edges_dict[k] = set(edges_dict[k])
            self.G.add_keyword_nodes(Knodes_dict[k])
            self.G.add_raw_edges(edges_dict[k])
        
        self.Knodes_dict = Knodes_dict
        self.edges_dict = edges_dict

    def parse_keywords(self, key_str):
        exclude_k = self.unique_ks
        def parse_colon(kstr):
            # parse "Paris attacks: What you need to know - CNN", get "Paris attacks"
            key_l = parse_by_comma[-1].split(":")
            if len(key_l) > 1:
                return key_l[0]
            return False
        key_list = []
        key_str = key_str.lower()
        parse_by_comma = key_str.split(", ")
        for k in parse_by_comma:
            if ":" in k:
                tmp = parse_colon(k)
                if tmp: 
                    k = tmp
                    # print(k)
            if k not in exclude_k:
                key_list.append(k.replace(" ", "-"))
        return key_list

    def parse_authors(self, author_str):
        '''
        This function parses the authors string into a list of authors

        args: 
            - author_str: string of authors
        return: list of authors
        '''
        exclude_list = self.exclude_authors
        authors = []
        author_str = author_str.lower()
        space_to = ""
        # remove noises
        # author_str = author_str.replace(" ", space_to)
        author_str = author_str.replace("\n", "")
        parse_by_comma = author_str.split(", ")
        for i, a in enumerate(parse_by_comma):
            if 'by' in a:
                a = a.split(' by ')[-1]
            # parse by 'and'
            if 'and' in a:
                al = []
                for v in a.split(' and '):
                    tmp = v.replace(" ", space_to)
                    if len(tmp)!=0 and (tmp not in exclude_list): 
                        al.append(tmp)
                a = al[:]
                        
            if type(a) is list:
                authors.extend(a)
            else:
                if len(a.replace(" ", space_to))!=0:
                    if a.replace(" ", space_to) in exclude_list: continue
                    authors.append(a.replace(" ", space_to))
        return authors

    def parse_date(self, time_str):
        return time_str.split(" ")[0]
    
    def parse_section(self, section_str):
        return section_str
    
    def parse_categories(self, cat_str):
        return cat_str
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a workload for the CNN News dataset")
    parser.add_argument("-d", "--data", type=str, default="../data/cnn_news/filtered_dataCNN.pickle", 
                        help="path to the data file")
    parser.add_argument("-nn", "--news_num", type=int, required=True, 
                        help="number of papers to generate queries from")
    parser.add_argument("-n", "--num", type=int, required=True,
        help="number of queries to generate per news article")
    parser.add_argument("-t", "--title", action="store_true",
        help="whether to include headline in the query")
    parser.add_argument("-s", "--save", type=str, default=None,
        help="where to save the workload (full path/name.csv)")
    parser.add_argument('--prob', type=str, default=None, help='path to the probability config file (json)')

    args = parser.parse_args()

    save_path = args.save
    # k = args.k
    paper_num = args.news_num
    num_queries_per_paper = args.num
    title = args.title
    prob_cfg_path = args.prob

    infor_prob = None
    if prob_cfg_path:
        with open(prob_cfg_path, 'r') as f:
            infor_prob = json.load(f)

    file = open(args.data, "rb")
    data = pickle.load(file)
    file.close()
    data.reset_index(drop=True, inplace=True)

    # sampling and parse the data => dict
    sample_dict_list = []
    # print(len(data))
    samples = data.sample(paper_num)
    # print(len(samples))
    for index, row in samples.iterrows():
        one_sample = dict(row)
        sample_dict_list.append(one_sample)

    # generate queries
    paper_id_list = []
    queries_list = []
    for d in tqdm(sample_dict_list):
        query_template = CNN_NewsQueryTemplate()
        if infor_prob is not None:
            query_template.update_prob(infor_prob)
        query_template.parse_info(d)
        queries_list.extend(
            query_template.generate_queries(title=title, num=num_queries_per_paper)
        )
        paper_id_list.extend([d["Url"]] * num_queries_per_paper)

    if save_path:
        # make pandas dataframe
        # df = pd.DataFrame({'paper_id': paper_id_list, 'query': queries_list, 'k': [k]*len(queries_list)})
        df = pd.DataFrame({"paper_id": paper_id_list, "query": queries_list})
        df.to_csv(save_path, index=False)
    else:
        # print queries
        for i, q in enumerate(queries_list):
            print("{:5d}. {:10}\n     - {}".format(i + 1, paper_id_list[i], q))
