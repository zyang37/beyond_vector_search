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
from utils.build_graph import graph_extend_node_edge
from vector_graph.bipartite_graph_dict import BipartiteGraphDict

# fix the random seed
random.seed(0)
np.random.seed(0)


class Wiki_MovieQueryTemplate:
    def __init__(self, prob_cfg: dict = None):
        self.wiki_movies_parser = WikiMoviesParser(None)
        self.prompt = "find movies"
        if prob_cfg is not None:
            self.infor_prob = prob_cfg
        else:
            self.infor_prob = {
                "Origin": 0.5,
                "Director": 0.5,
                "Cast": 0.5,
                "Genre": 0.5,
                "Year": 0.5,
                "Plot": 1,
            }

        self.query_funcs = {
            "Origin": self.origin_query,
            "Director": self.director_query,
            "Cast": self.cast_query,
            "Genre": self.genre_query,
            "Year": self.year_query,
            "Plot": self.plot_query,
        }
        self.wiki_movies_parser.parse_func_dict["Plot"] = self.parse_plot

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
            query += self.plot_query()
        else:
            # if not random add infor to the query, use append_info()
            query += self.append_info()
            if query == start_text:
                # if no infor added, add title
                query += self.plot_query()
        return query.replace("\r", "")

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
        self.title = info["Plot Summary"]
        for key in self.infor_dict.keys():
            parsed_infor = self.wiki_movies_parser.parse_func_dict[key](info[key])
            self.infor_dict[key] = parsed_infor

        # keyword_scores: [(s1, k1), (s2, k2), ...]
        self.keyword_scores_dict = extract_keywords(
            self.infor_dict["Plot"], score=True
        )[:20]
        # get list of score
        self.keyword_weights = [s for s, k in self.keyword_scores_dict]
        # get list of clean keywords, note: this might reduce the number of keywords
        self.keywords = remove_noise_from_keywords(
            [k for s, k in self.keyword_scores_dict]
        )
        if len(self.keywords) == 0:
            """
            Note: some paper's abstract contains lots of math, and the keyword extractor cannot extract any keywords
            """
            # if no keywords, use the original keywords
            self.keywords = [k for s, k in self.keyword_scores_dict]

        # adjust the weights of keywords
        self.keyword_weights = self.keyword_weights[: len(self.keywords)]
        self.keyword_scores_dict = self.keyword_scores_dict[: len(self.keywords)]

    def parse_plot(self, plot):
        """
        replace new line with space
        """
        # replace multple space into one
        abstract_str = " ".join(plot.split())
        return abstract_str.replace("\n", " ")

    def plot_query(self, max_num=5):
        # random_keyword = random.choice(self.keywords)
        # Randomly add more keywords (max 5), return str like "k1 and k2". no repeated keywords
        num = random.randint(1, max_num)
        if num > len(self.keywords):
            """
            Note: some paper have very few keywords,
            so we need to make sure the number of keywords is not larger than the number of keywords in the paper
            """
            num = len(self.keywords)
        keywords = random.choices(self.keywords, k=num, weights=self.keyword_weights)
        keywords = list(set(keywords))
        random_keyword_list = "{}".format(" and ".join(keywords))
        return " about < {} >".format(random_keyword_list)

    def origin_query(self):
        # random select a keyword from the keywords list
        return " that are < {} >".format(self.infor_dict["Origin"])

    def director_query(self):
        try:
            random_keyword = random.choice(self.infor_dict["Director"])
        except:
            print(self.infor_dict)
        return " directed by < {} >".format(random_keyword)

    def cast_query(self, max_num=3):
        max_num = min(max_num, len(self.infor_dict["Cast"]))
        num = random.randint(1, max_num)
        casts = random.sample(self.infor_dict["Cast"], num)
        random_cast_list = "{}".format(" and ".join(casts))
        return " played by < {} >".format(random_cast_list)

    def genre_query(self):
        return " with genre {}".format(self.infor_dict["Genre"])

    def year_query(self):
        return " in the year {}".format(self.infor_dict["Year"])

    def print_info(self):
        print("Plot Summary: {}".format(self.title))
        for key in self.infor_dict.keys():
            print("{}: {}".format(key, self.infor_dict[key]))


class WikiMoviesParser:
    def __init__(self, df=None, id_col="Title"):
        self.df = df
        self.unique_ks = []
        self.exclude_authors = []

        self.parse_func_dict = {
            "Origin": self.parse_origin,
            "Director": self.parse_director,
            "Cast": self.parse_casts,
            "Genre": self.parse_genres,
            "Year": self.parse_year,
        }

        if self.df is not None:
            self.id_col = id_col
            self.df[self.id_col] = self.df[self.id_col].astype("string")
            # drop duplicates
            self.df.drop_duplicates(subset=[self.id_col], inplace=True)
            # self.unique_ks = np.unique(df["Section"])
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
                Knodes_dict[k], edges_dict[k] = graph_extend_node_edge(
                    idx,
                    parse_infor_dict[k],
                    k,
                    Knodes_dict[k],
                    document_id,
                    edges_dict[k],
                )

        self.G.add_data_nodes(set(self.df[self.id_col].tolist()))
        for k in self.parse_func_dict.keys():
            Knodes_dict[k] = set(Knodes_dict[k])
            edges_dict[k] = set(edges_dict[k])
            self.G.add_keyword_nodes(Knodes_dict[k])
            self.G.add_raw_edges(edges_dict[k])

        self.Knodes_dict = Knodes_dict
        self.edges_dict = edges_dict

    def parse_year(self, year):
        return str(year)

    def parse_origin(self, origin):
        return str(origin)

    def parse_director(self, director_str):
        directors = []
        director_str = director_str.lower()
        space_to = ""
        director_str = director_str.replace("\n", "")
        parse_by_comma = director_str.split(", ")
        for i, a in enumerate(parse_by_comma):
            # parse by 'and'
            if "and" in a:
                al = []
                for v in a.split(" and "):
                    tmp = v.replace(" ", space_to)
                    if len(tmp) != 0:
                        al.append(tmp)
                a = al[:]

            if type(a) is list:
                directors.extend(a)
            else:
                if len(a.replace(" ", space_to)) != 0:
                    directors.append(a.replace(" ", space_to))
        return directors

    def parse_genres(self, genre_str):
        genres = []
        genre_str = genre_str.lower()
        space_to = ""
        genre_str = genre_str.replace("\n", "")
        parse_by_comma = genre_str.split(", ")
        for i, a in enumerate(parse_by_comma):
            # parse by 'and'
            if "and" in a:
                al = []
                for v in a.split(" and "):
                    tmp = v.replace(" ", space_to)
                    if len(tmp) != 0:
                        al.append(tmp)
                a = al[:]

            if type(a) is list:
                genres.extend(a)
            else:
                if len(a.replace(" ", space_to)) != 0:
                    genres.append(a.replace(" ", space_to))
        return genres

    def parse_casts(self, cast_str):
        casts = []
        cast_str = cast_str.lower()
        space_to = ""
        cast_str = cast_str.replace("\n", "")
        parse_by_comma = cast_str.split(", ")
        for i, a in enumerate(parse_by_comma):
            # parse by 'and'
            if "and" in a:
                al = []
                for v in a.split(" and "):
                    tmp = v.replace(" ", space_to)
                    if len(tmp) != 0:
                        al.append(tmp)
                a = al[:]

            if type(a) is list:
                casts.extend(a)
            else:
                if len(a.replace(" ", space_to)) != 0:
                    casts.append(a.replace(" ", space_to))
        return casts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a workload for the CNN News dataset"
    )
    parser.add_argument(
        "-d",
        "--data",
        type=str,
        default="../data/wiki_movies/filtered_data_wiki_movies.pickle",
        help="path to the data file",
    )
    parser.add_argument(
        "-nn",
        "--news_num",
        type=int,
        required=True,
        help="number of papers to generate queries from",
    )
    parser.add_argument(
        "-n",
        "--num",
        type=int,
        required=True,
        help="number of queries to generate per news article",
    )
    parser.add_argument(
        "-t",
        "--title",
        action="store_true",
        help="whether to include headline in the query",
    )
    parser.add_argument(
        "-s",
        "--save",
        type=str,
        default=None,
        help="where to save the workload (full path/name.csv)",
    )
    parser.add_argument(
        "--prob",
        type=str,
        default=None,
        help="path to the probability config file (json)",
    )

    args = parser.parse_args()

    save_path = args.save
    # k = args.k
    paper_num = args.news_num
    num_queries_per_paper = args.num
    title = args.title
    prob_cfg_path = args.prob

    infor_prob = None
    if prob_cfg_path:
        with open(prob_cfg_path, "r") as f:
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
        query_template = Wiki_MovieQueryTemplate()
        if infor_prob is not None:
            query_template.update_prob(infor_prob)
        query_template.parse_info(d)
        queries_list.extend(
            query_template.generate_queries(title=title, num=num_queries_per_paper)
        )
        paper_id_list.extend([d["Title"]] * num_queries_per_paper)

    if save_path:
        # make pandas dataframe
        # df = pd.DataFrame({'paper_id': paper_id_list, 'query': queries_list, 'k': [k]*len(queries_list)})
        df = pd.DataFrame({"id": paper_id_list, "query": queries_list})
        df.to_csv(save_path, index=False)
    else:
        # print queries
        for i, q in enumerate(queries_list):
            print("{:5d}. {:10}\n     - {}".format(i + 1, paper_id_list[i], q))
