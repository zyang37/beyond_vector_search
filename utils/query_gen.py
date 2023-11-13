class QueryTemplate:
    def __init__(self):
        self.title = None
        self.author = None
        self.year = None
        self.categories = None
        self.keywords = None
        self.journal = None

    def parse_info(self, info: dict):
        # set the attributes of the class
        for key, value in info.items():
            setattr(self, key, value)

    def title_query(self):
        return "paper titled {self.title}"
    
    def author_query(self):
        return "papers written by {self.author}"
    
    def year_query(self):
        return "papers from {self.year}"
    
    def categories_query(self):
        return "papers written about {self.categories}"
    
    def keywords_query(self):
        return "papers written about {self.keywords}"
    
    def journal_query(self):
        return "papers published at {self.journal}"
    