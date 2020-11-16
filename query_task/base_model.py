import evaluator
from evaluator import QueryExpander


class BaseQuery(QueryExpander):
    
    def __init__(self, es, index):
        super().__init__()
        self._es = es
        self.index = index
        
    def generate_query(self, query, depth, topic_num):
        return evaluator.analyzer(self._es, query, self.index)
