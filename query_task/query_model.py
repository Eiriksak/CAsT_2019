from collections import Counter
import os
import utils
import evaluator
from evaluator import test_topics, test_qrels, train_topics, train_qrels, QueryExpander
import numpy as np



class POSQuery(QueryExpander):
    """Class for advanced query expansion using part-of-speech tagging
    Example usage: 
    qm = POSQuery(es, 'trec2019_stem', nlp)
    qm.fit(train_topics, train_qrels, 100)
    
    qm.get_query('What is throat cancer?', depth=1, topic_num=1)
     => ['throat', 'cancer'] (cleans query in category 1)
     
    qm.get_query('Is it treatable?', depth=2, topic_num=1)
     => ['treatable', 'throat', 'cancer'] (expands query in category 2)
    """
    
    
    
    def __init__(self, es, index, nlp):
        super().__init__()
        self._es = es
        self.index = index
        self.nlp = nlp
        self.pos_vocab = None
        self.cat1 = {} # First question or topic shift
        self.cat2 = {} # Need context
        self.I = {}
        
    def init_pos_vocab(self, X):
        """Initialize vocab of pos tags based on all questions in training data"""
        pos_vocab = []
        for topic in X:
            for turn in topic['turn']:
                doc = self.nlp(turn['raw_utterance'])
                for token in doc:
                    if token.pos_ not in pos_vocab:
                        pos_vocab.append(token.pos_)
        self.pos_vocab = pos_vocab
        return 
    
    
    def init_categories(self, X, y):
        """Initialize data for both categories
        Each category is represented as a dictionary where 
        """
        cat1 = {} 
        cat2 = {} 
        I = {}
        for topic in X:
            for turn in topic['turn']:
                qid = turn['qid']
                _qrels = utils.get_qrels(qid, y)
                q = ' '.join(evaluator.analyzer(self._es, turn['raw_utterance'], self.index))
                I[qid] = {'text': q}
                if not _qrels: # Only include turns with labeled relevancy
                    continue
                txt, pos = self.pos_parser(q, self.pos_vocab)
                pos_vec = self.get_pos_counts(pos)
                cat = self.get_category(pos)
                if turn['number'] == 1 or cat==1:
                    cat1[qid] = {'pos_vec': pos_vec}
                else:
                    cat2[qid] = {'pos_vec': pos_vec}
        self.cat1 = cat1
        self.cat2 = cat2
        self.I = I
        return
    
    
    def pos_parser(self, text, pos_filter):
        """Returns tokens and tags within a given filter of pos tags"""
        doc = self.nlp(text)
        tags = []
        res = []
        for token in doc:
            if token.pos_ in pos_filter:
                tags.append(token.pos_)
                res.append(token.text)
        return res, tags
    
    
    def get_pos_counts(self, pos_list):
        """Returns vector of counts per pos-tag in pos_vocab"""
        return [pos_list.count(i) for i in self.pos_vocab]
    
    
    def get_category(self, pos):
        """Finds out which category a query belongs to based on pos-tags in it"""
        num = sum([1 for tag in pos if tag in ['NOUN', 'PROPN']])
        if num > 2:
            return 1
        return 2
    
    
    def euclidean_distance(self, x, y): 
        """Returns the euclidean distance between two pos_count vectors"""
        return np.sqrt(np.sum((np.array(x) - np.array(y)) ** 2))
    
    
    def knn(self, pos_vec, cat, qid='None', k=1):
        """Returns k nearest query neighbors within a category based on pos_count vectors"""
        dists = {}
        for key, v in cat.items():
            if key != qid:
                dist = self.euclidean_distance(v['pos_vec'], pos_vec)
                dists[key] = dist
        dists = {key: v for key, v in sorted(dists.items(), key=lambda item: item[1])}
        return {key:dists[key] for key in list(dists.keys())[:k]}
        
        
    def get_base_score(self, q, qid, y, size=100):
        """Returns ndcg@3 score for a query q (base query)"""
        _, system_ranking, ground_truth = evaluator.get_search_data(q, qid, y, self._es, self.index, size)
        return utils.ndcg(system_ranking, ground_truth, k=3)
        
        
    def get_top_res(self, qid, y):
        """Returns relevancy qrels with a score above 1 in sorted descending order"""
        _qrels = utils.get_qrels(qid, y)
        if not _qrels:
            return None
        ground_truth = utils.get_ground_truth(_qrels)
        return {k: v for k, v in sorted(ground_truth.items(), key=lambda item: item[1], reverse=True) if v>0}
    
    
    def get_keywords(self, text, k=20):
        """Returns k most common keywords in a text. All keywords must be within given pos_vocab"""
        doc = self.nlp(text)
        tokens = [token.text for token in doc if token.pos_ in self.pos_vocab]
        tokens = Counter(tokens).most_common(k) # [(word, count), (word, count)..] descending order by count
        return tokens
    
    
    def get_kw_stats(self, top_res):
        """Computes keyword statistics for top results. All keywords must be within given pos_vocab 
        Keyword counts from the top results of a query can be used to derive which types of words were
        most important for the query
        
        Args:
            top_res (dict): doc_id:relevancy pairs from the training qrels
            
        Returns:
            res (dict): docuement text with sorted keyword counts
            kw_count (list): sorted keyword count of all documents together
        """
        res = {}
        kw_count = {}
        for doc_id, relevance in top_res.items():
            search_res = self._es.get(index=self.index, id=doc_id)
            doc = search_res['_source']['body']
            doc = ' '.join(evaluator.analyzer(self._es, doc, self.index))
            keywords = self.get_keywords(doc)
            for kw in keywords:
                if kw[0] not in kw_count:
                    kw_count[kw[0]] = 0
                kw_count[kw[0]] +=  kw[1] # 2 in relevancy more important
            res[doc_id] = {'doc': doc, 'keywords': keywords}
        return res, kw_count
    
    
    def get_perfect_query(self, qid, y, size=100):
        """Reverse engineers a query into a 'perfect' query
        Args:
            qid: query id for the one we want to reverse engineer
            y: training qrels
            pos_vocab: pos-tags to filter on
            size: number of search results to return
        
        Returns:
            top_query: query based on top keywords of search results we know was most relevant
                       for the given query. we check ndcg@3 scores for top k keywords where 
                       k=20,..,1. the k with highest score is returned, with longer k's prioritized
                       over shorter k's
        
        """
        base_score = self.get_base_score(self.I[qid]['text'], qid, y)
        top_res = self.get_top_res(qid, y)        
        _, kw_count = self.get_kw_stats(top_res)
        topk_kw = {key:kw_count[key] for key in list(kw_count.keys())}
        last_query = []
        top_query = {'score': 0, 'base_score': base_score ,'q': None}
        for s in range(20, 1, -1):
            query = [i for i in topk_kw if i not in self.nlp.Defaults.stop_words][:s] # Top s keywords
            if query == last_query:
                continue
            query = evaluator.analyzer(self._es, ' '.join(query), self.index)
            last_query = query
            _, system_ranking, ground_truth = evaluator.get_search_data(query, qid, y, self._es, self.index, size)
            if system_ranking is None:
                continue

            score = utils.ndcg(system_ranking, ground_truth, k=3)
            if score >= top_query['score']:
                top_query = {'score': score, 'base_score': base_score, 'q': query}

        q = ' '.join([i for i in top_query['q']])
        txt, pos =  self.pos_parser(q, self.pos_vocab)
        pos_vec = self.get_pos_counts(pos)
        top_query['pos_vec'] = pos_vec
        top_query['pos'] = set(pos)
        return top_query
    
    
    def rewrite_query(self, q, q_pos, history, pos_filter, pq):
        """Rewrites a query by expanding it with tokens from previous questions in conversation
        Args:
            q: parsed query
            q_pos: pos-tags in the parsed query
            history (list): previous questions within the conversatin
            pos_filter: we will only expand with tokens that has a tag in this filter
            pq: perfect query from the nearest neighbor (in training data) of q 
        
        Returns:
            q + expanding tokens. The longest expansion where the score maximum is 0.5
            more than the best one is returned. 
        """
        expanded_queries = [q.copy()] # Every expansion starts with the original parsed query
        expanded_queries_pos = [q_pos.copy()] # Store all pos-tags in expanded queries so we can compute pos_vecs
        for h in history:
            txt, pos = self.pos_parser(h, pos_filter)
            for t, p in zip(txt, pos):
                if len(expanded_queries) > 25000:
                    if p in ['NOUN', 'PROPN', 'ADV']:
                        tmp = q.copy() + [t]
                        tmp_p = q_pos.copy() + [p]
                        expanded_queries.append(tmp)
                        expanded_queries_pos.append(tmp_p)
                else:
                    for ep, e in zip(expanded_queries_pos ,expanded_queries):
                        if t not in e:
                            tmp = e + [t]
                            tmp_p = ep + [p]
                            expanded_queries.append(tmp)
                            expanded_queries_pos.append(tmp_p)
        
        # Calculate pos_vector for all expanded queries
        pos_vecs  = [self.get_pos_counts(p) for p in expanded_queries_pos]
        # Compute euclidean distance between perfect query and all expanded queries
        dists = [self.euclidean_distance(pq['pos_vec'], pos_vec) for pos_vec in pos_vecs]
        top_q = expanded_queries[np.argmin(dists)] # Query with the best score
        s = dists[np.argmin(dists)]
        # Switch top query if we find a longer one with a score maximum 0.5 more than original top query
        for i in range(len(expanded_queries)):
            l = len(expanded_queries[i])
            if l > len(top_q) and (s-dists[i]) < .5:
                top_q = expanded_queries[i]
        return top_q
    
    
    def fit(self, X, y, size=100):
        # Initialize pos_vocab and categories from training data
        self.init_pos_vocab(X)
        self.init_categories(X, y)
        # Compute perfect queries for both categories
        for qid in self.cat1:
            perfect_query = self.get_perfect_query(qid, y)
            self.cat1[qid]['perfect_query'] = perfect_query
        print(f'Done computing perfect queries for category 1')
        for qid in self.cat2:
            perfect_query = self.get_perfect_query(qid, y)
            self.cat2[qid]['perfect_query'] = perfect_query
        print(f'Done computed perfect queries for category 2')
        return
            

    def generate_query(self, query, depth, topic_num):
        """Generate a query based on the question and previous conversational history"""
        q = ' '.join(evaluator.analyzer(self._es, query, self.index))
        txt, pos = self.pos_parser(q, self.pos_vocab)
        pos_vec = self.get_pos_counts(pos)
        cat = self.get_category(pos)
        
        if depth == 1 or cat == 1: 
            # Category 1 data => parse original query based on pos-tags from top results
            # of similar queries in training data
            nb = self.knn(self.get_pos_counts(pos), self.cat1, k=3)
            pos_filter = []
            for n in nb: # n=qid
                pos_filter = pos_filter + list(self.cat1[n]['perfect_query']['pos'])
            pos_filter = set(pos_filter)
            parsed_query, _ = self.pos_parser(q, pos_filter)
            return parsed_query
        
        else:
            # Category 2 data => parse original query based on pos-tags from top results
            # of similar queries in training data. perform rewriting algorithm on the
            # parsed query and return the response
            nb = self.knn(self.get_pos_counts(pos), self.cat2, k=3)
            pos_filter = []
            for n in nb: # n=qid
                pos_filter = pos_filter + list(self.cat2[n]['perfect_query']['pos'])
            pos_filter = set(pos_filter)
            parsed_query, pos = self.pos_parser(q, pos_filter)
            history = self.question_history[topic_num]
            pq = self.cat2[list(nb.keys())[0]]['perfect_query']
            new_q = self.rewrite_query(parsed_query, pos, history, pos_filter, pq)
            if not new_q:
                return evaluator.analyzer(self._es, query, self.index)
            return new_q
