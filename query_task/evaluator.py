import os
import json
import elasticsearch
from elasticsearch import Elasticsearch
import utils
from abc import ABC, abstractmethod


# Load test topics
path = "../treccastweb/2019/data/evaluation/evaluation_topics_v1.0.json"
with open(path, 'r') as f:
    test_topics = json.load(f)

for topic in test_topics:
    utils.add_qid(topic)


# Load train topics
path = "../treccastweb/2019/data/training/train_topics_v1.0.json"
with open(path, 'r') as f:
    train_topics = json.load(f)

for topic in train_topics:
    utils.add_qid(topic)

# Load test relevancy judgements
test_qrels_path = "../baseline/2019qrels_nowapo.txt"
with open(test_qrels_path, 'r') as f:
    test_qrels = []
    for line in f:
        test_qrels.append(line.strip())

# Load train relevancy judgements
train_qrels_path = "../baseline/train_topics_mod_nowapo.qrel"
with open(train_qrels_path, 'r') as f:
    train_qrels = []
    for line in f:
        train_qrels.append(line.strip())
        
# Load resolved test queries (reverse engineer TSV format into same JSON format as the rest)
path = "../treccastweb/2019/data/evaluation/evaluation_topics_annotated_resolved_v1.0.tsv"
with open(path, 'r') as f:
    test_topics_resolved = []
    last_id = -1
    for line in f:
        line = line.strip()
        qid, txt = line.split("\t")
        topic_num, turn_num = qid.split("_")
        if topic_num != last_id:
            last_id = topic_num
            test_topics_resolved.append({
                'number': topic_num,
                'turn':[],
                'description': '',
                'title': ''
            })
        
        num = len(test_topics_resolved)-1
        test_topics_resolved[num]['turn'].append({
            'qid': qid,
            'raw_utterance': txt,
            'number': turn_num
        }) 

class QueryExpander(ABC):
    
    def __init__(self):
        self.question_history = {}
        self.query_history = {}
        
    def get_query(self, query, depth, topic_num):        
        generated_query = self.generate_query(query, depth, topic_num)
        
        if topic_num not in self.question_history:
            self.question_history[topic_num] = []
            self.query_history = []
        self.question_history[topic_num].append(query)
        self.query_history.append(generated_query)
        return generated_query
        
    @abstractmethod
    def generate_query(self, query, depth, topic_num):
        pass



def get_search_data(query, qid, qrels, es, index, size=10):
    """Generates a list of search results together with system ranking and ground truth
    The generated data can be used to evaluate the system
    
    Args:
        query: query tokens 
        qid: query id for a turn in a topic
        qrels: relevant list of qrel assesments
        es: running elasticsearch instance
        index: name of es index to query against
        size: number of items to return from the es search
        
    Returns:
        search_results: list of search result objects containing; doc_id, rank, score, doc
        system_rankings: list of document id's in descending rank order
        ground_truth: dictionary with labeled relevancy judgements for this query
    """
    search_results = utils.search(query, es, index, size)
    if search_results is None:
        return None, None, None
    system_ranking = utils.get_system_ranking(search_results)
    qrels = utils.get_qrels(qid, qrels)
    if qrels is None:
        return None, None, None
    ground_truth = utils.get_ground_truth(qrels)
    return search_results, system_ranking, ground_truth


def to_run_file(data, filename, literal, tag):
    """ Generates and stores a run file based on search results
    Args:
        data (dict): qid as key, list of {doc_id: .., rank: .., score: ..} dict as value
        filename: name of the run file. will be stored under /runs directory
        literal: literal column in run file
        tag: tag column in run file
    """
    # qid literal docno rank score tag
    res = []
    for qid, values in data.items():
        for d in values:
            inp = "{} {} {} {} {} {}".format(qid, literal, d['doc_id'], d['rank'], d['score'], tag)
            res.append(inp)

    res = "\n".join(res)
    with open("../runs/"+filename+".txt", 'w') as f:
        f.write(res)
    return


def analyzer(es, query, index):
    """Performs analysis process on the query text with respect to a given index
    Args:
        es: running elasticsearch instance
        query (string): query tokens
        index: name of es index to query against
    Returns:
        list of processed tokens in the query
    """
    analyzed = es.indices.analyze(index=index, body={'text': query})
    analyzed = sorted(analyzed['tokens'], key=lambda x: x['position']) # Ensure query is in correct order
    return [i['token'] for i in analyzed]



def evaluate(es, index, size, qm, topics=test_topics, qrels=test_qrels, k=3):
    """
    Args:
        es: running elasticsearch instance
        index: name of es index to query against
        size: number of items to return from each es search
        qm: query model that inherits the QueryExpander class
        topics: topics file in json format (test or train)
        qrels: relevant list of qrel assesments (test or train)
        k: NDCG@k. default is 3 (official metric for TREC2019)

    Returns:
        data: dictionary with search result data. can be fed directly into to_run_file
        scores: NDCG@k scores for all queries
        turn_depth_scores: NDCG@k scores at each depth of an conversation
    """
    data = {}
    turn_depth_scores = {}
    scores = []
    for topic in topics:
        topic_num = topic['number']
        for depth, turn in enumerate(topic['turn']):
            query = qm.get_query(query=turn['raw_utterance'], depth=depth, topic_num=topic_num)
            search_results, system_ranking, ground_truth = get_search_data(query,
                                                                           turn['qid'],
                                                                           qrels,
                                                                           es,
                                                                           index,
                                                                           size)

            if search_results is None:
                continue

            data[turn['qid']] = search_results
            score = utils.ndcg(system_ranking, ground_truth, k=k)
            scores.append(score)

            if turn['number'] not in turn_depth_scores:
                turn_depth_scores[turn['number']] = []
            turn_depth_scores[turn['number']].append(score)
    return data, scores, turn_depth_scores


def save_turn_depths(data, filename):
    """ Stores NDCG@3 scores for each depth of a run in a JSON file
    Args:
        data (dict): depth as key, list of NDCG@3 scores as values
        filename: name of the run file. will be stored under /results directory
                  you should use the same filename as in to_run_file
    """
    
    with open("../results/"+filename+".json", "w") as f:
        json.dump(data, f, indent=4)
    return
