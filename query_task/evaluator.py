import os
import json
import elasticsearch
from elasticsearch import Elasticsearch
import utils


es = Elasticsearch()

# Load test topica
path = "../treccastweb/2019/data/evaluation/evaluation_topics_v1.0.json"
with open(path, 'r') as f:
    test_topics = json.load(f)

for topic in test_topics:
    utils.add_qid(topic)


# Load test relevancy judgements
test_qrels_path = "../baseline/2019qrels_nowapo.txt"
with open(test_qrels_path, 'r') as f:
    qrels = []
    for line in f:
        qrels.append(line.strip())


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
