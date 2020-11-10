import os
import json
import elasticsearch
from elasticsearch import Elasticsearch
import math




def add_qid(topic):
    """Adds query id for each turn in a topic
    The query id is in the format (topic_number)_(turn_number)

    Args:
        topic (dict): one topic dictionary with keys; number, description, turn, title
    """
    number = topic['number']
    for turn in topic['turn']:
        turn['qid'] = '{}_{}'.format(number, turn['number'])


def get_qrels(qid, qrels):
    """Finds relevancy assesment for a certain qid
    Args:
        qid (string): query id for a turn in a topic. should be derived from add_qid()
        qrels (list): relevant list of qrel assesments
        
    Returns:
        list of tab separated (qid doc_id relevance) available for this qid.
        returns None if no judgement exists for the qid.
    """
    res = []
    relevancies = []
    for qrel in qrels:
        _qid, _, doc_id, relevance = qrel.split()
        if qid == _qid:
            relevancies.append(int(relevance))
            res.append('{} {} {}'.format(_qid, doc_id, relevance))
    if not res or sum(relevancies)==0:
        return None
    return res


def get_ground_truth(qrel):
    """returns dictionary of (document ID, relevance) pairs"""
    ground_truth = {}
    for q in qrel:
        _, doc_id, relevance = q.split()
        ground_truth[doc_id] = int(relevance)
    return ground_truth


def search(query, es, index, size=10):
    """Performs a base retrieval from an es instance"""
    search_results = es.search(index=index, q=query, size=size)['hits']['hits']
    if len(search_results) == 0:
        return None
    res = []
    for i, search_result in enumerate(search_results):
        doc_id = search_result['_id']
        doc = search_result['_source']['body']
        score = search_result['_score']
        res.append({
            'doc_id': doc_id,
            'rank': i,
            'score': score,
            'doc': doc
        })
    return res



def get_system_ranking(search_result):
    """Ranked results of document ids"""
    return [i['doc_id'] for i in search_result]



# Directly taken from:
# https://github.com/kbalog/ir-course/blob/master/solutions/20200922/Retrieval_evaluation_graded_relevance.ipynb
def dcg(relevances, k):
    """Computes DCG@k, given the corresponding relevance levels for a ranked list of documents.

    For example, given a ranking [2, 3, 1] where the relevance levels according to the ground
    truth are {1:3, 2:4, 3:1}, the input list will be [4, 1, 3].

    Args:
        relevances: List with the ground truth relevance levels corresponding to a ranked list of documents.
        k: Rank cut-off.

    Returns:
        DCG@k (float).
    """
    dcg = relevances[0]
    for i in range(1, min(k, len(relevances))):
        dcg += relevances[i] / math.log(i + 1, 2)  # Note: Rank position is indexed from 1.
    return dcg


# Directly taken from:
# https://github.com/kbalog/ir-course/blob/master/solutions/20200922/Retrieval_evaluation_graded_relevance.ipynb
def ndcg(system_ranking, ground_truth, k=10):
    """Computes NDCG@k for a given system ranking.

    Args:
        system_ranking: Ranked list of document IDs (from most to least relevant).
        ground_truth: Dict with document ID: relevance level pairs. Document not present here are to be taken with relevance = 0.
        k: Rank cut-off.

    Returns:
        NDCG@k (float).
    """
    relevances = []  # Holds corresponding relevance levels for the ranked docs.
    for doc_id in system_ranking:
        relevances.append(ground_truth.get(doc_id, 0))

    # Relevance levels of the idealized ranking.
    relevances_ideal = sorted([v for _, v in ground_truth.items()], reverse=True)

    return dcg(relevances, k) / dcg(relevances_ideal, k)
