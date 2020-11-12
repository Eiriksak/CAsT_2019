import os
import json
import elasticsearch
from elasticsearch import Elasticsearch, helpers
import re
import time

from dotenv import load_dotenv
load_dotenv()



es = Elasticsearch(http_auth=(os.environ['ES_USER'], os.environ['ES_PWD']))
print(es.info())

cwd = os.getcwd()

file_name = cwd + "/collection/collection.tsv.xml"
file_path  = os.path.join(cwd, file_name)
print(f'Using MS Marco collection file at {file_path}')

INDEX_SETTINGS = {
    'mappings': {
            'properties': {
                'body': {
                    'type': 'text',
                    'term_vector': 'with_positions',
                },
            }
        }
    }


INDEX_NAME = 'trec2019'
if not es.indices.exists(INDEX_NAME):
    print(f'Creating new Elastic Search index: {INDEX_NAME}')
    es.indices.create(index=INDEX_NAME, body=INDEX_SETTINGS)

def index_msmarco(file_path):
    start = time.time()
    body_doc = {}
    num = 0
    actions = []
    with open(file_path, encoding="utf-8") as in_file:
        for i, line in enumerate(in_file):
            if "<DOCNO>" in line:
                doc_id = line.split("<DOCNO>")[1].split("</DOCNO>")[0] # gets the id between <DOCNO>-tag
            if "<" not in line[0]: # check to see that it is not a tag-line
                if line =="\n": # check to see that it is not empty
                    pass
                else:
                    actions.append({
                            "_index": INDEX_NAME,
                            "_type": "_doc",
                            "_id": doc_id,
                            "_source": {"body": line}
                    })
                    num += 1
                    if num > 0 and num%5000 == 0:
                        print(f'New bulk index at document number {num}')
                        res = helpers.bulk(es, actions)
                        print("Bulk response:", res)
                        index_time = time.time() - start
                        print(f'Indexing has been running for: {index_time:.0f} seconds')
                        actions = []
    print("Last bulk insert")
    res = helpers.bulk(es, actions)
    print("Bulk response:", res)



start_time = time.time()
index_msmarco(file_path)
total_time = time.time() - start_time
print(f'Indexing completed in : {total_time} seconds')
