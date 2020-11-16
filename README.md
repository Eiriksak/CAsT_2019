# CAsT_2019
TREC 2019 Conversational Assistance Track

## Setup
### Clone CAST repositories from root of this repository
1. git clone https://github.com/daltonj/treccastweb
2. git clone https://github.com/grill-lab/trec-cast-tools
3. git clone https://github.com/TREMA-UNH/trec-car-tools
4. git clone https://github.com/usnistgov/trec_eval

trec-cast-tools and trec-car-tools contains scripts that processes CAR and MS MARCO files
into correct xml format. Create a /collection folder from root and store these xml output files there.
You can run the index files index_msmarco.py and index_car.py once processed xml files are available
under /collection.

### Evaluate run files
Go to the trec_eval directory and run the following command in order to evaluate a run file:
```
./trec_eval -m ndcg_cut.3 -m map -m recip_rank {QREL_FILE} {RUN_FILE}
```
We can for instance reproduce the Indri baseline retrieval score by running
```
./trec_eval -m ndcg_cut.3 -m map -m recip_rank ../baseline/2019qrels_nowapo.txt ../runs/ES_BASE.txt
```

### Produce OpenMatch results
Please see [OpenMatch experiments page](https://github.com/thunlp/OpenMatch/blob/master/docs/experiments-msmarco.md) to see how to reproduce the OpenMatch results. The only thing changed from these parameters are the queries, docs and trec file, all other parameters are as stated.


## Files and folders
- /results contains .trec-files produced by OpenMatch.
- python scripts starting with "index" are the scripts used for indexing.
- /runs contains run-files for both OpenMatch and trec-eval, both automatic queries and resolved queries.
- re-ranking_with_LTR.ipynb is the notebook used to create training data, and train a PointWiseLTR model.
- /query_task contains code and notebooks that was used to develop and test different query models.
