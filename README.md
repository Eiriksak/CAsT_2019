# CAsT_2019
TREC 2019 Conversational Assistance Track

## Setup
### Clone CAST repositories from root of this repository
1. git clone https://github.com/daltonj/treccastweb
2. git clone https://github.com/grill-lab/trec-cast-tools
3. git clone https://github.com/TREMA-UNH/trec-car-tools

trec-cast-tools and trec-car-tools contains scripts that processes CAR and MS MARCO files
into correct xml format. Create a /collection folder from root and store these xml output files there.
You can run the index files index_msmarco.py and index_car.py once processed xml files are available
under /collection.

### Produce OpenMatch results
Please see [OpenMatch experiments page](https://github.com/thunlp/OpenMatch/blob/master/docs/experiments-msmarco.md) to see how to reproduce the OpenMatch results.


## Files and folders
- /results contains .trec-files produced by OpenMatch.
- python scripts starting with "index" are the scripts used for indexing.
- /runs contains run-files for both OpenMatch and trec-eval, both automatic queries and resolved queries.
- re-ranking_with_LTR.ipynb is the notebook used to create training data, and train a PointWiseLTR model.
