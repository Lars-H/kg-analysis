"""
Build a bipartite graph from an n-triples Knowledge Graph representation.
"""

#!/usr/bin/python3

# TODO: Check Googles Python style guide
# TODO: Check pylint

import os
import sys
import time
from importlib import import_module
from tqdm import tqdm
from hdt import HDTDocument
from rdflib import Graph, RDFS
import pandas as pd

rdf = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
dbo = "http://dbpedia.org/ontology/"
dbr = "http://dbpedia.org/resource/"

# DBpedia classes: http://mappings.dbpedia.org/server/ontology/classes/

def query_subclasses(ontology, superclass):
    """ Query ontology for subclass rdfs-entailment """
    # Option 1: Sequential querying with pyHDT
    # Option 2: Mappings from dataset.join()
    # Option 3: https://github.com/comunica/comunica-actor-init-sparql-hdt
    t_start = time.time()
    subclass_query = f"""
    SELECT ?subclass
    WHERE 
    {{
        ?subclass <{str(RDFS['subClassOf'])}>* <{dbo + superclass}> .
    }}
    """
    subclasses = []
    results = ontology.query(subclass_query)
    for result in results:
        subclasses.append(str(result['subclass']))
    print(f"[Time] query-subclasses {time.time() - t_start:.3f} sec")
    return subclasses

def get_subject_predicate_tuples(dataset, subclasses, subject_limit, predicate_limit, blacklist):
    """ Get edgelist for a superclass and all its subclasses """
    t_start = time.time()
    subjects = []
    edgelist = []
    print("[Info] query subjects for each subclass")
    for subclass in tqdm(subclasses):
        if subject_limit > 0:
            triples = dataset.search_triples("", rdf + "type", subclass, limit=subject_limit)[0]
        else:
            triples = dataset.search_triples("", rdf + "type", subclass)[0]
        for triple in triples:
            subjects.append(triple[0])
    print("[Info] query predicates for each subject")
    for subject in tqdm(subjects):
        if predicate_limit > 0:
            triples = dataset.search_triples(subject, "", "", limit=predicate_limit)[0]
        else:
            triples = dataset.search_triples(subject, "", "")[0]
        for triple in triples:
            if not triple[1] in blacklist:
                edgelist.append((triple[0], triple[1]))
    print(f"[Time] get-subj-pred-tuples {time.time() - t_start:.3f} sec")
    return edgelist

def write_edgelist(classname, edgelist):
    """ Write edgelist to csv file """
    df = pd.DataFrame(edgelist, columns=["t", "b"])
    df.to_csv(f"out/{classname}.g.csv", index=False)

def add_results(run_name, superclass, **results):
    """ Append result columns in a superclass row """
    df = pd.read_csv(f"out/_results_{run_name}.csv")
    for resultname, result in results.items():
        df.loc[superclass, resultname] = result
    df.to_csv(f"out/_results_{run_name}.csv")

def main():
    run_name = sys.argv[1][:-3]
    run = import_module(run_name)
    dataset = HDTDocument(run.config["kg_source"])
    t_ontology = time.time()
    ontology = Graph().parse(run.config["kg_ontology"])
    print(f"\n[Time] load-ontology {time.time() - t_ontology:.3f} sec")
    subject_limit = run.config["subject_limit"]
    predicate_limit = run.config["predicate_limit"]
    with open("blacklist.txt", "r") as file:
        blacklist = file.read().splitlines()
    if os.path.exists(f"out/_results_{run_name}.csv"):
        print("[Info] Remove old results file")
        os.remove(f"out/_results_{run_name}.csv")
    results = pd.DataFrame(columns=["subclasses", "m"])
    results.to_csv(f"out/_results_{run_name}.csv", index=False)

    t_build = time.time()
    for superclass in run.config["classes"]:
        print("\n[Build] ", superclass)
        subclasses = query_subclasses(ontology, superclass)
        edgelist = get_subject_predicate_tuples(dataset, subclasses, subject_limit, predicate_limit, blacklist)
        write_edgelist(superclass, edgelist)
        add_results(run_name, superclass, subclasses=subclasses, m=len(edgelist))
        print(f"[Time] build-edgelists {time.time() - t_build:.3f} sec")

main()
