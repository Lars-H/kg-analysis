"""
Build a bipartite graph from an n-triples Knowledge Graph representation.
"""

#!/usr/bin/python3

# TODO: Check Googles Python style guide
# TODO: Check pylint
# TODO: Remove .pack file from git history https://rtyley.github.io/bfg-repo-cleaner/

import os
import sys
import time
from importlib import import_module
from tqdm import tqdm
from hdt import HDTDocument
from rdflib import Graph, RDFS
from logger import get_time
import pandas as pd
import networkx as nx
from yaml import load, Loader

rdf = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
dbo = "http://dbpedia.org/ontology/"
dbr = "http://dbpedia.org/resource/"

# DBpedia classes: http://mappings.dbpedia.org/server/ontology/classes/

@get_time
def query_subclasses(ontology, ontology_prefix ,superclass):
    """ Query ontology for subclass rdfs-entailment """
    # Option 1: Sequential querying with pyHDT
    # Option 2: Mappings from dataset.join()
    # Option 3: https://github.com/comunica/comunica-actor-init-sparql-hdt
    subclass_query = f"""
    SELECT ?subclass
    WHERE 
    {{
        ?subclass <{str(RDFS['subClassOf'])}>* <{ontology_prefix + superclass}> .
    }}
    """
    subclasses = []
    results = ontology.query(subclass_query)
    for result in results:
        subclasses.append(str(result['subclass']))
    return subclasses

@get_time
def get_subject_predicate_tuples(dataset, subclasses, subject_limit, predicate_limit, blacklist, whitelist=None, out=True):
    """ Get edgelist for superclass and all its subclasses """
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
    subjects = list(set(subjects)) # Include unique subjects if subject is both of type superclass and subclass
    print(f"[Info] query predicates for each subject: Outgoing edges: {out}")
    for subject in tqdm(subjects):
        if out:
            pattern = (subject, "", "")
        else:
            pattern = ("", "", subject)
        if predicate_limit > 0:
            triples = dataset.search_triples(*(pattern), limit=predicate_limit)[0]
        else:
            triples = dataset.search_triples(*(pattern))[0]

        for triple in triples:
            # Either blacklist
            if not triple[1] in blacklist:
                if out:
                    edgelist.append((triple[0], triple[1]))
                else:
                    edgelist.append((triple[2], triple[1]))
            # Or whitelist
            #if triple[1] in whitelist:
            #    edgelist.append((triple[0], triple[1]))
    return list(set(edgelist)) # Exclude duplicate entity-property relations

def write_edgelist(classname, edgelist, outdir="out"):
    """ Write edgelist to csv file """
    df = pd.DataFrame(edgelist, columns=["t", "b"])
    df.to_csv(f"{outdir}/{classname}/{classname}.g.csv", index=False)

@get_time
def write_integer_edgelist(classname, edgelist, outdir="out"):
    """ Write edgelist to csv file with node labels converted to unique integers """
    df = pd.DataFrame(edgelist, columns=["t", "b"])
    df["t"] = pd.Categorical(df["t"])
    df["b"] = pd.Categorical(df["b"])
    b_offset = df["t"].nunique()
    df["t"] = df["t"].cat.codes
    df["b"] = df["b"].cat.codes + b_offset
    df.to_csv(f"{outdir}/{classname}/{classname}.i.csv", index=False)

@get_time
def check_connected(bigraph):
    """ Check whether input graph is connected and throw NetworkXPointlessConcept if null graph """
    if nx.is_connected(bigraph):
        return True
    return False

@get_time
def check_bipartite(bigraph):
    """ Check whether input graph is bipartite """
    if not nx.bipartite.is_bipartite(bigraph):
        sys.exit("[Error] Input graph is not bipartite")

@get_time
def split_edgelist(edges):
    """ Split the input edgelist into top (t) and bottom (b) nodes """
    nodes_top = []
    nodes_bot = []
    for edge in edges:
        nodes_top.append(edge[0])
        nodes_bot.append(edge[1])
    nodes_top = list(set(nodes_top))
    nodes_bot = list(set(nodes_bot))
    return nodes_top, nodes_bot

def add_results(run_name, superclass, outdir="out",**results):
    """ Append result columns in a superclass row """
    df = pd.read_csv(f"{outdir}/_results_{run_name}.csv", index_col=0)
    for resultname, result in results.items():
        df.at[superclass, resultname] = result
    df.to_csv(f"{outdir}/_results_{run_name}.csv")

@get_time
def main():
    run_fn = sys.argv[1]
    run_name = run_fn.split("/")[-1].split(".")[0]
    run = load(open(run_fn).read(), Loader=Loader)
    outdir = run['config'].get('output_dir', 'out')
    ontology_prefix = run['config'].get('ontology_prefix', dbo)
    blacklist_fn = run['config'].get('blacklist', None)
    ontology_fn = run['config'].get('kg_ontology', None)
    direction_out = run['config'].get('direction', "out") == "out"

    if not os.path.isdir(outdir):
        print(f"[Info] Creating new dir: {outdir}.")
        os.mkdir(outdir)

    if blacklist_fn:
        with open(blacklist_fn, "r") as file:
            blacklist = file.read().splitlines()
            print(f"[Info] Loaded Blacklist file: {blacklist_fn}")
    else:
        blacklist = []

    if not os.path.exists(f"{outdir}/_results_{run_name}.csv"):
        df = pd.DataFrame(columns=["n_t"])
        df.to_csv(f"{outdir}/_results_{run_name}.csv")

    for superclass in run['config']["classes"]:
        print("\n[Build] ", superclass)
        if not os.path.exists(f"./{outdir}/{superclass}"):
                os.mkdir(f"./{outdir}/{superclass}")
        tsv_files = [fn for fn in os.listdir(f"{outdir}/{superclass}/") if fn.endswith(".tsv") ]
        print(tsv_files)
        if len(tsv_files) == 1:
            # Convert .tsv file in cleaned .g.csv edgelist
            print("[Info] build from tsv edgelist")
            df = pd.read_csv(f"{outdir}/{superclass}/{tsv_files[0]}", names=["t","b"])#, sep="\t")
            print(f"[Info] len edgelist {len(df)}, t_unique {df['t'].nunique()}, b_unique {df['b'].nunique()}")
            duplicate_predicates = df["b"].str.contains("/direct/")
            df = df[~duplicate_predicates]
            print(f"[Info] len after duplicates {len(df)}, t_unique {df['t'].nunique()}, b_unique {df['b'].nunique()}")
            # Either blacklist
            for blacklisted_predicate in blacklist:
                 m_before = len(df)
                 df = df[df["b"] != blacklisted_predicate]
                 m_after = len(df)
                 if m_before != m_after:
                     print(f"[Info] edges with blacklisted predicate removed\n       {blacklisted_predicate}")
                 print(f"[Info] len after blacklist {len(df)}, t_unique {df['t'].nunique()}, b_unique {df['b'].nunique()}")
            # Or whitelist
            #df = df[df["b"].isin(whitelist)]

            edgelist = list(df.itertuples(index=False, name=None))
        else:
            # Query .hdt data to create .g.csv edgelist
            dataset = HDTDocument(run['config']["kg_source"])
            classes = [f"{ontology_prefix}{superclass}"]
            if ontology_fn:
                t_ontology = time.time()
                ontology = Graph().parse(ontology_fn)
                print(f"\n[Time] load-ontology {time.time() - t_ontology:.3f} sec")
                subclasses = query_subclasses(ontology, ontology_prefix, superclass)
                classes.extend(subclasses)
            else:
                print("[Info] No Ontology file provided")


            subject_limit = int(run['config']["subject_limit"])
            predicate_limit = int(run['config']["predicate_limit"])
            edgelist = get_subject_predicate_tuples(dataset, classes, subject_limit, predicate_limit, blacklist, out=direction_out)
        try:
            bigraph = nx.Graph()
            bigraph.add_edges_from(edgelist)
            check_connected(bigraph)
            check_bipartite(bigraph)
            nodes_top, nodes_bot = split_edgelist(edgelist)
            n_t, n_b = len(nodes_top), len(nodes_bot)
            m_g = len(edgelist)
            dens_g = m_g / (n_t * n_b)
            k_t_g = m_g / n_t
            k_b_g = m_g / n_b
            print(f"[Info] n_t {n_t}, n_b {n_b}, m_g {len(edgelist)}")
            write_edgelist(superclass, edgelist, outdir=outdir)
            write_integer_edgelist(superclass, edgelist, outdir=outdir)
            # In onemode network edgelists, information about disconnected nodes gets lost
            add_results(run_name, superclass, outdir=outdir,
                        n_t=n_t, n_b=n_b,
                        m_g=m_g, dens_g=dens_g,
                        k_t_g=k_t_g, k_b_g=k_b_g)
        except nx.NetworkXPointlessConcept as e:
            print(f"[Info] {superclass} graph is the null graph\n{e}")


if __name__ == '__main__':
    main()
