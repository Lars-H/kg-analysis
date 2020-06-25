from hdt import HDTDocument
from tqdm import tqdm
import sys
import json

def extract_classes(fn):
    doc = HDTDocument(fn)

    rdf_type = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
    types = set()
    types_dct = {}
    (triples, count) = doc.search_triples("", rdf_type, "")

    for triple in tqdm(triples, total=count):
        types.add(triple[2])

    for type in tqdm(types):
        (instances, instance_count) = doc.search_triples("", rdf_type, type)
        types_dct[type] = instance_count

    return types_dct

if __name__ == '__main__':
    fn = sys.argv[-1]
    types_dct = extract_classes(fn)
    dataset_name = fn.split("/")[-1].split(".")[0]
    with open(f"{dataset_name}_types.json", "w") as outfile:
        json.dump(types_dct, outfile)

