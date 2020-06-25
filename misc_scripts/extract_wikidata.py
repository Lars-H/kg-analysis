from hdt import HDTDocument
import csv
from tqdm import tqdm

def extract_by_instance(fn, wdt_class, property, out=True):

    doc = HDTDocument(fn)

    wd = "http://www.wikidata.org/entity/"
    wdt = "http://www.wikidata.org/prop/direct/"

    properties = {
        "instance_of" :  "P31",
        "occupation" : "P106"
    }

    instances = set()



    (triples, count) = doc.search_triples("", f"{wdt}{properties[property]}", f"{wd}{wdt_class}")

    for triple in tqdm(triples, len(list(instances))):
        instances.add(triple[0])

    with open(f'{wdt_class}.csv', "w") as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')

        for instance in tqdm(instances, total=len(instances)):
            if out:
                pattern = (instance, "","")
            else:
                pattern = ("", "", instance)

            (triples, count) = doc.search_triples(*pattern)

            for triple in triples:
                if out:
                    spamwriter.writerow([triple[0], triple[1]])
                else:
                    spamwriter.writerow([triple[2], triple[1]])



if __name__ == '__main__':

    fn = "kg/wikidata-20170313-all-BETA.hdt"
    wdt_classes = {
        "Country" : "Q6256",
        "Boxer" : "Q11338576",
        "FilmFestival" : "Q220505"
    }

    extract_by_instance(fn, wdt_classes['Boxer'], "occupation", out=False)

