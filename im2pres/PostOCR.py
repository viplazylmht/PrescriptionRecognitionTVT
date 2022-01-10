import json, urllib, timeit, requests
from bson.json_util import dumps, loads
import pymongo

# install python-Levenshtein for up to 10x faster 
# from thefuzz import process, fuzz
from rapidfuzz import process, fuzz

from bson import json_util

class PostOCR():
    def __init__(s, FUZZ_THRESHOLD=0.8):
        s.FUZZ_THRESHOLD = FUZZ_THRESHOLD*100
        import os, re
        MURI = "mongodb+srv://prescluster.pfayl.mongodb.net/myFirstDatabase?authSource=%24external&authMechanism=MONGODB-X509&retryWrites=true&w=majority"
        os.system('wget -q https://cdn.discordapp.com/attachments/817270986423664670/907502397595717682/client_cert.pem')
        s.mclient = pymongo.MongoClient(MURI, tls=True, tlsCertificateKeyFile='client_cert.pem')
        os.system('rm client_cert.pem')
        
        mdb = s.mclient['DrugBank']

        s.medicine_collection = mdb.medicines
        s.fuzz_search_collection = mdb.fuzz_searchs
        
        # load medicineClassifer
        os.system('wget -qO- https://cdn.discordapp.com/attachments/817270986423664670/907516235384766475/MedicineClassifer.py > MedicineClassifer.py')
        from MedicineClassifer import MedicineClassifer
        os.system('rm MedicineClassifer.py')
        
        s.medicineClassifer = MedicineClassifer()
        
        clean_drugname_pattern = r"((?:[^\x00-\x7F]|\w)+(?:(?: |\/|\-|\.|\%|\')+(?:[^\x00-\x7F]|\w)+)*).*" # support unicode (vietnamese)
        clean_drugname_matcher = re.compile(clean_drugname_pattern)
        
        os.system('wget -q https://cdn.discordapp.com/attachments/817270986423664670/907511360622067743/drugname.json')
        drugs_dict = open('drugname.json').readlines()
        os.system('rm drugname.json')
        print(f"Total drug istance: {len(drugs_dict)}")
        
        s.drugnames = {}
        tmps = {}
        
        for drug in drugs_dict:
            item = json.loads(drug)
            item["tenThuoc"] = clean_drugname_matcher.findall(item["tenThuoc"])[0]
            
            if item["tenThuoc"] not in tmps:
                tmps[item["tenThuoc"]] = True
                
                s.drugnames[item['id']] = item["tenThuoc"].lower()

            # URL = f"{item['tenThuoc']}&{item['id']}"
            # item['url'] = urllib.parse.quote_plus(URL,safe="")

            # s.drugnames_search_output[item['id']] = item
        print(f"Total drug istance truncated: {len(s.drugnames)}")
        
    def search(s, drugs, use_spell_check=False, verbose=False, timer=False):
        """
            drugs: an string, or list of strings, or list of list of strings [array2D], or also list of dict is fine xD
                for example: 'vitaminn' | ['entoci', 'vitamin'] | [['vitamin', 'etoncii'], [paracetamol]] | [{'gs': ['vitamin'], 'box': any }]
            return: dicts
        """
        classifer_time, fuzz_time = 0., 0.
        
        if use_spell_check:
            print("LOG: spellcheck active on this session")
        
        results = []
        if not isinstance(drugs, list):
            drugs = [drugs, ]
        if not isinstance(drugs[0], list) and not isinstance(drugs[0], dict):
            drugs = [drugs, ]
            
        # now drugs is array2D
        for line in drugs:
            result = {}
            bbox = None
            if isinstance(line, dict):
                if 'box' in line:
                    bbox = line['box']
                line = line['gs']
                if not isinstance(line, list):
                    line = [line, ]

            start = timeit.default_timer()
            mc = s.medicineClassifer.predict(line)
            mid = timeit.default_timer()
            classifer_time += (mid - start)
            
            for drug, scores in zip(line, mc):

                non_drug_score, medicine_score, ingredient_score = scores
                # check if it is medicine first if required 
                if use_spell_check:
                    print("DEBUG scores:", drug, ':', scores)
                    #if medicine_score < 0.7:
                        # skip this drug due to not enough score to be a medicine 
                    #    continue

                    if ingredient_score < 0.5 and medicine_score < 0.5:
                        continue

                # if medicine_score >= 70 or skipped checker, we perform correct task now
                fuz_term = s.fuzz_search_collection.find_one({'fuzz_term' : drug})

                if fuz_term is None:
                    # new_entrie = process.extractOne(drug, drugnames, scorer=fuzz.token_sort_ratio)
                    new_entrie = process.extractOne(drug, s.drugnames, scorer=fuzz.token_set_ratio)
                    if verbose:
                        print(new_entrie)
                    # result[drug] = new_entrie   
                    if new_entrie[1] > s.FUZZ_THRESHOLD:

                        # save fuzz result
                        oid = s.medicine_collection.find_one({'id' : new_entrie[-1]}, {'_id' : 1, 'tenThuoc' : 1,})

                        s.fuzz_search_collection.insert_one({
                            "fuzz_term": drug,
                            "target_id": oid['_id'],
                            "score": new_entrie[1],
                        })

                        result[drug] = { '_id': oid['_id'], 'drugName': oid['tenThuoc'],
                            'fuzz_score': new_entrie[1],
                            'medicine_score': float(medicine_score), 'ingredient_score': float(ingredient_score),
                        }  
                        
                        if bbox is not None:
                            result[drug]['box'] = bbox

                else:
                    # retrive result
                    if verbose:
                        print(fuz_term)
                        
                    if fuz_term['score'] > s.FUZZ_THRESHOLD:

                        oid = s.medicine_collection.find_one({'_id' : fuz_term['target_id']}, {'tenThuoc' : 1,})

                        result[drug] = { '_id': fuz_term['target_id'], 'drugName': oid['tenThuoc'],
                            'fuzz_score': fuz_term['score'],
                            'medicine_score': float(medicine_score), 'ingredient_score': float(ingredient_score),
                        }  
                        
                        if bbox is not None:
                            result[drug]['box'] = bbox
            
            results.append(result)
            
            end = timeit.default_timer()
            fuzz_time += (end - mid)
        
        result = {}
        if verbose:
            print(results)
            
        for item in results:
            if len(item) > 0:
                #print(item)
                best = None
                best_mscore = 0.
                for k,v in item.items():
                    if v['fuzz_score'] > 80 and v['medicine_score'] > best_mscore:
                        best_mscore = v['medicine_score']
                        best = k
                if best is not None:
                    result[best] = item[best]
        
        if timer:
            return result, {'classifer_time': classifer_time, 'fuzz_time': fuzz_time}
        return result