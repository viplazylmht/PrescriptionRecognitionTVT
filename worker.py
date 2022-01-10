from __future__ import absolute_import, generators
from __future__ import division
from __future__ import print_function

from im2pres.CRVOCR import CRVOCR 
from im2pres.MergeOCR import MergeOCRED 
from im2pres.DrugExtractor import ExtractDrug
from im2pres.PostOCR import PostOCR

#from im2pres.spellcheck import SpellCheck

from bson.json_util import dumps, loads

import time, timeit
import os
import threading
import hashlib

import queue
from PIL import Image

queueLock = threading.Lock()

class OCRThread (threading.Thread):
    def __init__(self, threadID, name):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.q = queue.Queue()
        self.job_id = None

        self.crvocr = None
        self.mergeOCR = None
        self.extractor = None
        self.post_ocr = None 

        self.db_url = 'https://PresMongoDB.viplazy.repl.co/api/v2/search'

        self.results = {}

    def updateStatus(self, job_id, status):
        if job_id in self.results:
            self.results[job_id] = status

            return self.results[job_id]
        else:
            return {'result': 'error', 'message': 'job not found'}
    
    def updateStatusMessage(self, job_id, statusName, message=None):
        if job_id in self.results:
            self.results[job_id]['status'] = statusName

            if message:
                self.results[job_id]['message'] = message

            return self.results[job_id]
        else:
            return {'result': 'error', 'message': 'job not found'}

    def getResult(self, job_id):
        if job_id in self.results:
            return self.results[job_id]
        else:
            return {'result': 'error', 'message': 'job not found'}

    def md5(self, fname):
        hash_md5 = hashlib.md5()
        with open(fname, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def pushJob(self, filepath):
        # return job_id
        # create job_id first
        job_id = self.md5(filepath)

        queueLock.acquire()

        self.q.put({'job_id': job_id, 'filepath': filepath})

        queueLock.release()

        self.results[job_id] = {'status': 'queued', 'result': ''}

        return job_id
    
    def run(self):
        print("Starting " + self.name)

        print("Loading CRVOCR...")
        self.crvocr = CRVOCR(output_dir=None, use_gpu=False)

        print("Loading MergeOCR...")
        self.mergeOCR = MergeOCRED(threshold=0.018)

        print("Loading DrugExtractor...")
        self.extractor = ExtractDrug()

        print("Loading PostOCR...")
        os.system("pip install keras-tcn --no-dependencies")

        self.post_ocr = PostOCR()

        while True:
            self.process_data_session()
            time.sleep(0.1)

        print("Exiting " + self.name)

    def process_data_session(self):
        data = None

        queueLock.acquire()
        if not self.q.empty():
            data = self.q.get()
        queueLock.release()

        if data:
            print(f"{self.name} processing {data}...")
            self.job_id, filepath = data['job_id'], data['filepath']

            self.updateStatus(self.job_id, {'status': 'ongoing', 'result': ''})
            res = self.predict_task(filepath)

            if res:
                self.updateStatus(
                    self.job_id, {'status': 'completed', 'result': res})

        time.sleep(1)

        return None

    def predict_task(self, filepath):
        
        r, t = self.readtext(filepath)

        result = []
        result.append(f"Result for image {os.path.basename(filepath)}:")
        
        print(r['result'])
        texts = list(set([i['drugName'] for i in r['result']]))

        if len(texts) > 0:
            result.extend(texts)
        else:
            result.append("Not found medical data in your image!")
        return result

    def readtext(self, imagePath):

        self.updateStatusMessage(self.job_id, 'handle_output', 'Detecting and Recognizing...')
        src_img = Image.open(imagePath)

        start = timeit.default_timer()
        ocred = self.crvocr.extract_text(imagePath)

        timer = {}

        lines = ocred['contents']
        timer['detection'] = ocred['timer']['detection']
        timer['recognition'] = ocred['timer']['recognition']

        self.updateStatusMessage(self.job_id, 'handle_output', 'Merging line...')

        merge_result, timer['merge_time'] = self.mergeOCR.merge(src_img, lines, verbose=False)

        self.updateStatusMessage(self.job_id, 'handle_output', 'Extracting medicine...')
        extract_result, time_taken = self.extractor.extract(merge_result, timer=True)
        timer['regex'] = time_taken

        self.updateStatusMessage(self.job_id, 'handle_output', 'Post-processing, nearly complete...')
        mid = timeit.default_timer()
        result, t = {}, {}
        
        if len(extract_result) > 0:
            result, t = self.post_ocr.search(extract_result, verbose=False, timer=True)
    
        stop = timeit.default_timer()
        
        for k, v in t.items():
            timer[k] = v
        
        timer['recorrect'] = stop - mid
        timer['total'] = stop - start

        item = {'img_name': imagePath}
    
        drugs = []
        for k, v in result.items():        
            drugs.append(v.copy())
            del drugs[-1]['_id']

            drugs[-1]['box'] = [b.tolist() for b in drugs[-1]['box']]

        item['result'] = drugs
        item['times'] = t

        open(f"tmp/{self.job_id}.log.json", 'wt').write(dumps(item, indent=4))

        return item, timer