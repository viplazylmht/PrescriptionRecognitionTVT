from __future__ import absolute_import, generators
from __future__ import division
from __future__ import print_function

from im2pres.detection import get_detector, get_textbox
from im2pres.utils import group_text_box, get_image_list, diff, reformat_input, tesseract, cleanName
from im2pres.spellcheck import SpellCheck

from bson.json_util import dumps, loads

import time, timeit
import os
import gdown
import threading
import hashlib
import requests

from fuzzywuzzy import fuzz, process as fuzzprocess
import pandas as pd
import torch
import queue

queueLock = threading.Lock()

class OCRThread (threading.Thread):
    def __init__(self, threadID, name):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.q = queue.Queue()
        self.job_id = None

        self.device = None
        self.detector = None

        self.DETECTOR_FILENAME = 'im2pres/data/craft_mlt_25k.pth'
        self.db_url = 'https://PresMongoDB.viplazy.repl.co/api/v2/search'

        self.imgH = 64
        self.input_channel = 1
        self.output_channel = 512
        self.hidden_size = 512

        self.medicinePath =  'im2pres/data/main_dict/drugbank_data.csv'

        self.df = None
        self.medicineClassifierData, self.medicineCorrectionData = None, None
        self.medicineClassifier = None

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

        print("Downloading model parameters")
        gdown.download(
            'https://drive.google.com/uc?id=1iPJ99JdU-CRYWm2iLuA7XoxX9H382YKr')
            
        os.system('mv craft_mlt_25k.pth im2pres/data/craft_mlt_25k.pth')

        self.df = pd.read_csv(self.medicinePath, quotechar="\"", header=0, dtype=str)
        #df_common_name = pd.read_json(commonNamePath, orient='records')
        self.medicineClassifierData, self.medicineCorrectionData = self.readData()

        #medicineClassifierData các tên thuốc sẽ được tách ra thành các từ cefuroxim, 500mg, (efodyl 500mg)
        #các dòng có sự xuất hiện của các từ này sẽ có khả năng cao là tên thuốc
        #xoá các kí tự đặc biệt cho dict và input
        
        self.medicineClassifier = SpellCheck(self.medicineClassifierData)

        self.device = 'cpu'
        if torch.cuda.is_available():
              self.device = 'cuda'

        self.detector = get_detector(self.DETECTOR_FILENAME, self.device)

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
            # todo
            self.job_id, filepath = data['job_id'], data['filepath']

            self.updateStatus(self.job_id, {'status': 'ongoing', 'result': ''})
            res = self.predict_task(filepath)

            if res:
                self.updateStatus(
                    self.job_id, {'status': 'completed', 'result': res})

        time.sleep(1)

        return None

    def predict_task(self, filepath):
        
        texts = self.readtext(filepath)

        result = []
        result.append(f"Result for image {os.path.basename(filepath)}:")
        
        if len(texts) > 0:
            result.extend(texts)
        else:
            result.append("Not found medical data in your image!")
        return result

    def readData(self):
        full_name = self.df[self.df.columns[0]].tolist()
        contains = self.df[self.df.columns[2]].tolist()

        singleName = [name.split() for name in full_name]
        singleContain = [contain.split() for contain in contains]

        singleName = [cleanName(item) for sublist in singleName for item in sublist]
        singleContain = [cleanName(item) for sublist in singleContain for item in sublist]

        full_name_process = [cleanName(name) for name in full_name]

        return singleName + singleContain, full_name_process

    def findObj(self, name, data):
        index = data.index(name)
        return self.df.iloc[index].to_dict()

    def readtext(self, imagePath,\
                    min_size = 0, contrast_ths = 0.1, adjust_contrast = 0.5, filter_ths = 0.003,\
                    text_threshold = 0.7, low_text = 0.4, link_threshold = 0.4, canvas_size = 2560,\
                    mag_ratio = 1., slope_ths = 0.1, ycenter_ths = 0.5, height_ths = 0.5,\
                    width_ths = 0.5, add_margin = 0.1):

        self.updateStatusMessage(self.job_id, 'handle_output', 'Detecting...')

        img, img_cv_grey = reformat_input(imagePath)

        text_box = get_textbox(self.detector, img, canvas_size, mag_ratio,\
                                text_threshold, link_threshold, low_text,\
                                False, self.device)
        
        horizontal_list, free_list = group_text_box(text_box, slope_ths,\
                                                    ycenter_ths, height_ths,\
                                                    width_ths, add_margin)

        if min_size:
            horizontal_list = [i for i in horizontal_list if max(i[1]-i[0],i[3]-i[2]) > min_size]
            free_list = [i for i in free_list if max(diff([c[0] for c in i]), diff([c[1] for c in i]))>min_size]

        self.updateStatusMessage(self.job_id, 'handle_output', 'Getting croped image list...')
        image_list, _ = get_image_list(horizontal_list, free_list, img_cv_grey, model_height = self.imgH)

        croped_images = list(map(lambda x: x[1], image_list))

        tesseractResultArr = []

        self.updateStatusMessage(self.job_id, 'handle_output', 'Recognizing...')

        final_result = []
        count, total = 0, len(croped_images)

        for item in croped_images:
            isExist = False
            count += 1
            self.updateStatusMessage(self.job_id, 'handle_output', f"Recognizing...{count}/{total}")

            tesseractResult = tesseract(item)
            # print('tesseractResult ' + tesseractResult)
            tesseractResult = cleanName(tesseractResult)

            tesseractResultArr.append(tesseractResult) 

        # remove duplicates and filtered any line with len < 3 (there is no idea to deal w/ this pres name, 3 is optional)
        fixedTessResults = list(filter(lambda x: len(x) > 3, set(tesseractResultArr)))
        
        self.updateStatusMessage(self.job_id, 'handle_output', f"POST-Processing...")

        headers = {"S-Token": "presmongo", "extra_spell_check": "true"}

        start = timeit.default_timer()
        resp = requests.post(self.db_url, json=fixedTessResults, headers=headers)
        stop = timeit.default_timer()

        rdata = loads(resp.text)
        print(rdata)

        for k, v in rdata.items():
            final_result.append([ {k :v[1:]},])

        print('LOG: db request time: ', stop - start)
        
        with open(f"tmp/{self.job_id}.tesslog", 'wt') as f:
            f.write("\n".join(fixedTessResults))

        return final_result