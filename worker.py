from __future__ import absolute_import, generators
from __future__ import division
from __future__ import print_function

from im2pres.detection import get_detector, get_textbox
from im2pres.utils import group_text_box, get_image_list, diff, reformat_input, tesseract, cleanName
from im2pres.spellcheck import SpellCheck

import time
import os
import gdown
import threading
import hashlib

import fuzzywuzzy
import pandas as pd
import torch

queueLock = threading.Lock()

class OCRThread (threading.Thread):
    def __init__(self, threadID, name):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.q = queue.Queue()

        self.results = {}

    def updateStatus(self, job_id, status):
        if job_id in self.results:
            self.results[job_id] = status

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
            'https://drive.google.com/uc?id=1MVjeBuTTO9prdnGBm1wyP_OV-LESwimk')
            

        os.system('mv craft_mlt_25k.pth im2pres/data/craft_mlt_25k.pth')

        DETECTOR_FILENAME = 'im2pres/data/craft_mlt_25k.pth'

        imgH = 64
        input_channel = 1
        output_channel = 512
        hidden_size = 512

        medicinePath =  'im2pres/data/main_dict/high_res_low_res.csv'

        df = pd.read_csv(medicinePath, sep=';', quotechar="\"", header=0, dtype=str)
        #df_common_name = pd.read_json(commonNamePath, orient='records')
        medicineClassifierData, medicineCorrectionData = readData()

        device = 'cpu'
        if torch.cuda.is_available():
              device = 'cuda'

        detector = get_detector(DETECTOR_FILENAME, device)

        while True:
            process_data_session(self, self.q, detector, device, df, medicineClassifierData, medicineCorrectionData, imgH)
            time.sleep(1)

        print("Exiting " + self.name)

def process_data_session(thread_task, q, detector, device, df, medicineClassifierData, medicineCorrectionData, imgH):
    data = None

    queueLock.acquire()
    if not q.empty():
        data = q.get()
    queueLock.release()

    if data:
        print(f"{thread_task.name} processing {data}...")
        # todo
        job_id, filepath = data['job_id'], data['filepath']

        thread_task.updateStatus(job_id, {'status': 'ongoing', 'result': ''})
        res = predict_task(detector, device, df, medicineClassifierData, medicineCorrectionData, imgH, filepath)

        if res:
            thread_task.updateStatus(
                job_id, {'status': 'completed', 'result': res})

    time.sleep(1)

    return None

def predict_task(detector, device, df, medicineClassifierData, medicineCorrectionData, imgH, filepath):
    
    texts = readtext(detector, device, df, medicineClassifierData, medicineCorrectionData, imgH, filepath)

    result = []
    result.append(f"Result for image {os.path.basename(filepath)}:")
    
    if len(texts) > 0:
        result.extend(texts)
    else:
        result.append("Not found medical data in your image!")
    return result

def readData(df):
    full_name = df[df.columns[0]].tolist()
    contains = df[df.columns[2]].tolist()

    singleName = [name.split() for name in full_name]
    singleContain = [contain.split() for contain in contains]

    singleName = [cleanName(item) for sublist in singleName for item in sublist]
    singleContain = [cleanName(item) for sublist in singleContain for item in sublist]

    full_name_process = [cleanName(name) for name in full_name]

    return singleName + singleContain, full_name_process

def findObj(df, name, data):
    index = data.index(name)
    return df.iloc[index].to_dict()

def readtext(detector, device, df, medicineClassifierData, medicineCorrectionData, imgH, imagePath,\
                min_size = 0, contrast_ths = 0.1, adjust_contrast = 0.5, filter_ths = 0.003,\
                text_threshold = 0.7, low_text = 0.4, link_threshold = 0.4, canvas_size = 2560,\
                mag_ratio = 1., slope_ths = 0.1, ycenter_ths = 0.5, height_ths = 0.5,\
                width_ths = 0.5, add_margin = 0.1):

    img, img_cv_grey = reformat_input(imagePath)

    text_box = get_textbox(detector, img, canvas_size, mag_ratio,\
                            text_threshold, link_threshold, low_text,\
                            False, device)
    
    horizontal_list, free_list = group_text_box(text_box, slope_ths,\
                                                ycenter_ths, height_ths,\
                                                width_ths, add_margin)

    if min_size:
        horizontal_list = [i for i in horizontal_list if max(i[1]-i[0],i[3]-i[2]) > min_size]
        free_list = [i for i in free_list if max(diff([c[0] for c in i]), diff([c[1] for c in i]))>min_size]

    
    image_list, _ = get_image_list(horizontal_list, free_list, img_cv_grey, model_height = imgH)

    croped_images = map(lambda x: x[1], image_list)

    tesseractResultArr = []

    #medicineClassifierData các tên thuốc sẽ được tách ra thành các từ cefuroxim, 500mg, (efodyl 500mg)
    #các dòng có sự xuất hiện của các từ này sẽ có khả năng cao là tên thuốc
    #xoá các kí tự đặc biệt cho dict và input
    
    medicineClassifier = SpellCheck(medicineClassifierData)

    final_result = []
    for item in croped_images:
        isExist = False
        tesseractResult = tesseract(item)
        # print('tesseractResult ' + tesseractResult)
        tesseractResult = cleanName(tesseractResult)

        tesseractResultArr.append(tesseractResult) 

        #put each line in to check func 
        # print(medicineClassifier.check(tesseractResult))
        medicine_name, isMedicine = medicineClassifier.check(tesseractResult)
        if isMedicine >= 50: #xét trường hợp tên thuốc == từ điển thì xuất ra ~ 100% matching => handle case này riêng
            #check xem đây là tên thuốc nào, nếu check 0 ra tên thuốc thì hiển thị đây là thuốc nhưng chưa có trong db

            #nếu kết quả so khớp theo proccess và kết quả so kớp theo từ ~90% thì hiển thị ra tên, còn nếu không thì sẽ hiển thị warning và các option của hệ thống
            #Check độ dài input, độ dài sửa theo từ, độ dài sửa theo dòng gần bằng nhau thì sẽ cho kết quả dạng option
            correct = fuzzywuzzy.process.extract(cleanName(tesseractResult), medicineCorrectionData, limit=5, scorer=fuzzywuzzy.fuzz.token_set_ratio)
            first_match, percent_match = correct[0]
            
            for previousResult in final_result:
                for item in previousResult:
                    if cleanName(item['line']) == first_match:
                        isExist = True

            if isExist:
                continue

            if percent_match >= 95:
                    obj = findObj(df, first_match, medicineCorrectionData)
                    print(obj)
                    final_result.append([obj])
    
    return final_result