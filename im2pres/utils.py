from __future__ import print_function

import time, re
from PIL import Image

def crop_image(img, box, paddings=(1., 1., 5., 5.)):
    # paddings is in float
    
    l, t = box[0]
    r, bt = box[1]

    t_min = min(box[0][1], box[1][1], box[2][1], box[3][1])
    t_max = max(box[0][1], box[1][1], box[2][1], box[3][1])
    # them padding vao day (pixel)
    (pl, pr, pt, pb) = paddings

    return img.crop((l-pl, t_min-pt, r+pr, t_max+pb))    
    
def simplify_bboxes(boxes):
    #return boxes[:, [0,2]]
    return boxes

def craft_time(craft, img_path):
    start_time = time.time()
    content = craft.detect_text(img_path)
    running_time = time.time()-start_time
    return running_time, content['boxes'] # content['text_crop_paths']

def vietocr_time(vietocr, craft, img_path):

    img = Image.open(img_path)

    det_time, img_boxes = craft_time(craft, img_path)

    start_time = time.time()

    contents = []
    for box in simplify_bboxes(img_boxes):
        new_img = crop_image(img, box)
        w, h = new_img.size
        if w != 0 and h != 0:
            contents.append({'line': vietocr.predict(new_img), 'box': box})
            
    reg_time = time.time()-start_time

    return [det_time, reg_time], contents

def cleanName(name):
    #cần kiểm tra xem hàm này đã xoá các kí tự tiếng việt chưa ví dụ ê, ư, ó, ò, é nếu chưa cần gọi thêm hàm depunc
    dePunc(name) #thay thế các kí tự tiếng việt thành kí tự tiếng a tương ứng
    # print(name)
    string = re.sub(r'[^a-z0-9]', ' ', name.lower()) 

    string = [x.strip() for x in string.split() if len(x) > 2]
   
    return  ' '.join(string)

def dePunc(string):
        tmp = ''
        for text in string:
            if text in 'aáàảãạăắằẳẵặâấầẩẫậ':
                tmp += 'a'
            elif text in 'eéèẻẽẹêếềểễệ':
                tmp += 'e'
            elif text in 'iíìỉĩị':
                tmp += 'i'
            elif text in 'IÍÌỈĨỊ':
                tmp += 'I'
            elif text in 'yýỳỷỹỵ':
                tmp += 'y'
            elif text in 'AÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬ':
                tmp += 'A'
            elif text in 'EÉÈẺẼẸÊẾỀỂỄỆ':
                tmp += 'E'
            elif text in 'OÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢ':
                tmp += 'O'
            elif text in 'oóòỏõọôốồổỗộơớờởỡợ':
                tmp += 'o'
            elif text in 'UÚÙỦŨỤƯỨỪỬỮỰ':
                tmp += 'U'
            elif text in 'uúùủũụưứừửữự':
                tmp += 'u'
            elif text in 'YÝỲỶỸỴ':
                tmp += 'Y'
            elif text in 'dđ':
                tmp += 'd'
            elif text in 'DĐ':
                tmp += 'D'
            elif text in '(':
                tmp += ' ('
            elif text in ')':
                tmp +=  ') '
            elif text in '/':
                tmp += ' / '
            else:
                tmp += text
        # if tmp[-1] == ' ':
        #     tmp = tmp[:-1]
        return tmp

