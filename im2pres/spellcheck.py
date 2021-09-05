from fuzzywuzzy import fuzz
import re

THRES_HOLD = 75

class SpellCheck:
    def __init__(self, data):
        for i in data:
            i.lower()
            i.rstrip()

        # remove all the duplicates in the list
        data = set(data)
        
        self.dictionary = list(data)

    #Check is medicine
    # def isMedicine(self):
    #     result, score = self.check(self.string_to_check)
    #     if score >= 0.5:
    #         return True, result
    #     else:
    #         return False, ''

    def check(self, string_to_check):
        string_words = string_to_check.split(" ")
        string_len = 0
        
        words_percent = []
        for i in range(len(string_words)):
            #nếu chữ là dạng 500MG ...... thì sẽ cộng vào ~40 như tên thuốc

            max_percent = 0
            temp_name = ""
            ignore_symbol = ['lan', 'uong', 'moi', 'sang', 'trua', 'chieu', 'vien', 'sau', 'ngay']
            if len(string_words[i]) > 2:
                string_len += 1
                
                for name in self.dictionary:
                    percent = fuzz.token_set_ratio(string_words[i].lower(), name.lower())
                    if percent >= THRES_HOLD:
                        if percent > max_percent:
                            temp_name = name
                            max_percent = percent

            else:
                string_len += 1
                words_percent.append(0)
           
            if (max_percent >= THRES_HOLD):
                # print(max_percent, '\t', temp_name, string_words[i])
                if max_percent == 100:
                    string_words[i] = temp_name
                    words_percent.append(140)
                elif string_words[i] in ignore_symbol:
                    words_percent.append(-20)
                else:
                    string_words[i] = temp_name
                    words_percent.append(max_percent)
                    
        if string_len > 0 and string_words[0].isalpha():
            total_percent = sum(words_percent) / string_len
            # print(" ".join(string_words), '\t',  total_percent)
            return " ".join(string_words), total_percent
        else:
            return '', 0.0

    # this method returns the possible suggestions of the correct words
    def suggestions(self):
        string_words = self.string_to_check.split()

        suggestions = []

        for i in range(len(string_words)):
            for name in self.dictionary:
                if fuzz.ratio(string_words[i].lower(), name.lower()) >= 75:
                    print('confidence: ' + str(fuzz.ratio(string_words[i].lower(), name.lower())))
                    suggestions.append(name)

        return suggestions
  
    # def correct(self, string_to_check):
    #     string_len = 0

    #     # correction_count = 0.0  
    #     # for i in range(len(string_words)):
    #     #     max_percent = 0
    #     #     temp_name = ""
    #     #     ignore_symbol = ['mg', 'ml', '%']
    #     #     if string_words[i].isalpha() and (string_words[i] not in ignore_symbol):
    #     #         string_len += 1
    #     # for name in self.dictionary:
    #     # process.extract(str(string_words), self.dictionary, limit=10, scorer=fuzz.token_set_ratio)
    #     return process.extract(str(string_words), self.dictionary, limit=10)

  
    
    
    def preproccess_string(self, text):
        #replace vietnamese character dePunc
        
        #clear special character
        res = re.sub('[^A-Za-z0-9]+', '', text.lower())

        res = [x.strip() for x in res.split(',')]
        res = "".join(res)
        res = res.strip()

        # print("preproccess_string(self, text)")
        # print(list(res.split(" ")))
        return list(res.split(" "))
