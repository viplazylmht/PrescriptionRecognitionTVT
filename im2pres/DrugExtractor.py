import re, timeit
class ExtractDrug:
    def __init__(s):
        #s.f_match_1 = re.compile(r"([0-9]*?\.+ *)(?:(.*?)\((.*?)\)*|.*?|\(.*?\))")
        s.f_match_2 = re.compile(r"^([0-9]+)(?:\.|\,)* *(?:(.*?) *\((.*?)\)*|(.*?) *\((.*)|(.*)|(\(.*?\)))", re.MULTILINE) # start with number
        s.f_match_3 = re.compile(r"^(?:(?!(?:\(| ))(.+?) *\(+([^)\n\r]+)\)*)", re.MULTILINE) # start without number, but fllow format "G1 (G2)"
        
        s.stop_words = ['gói', 'viên', 'lần', 'cách', 'ngày', 'vien', 'ngay', 'cach', 'goi', 'lan', 'sáng', 'Sáng', ]
    def extract(s, lines, timer=False):
        results = []
        start = timeit.default_timer()
        
        for line in lines:
            for f_match in [s.f_match_2, s.f_match_3]:
                matcher = f_match.findall(line['line'])
            
                if matcher is not None and len(matcher) > 0:
                    l = list(filter(lambda x: not x.strip().isdigit() and len(x) > 3 and x.strip().lower() not in s.stop_words, matcher[0]))
                    if len(l) > 0:
                        results.append({'gs': l, 'box': line['box']})
                    break # we done w/ this line
        
        stop = timeit.default_timer()
        if timer:
            return results, (stop - start) 
        return results