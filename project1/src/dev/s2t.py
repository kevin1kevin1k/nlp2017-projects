from translation import langconv

with open('../../data/ChineseStopWords.txt') as stopwords:
    for line in stopwords:
        sw = line.strip()
        s2t = langconv.Converter('zh-hant')
        sw_zh = s2t.convert(sw.decode('utf-8')).encode('utf-8')
        print sw_zh
    
