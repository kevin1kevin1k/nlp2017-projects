#!/usr/bin/env python2

import jieba
from translation import langconv

t2s = langconv.Converter('zh-hans')
s2t = langconv.Converter('zh-hant')

def seg(line):
    s_list = t2s.convert(line.decode('utf-8'))
    s = ' '.join(jieba.cut(s_list, cut_all=True))
    t = s2t.convert(s).encode('utf-8')
    t_list = t.split(' ')
    return '|'.join(t_list)
