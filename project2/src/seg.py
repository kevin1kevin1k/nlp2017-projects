#!/usr/bin/env python2

import jieba
from translation import langconv
import argparse
import re

t2s = langconv.Converter('zh-hans')
s2t = langconv.Converter('zh-hant')


def seg(line):
    s_list = t2s.convert(line.decode('utf-8'))
    # TODO: cut_all=True (more segment outputs, but may be noisy)
    s = ' '.join(jieba.cut(s_list, cut_all=False))
    t = s2t.convert(s).encode('utf-8')
    t_list = t.split(' ')
    multi_pipes = '|'.join(t_list)
    return re.sub(r'\|+', '|', multi_pipes)


def get_config():
    parser = argparse.ArgumentParser(description='word segmentation')
    parser.add_argument('input_path',
                        help='path of input file')
    parser.add_argument('-o', '--output_path', default='segmented.txt',
                        help='path of output file')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    config = get_config()
    print config
    
    with open(config.input_path, 'r') as input_f, open(config.output_path, 'w') as output_f:
        for i, line in enumerate(input_f):
            # print i, line
            if i == 0:
                output_f.write(line)
                continue
            if 'train' in config.input_path:
                id_, clause1, clause2, relation = line.strip().split(',')
            else:
                id_, clause1, clause2 = line.strip().split(',')
            
            seg1, seg2 = seg(clause1), seg(clause2)
            
            # TODO: filter out stop words
            
            # for s in (seg1, seg2):
            #     if s.startswith('|') or s.endswith('|'):
            #         print s
            
            if 'train' in config.input_path:
                output_f.write(','.join([id_, seg1, seg2, relation]) + '\n')
            else:
                output_f.write(','.join([id_, seg1, seg2]) + '\n')

    