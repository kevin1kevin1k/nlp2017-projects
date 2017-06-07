# encoding: utf-8

import jieba
from translation import langconv
import argparse
import re
import CKIPParserClient
import os

t2s = langconv.Converter('zh-hans')
s2t = langconv.Converter('zh-hant')


def seg(line):
    s_list = t2s.convert(line.decode('utf-8'))
    s = ' '.join(jieba.cut(s_list, cut_all=False))
    t = s2t.convert(s).encode('utf-8')
    t_list = t.split(' ')
    multi_pipes = '|'.join(t_list)
    return re.sub(r'\|+', '|', multi_pipes)


def seg_CKIP(s, verbose=False):
    options = {
        'divide': 300,
        'encoding': 'UTF-8',
        'pos': False,
        'server': '140.109.19.130',
        'port': 8002,
        'xml': False,
    }

    input_filename = 'tmp_input.txt'
    output_filename = 'tmp_output.txt'
    uwfile = None

    with open(input_filename, 'w') as input_f:
        input_f.write(s.decode('utf-8').encode('big5'))

    srv = CKIPParserClient.CkipSrv('ckip', 'ckip', server=options['server'], port=options['port'])
    srv.segFile(input_filename, output_filename, uwfile, options)

    whole = ''
    with open(output_filename, 'r') as output_f:
        for line in output_f:
            result = line.decode('big5').encode('utf-8')
            result = re.sub(r'([a-zA-Z0-9.:_#\[\]\(\)\|]|\s|•...)+', ' ', result)
            if verbose:
                print result.strip(),
            whole += result.strip() + ' '
        if verbose:
            print
    
    os.remove(input_filename)
    os.remove(output_filename)
    
    return whole.strip()


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
            
            try:
                seg1, seg2 = seg_CKIP(clause1, verbose=True), seg_CKIP(clause2, verbose=True)
            except:
                print 'Error: %s,%s,%s' % (id_, clause1, clause2)
                continue
            
            # TODO: filter out stop words
            
            seg1 = seg1.strip('|')
            seg2 = seg2.strip('|')
            
            # for s in (seg1, seg2):
            #     if s.startswith('|') or s.endswith('|'):
            #         print s
            
            if 'train' in config.input_path:
                output_f.write(','.join([id_, seg1, seg2, relation]) + '\n')
            else:
                output_f.write(','.join([id_, seg1, seg2]) + '\n')
            
            print 'line %d done' % i
    