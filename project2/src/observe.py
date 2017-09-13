
SPLIT_SYMBOL = ' '
def get_max_min_number_of_words_in_a_line():
    for type_ in ['train', 'test']:
        Mx = 0
        mx = 1000
        Mx2 = 0
        mx2 = 1000
        filename = '../data/%s_seg_CKIP.txt' % type_
        with open(filename) as f:
            for i, line in enumerate(f):
                if i == 0:
                    continue
                clauses = line.split(',')[1:3]
                for line in clauses:
                    if line.count(SPLIT_SYMBOL) > Mx:
                        Mx = line.count(SPLIT_SYMBOL)
                        Ml = line.strip()
                    if line.count(SPLIT_SYMBOL) < mx:
                        mx = line.count(SPLIT_SYMBOL)
                        ml = line.strip()
                if SPLIT_SYMBOL.join(clauses).count(SPLIT_SYMBOL) > Mx2:
                    Mx2 = SPLIT_SYMBOL.join(clauses).count(SPLIT_SYMBOL)
                    Ml2 = SPLIT_SYMBOL.join(clauses).strip()
                if SPLIT_SYMBOL.join(clauses).count(SPLIT_SYMBOL) < mx2:
                    mx2 = SPLIT_SYMBOL.join(clauses).count(SPLIT_SYMBOL)
                    ml2 = SPLIT_SYMBOL.join(clauses).strip()
                
        print filename
        print 'single clause:'
        print Mx, Ml
        print mx, ml
        print 'combined clause:'
        print Mx2, Ml2
        print mx2, ml2
        print
        

def get_relations_distribution(filename='../data/train.csv'):
    cnt = {}
    with open(filename) as f:
        for line in f:
            relation = line.strip().split(',')[-1]
            if relation == 'Relation':
                continue
            if relation not in cnt:
                cnt[relation] = 0
            cnt[relation] += 1
    
    s = sum(cnt.values())
    for relation in cnt:
        print relation, cnt[relation], cnt[relation] * 1.0 / s
    
    
def get_frequent_words_of_each_relation(filename='../data/train_seg_CKIP.txt'):
    relation2word2count = {}
    word2count_all = {}
    with open(filename) as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            id_, clause1, clause2, relation = line.strip().split(',')
            if relation not in relation2word2count:
                relation2word2count[relation] = {}
            for clause in [clause1, clause2]:
                for word in clause.split(' '):
                    if word not in relation2word2count[relation]:
                        relation2word2count[relation][word] = 0
                    relation2word2count[relation][word] += 1
                    
                    if word not in word2count_all:
                        word2count_all[word] = 0
                    word2count_all[word] += 1
    
    
    count2words_all = {}
    for w, c in word2count_all.items():
        if c not in count2words_all:
            count2words_all[c] = []
        count2words_all[c].append(w)
    counts = list(reversed(sorted(count2words_all.keys())))
    for c in counts[:22]:
        print c, ' '.join(count2words_all[c])
        
        for word in count2words_all[c]:
            for relation in relation2word2count:
                if word in relation2word2count[relation]:
                    del relation2word2count[relation][word]
    print
    
    for relation, w2c in relation2word2count.items():
        count2words = {}
        for w, c in w2c.items():
            if c not in count2words:
                count2words[c] = []
            count2words[c].append(w)
        counts = list(reversed(sorted(count2words.keys())))
        
        print relation
        for c in counts[:20]:
            print c, ' '.join(count2words[c])
            if len(count2words[c]) > 10:
                break
        print
    
    
    
if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        get_relations_distribution(filename)
    else:
        get_relations_distribution()
    
    # get_max_min_number_of_words_in_a_line()

    # get_frequent_words_of_each_relation()
