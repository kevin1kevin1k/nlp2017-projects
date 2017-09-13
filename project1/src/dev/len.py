with open('../../data/polarity_review_seg_X.txt') as seg_file:
    for line in seg_file:
        print line.strip().replace('|', '')