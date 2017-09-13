with open('../../data/aspect_review_seg_X.txt') as seg_file, open('../../data/aspect_review.txt') as all_file, open('../../data/aspect_review_seg.txt', 'w') as out_file:
    all_lines = all_file.readlines()
    # print(len(all_lines))
    i = 0
    for line in seg_file:
        review = line#.strip()
        out_file.write(all_lines[4*i])
        out_file.write(review)
        out_file.write(all_lines[4*i+2])
        out_file.write(all_lines[4*i+3])
        
        i += 1