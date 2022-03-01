def check_dna(s):
    for ch in s:
        if ch != 'A' and ch != 'T' and ch != 'C' and ch != 'G':
            return False
    return True



with open('../raw/hg19.fa', 'r', encoding = 'utf-8') as fin:
    with open('../hg19', 'w', encoding = 'utf-8') as fout:
        new_line = 1
        for line in fin:
            line = line.strip().upper()
            if check_dna(line) == False:
                if new_line == 0:
                    fout.write('\n')
                    new_line = 1
            else:
                fout.write(line)
                new_line = 0