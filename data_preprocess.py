from tqdm import tqdm

def process(src, des):
    try:
        i, res, t = 0, [], {}
        for i, line in enumerate(tqdm(open(src, encoding="utf8").readlines())):
            if line == "\n":
                res.append(line)
                continue
            word, tag = line.split()
            new_item = word + " " + tag + "\n"
            if tag != "O":
                if not t:
                    t[tag] = i
                    new_item = word + " B-" + tag + "\n"
                elif tag in t:
                    new_item = word + " I-" + tag + "\n"
                elif t and tag not in t:
                    t = {}
                    t[tag] = i
                    new_item = word + " B-" + tag + "\n"
            res.append(new_item)
    except Exception as e:
        print(e)
    with open(des, "w", encoding="utf8") as fout:
        fout.write(''.join(res))
    a=1


if __name__ == "__main__":
    process("../rnn_crf/ner_data/train_ner", "data/data/train.txt")
    process("../rnn_crf/ner_data/test_ner", "data/data/test.txt")
    pass