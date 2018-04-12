import csv
import os

def doc_topic(in_path, out_path):
    with open(in_path) as file:
        text = file.readlines()
        output = []
        for line in text:
            output.append(line.split())
    with open(out_path, 'w') as file:
        table = csv.writer(file)
        table.writerows(output)

def word_topic(in_path, out_path):
    with open(in_path) as file:
        text = file.readlines()
        tlist = []
        for line in text:
            tlist.append(line.split())
        # 二维列表转置https://www.zhihu.com/question/39660985
        output = map(list, zip(*tlist))
    with open(out_path, 'w') as file:
        table = csv.writer(file)
        table.writerows(output)

if __name__=="__main__":

    ldahome = './data/embedding/data_lda/data_abstract'
    dirnames = [name for name in os.listdir(ldahome)
                if os.path.isdir(os.path.join(ldahome, name))]

    for dir in dirnames:
        doc_topic_file = os.path.join(ldahome, dir, 'model-final.theta')
        doc_topic_csv = os.path.join(ldahome, dir, 'doc_topic.csv')
        doc_topic(doc_topic_file, doc_topic_csv)

        word_topic_file = os.path.join(ldahome, dir, 'model-final.phi')
        word_topic_csv = os.path.join(ldahome, dir, 'word_topic.csv')
        word_topic(word_topic_file, word_topic_csv)