import json

def eval_ner_cluner(inputfile):
    X, Y, Z = 1e-10, 1e-10, 1e-10
    X_e = 1e-10
    with open(inputfile,'r',encoding='utf-8') as f:
        for line in f:
            line = json.loads(line)
            labels = eval(line['labels'])
            pred = line['predict']
            try:
                pred = eval(pred)
            except:
                pred = {}
                pass
            
            for key in labels:
                Z += len(labels[key])

            for key in pred:
                Y += len(pred[key])

            for k1 in labels:
                for k2 in pred:
                    if k1==k2 and len(labels[k1])>=1:
                        t = [l for l in labels[k1] if l in pred[k2]]
                        X += len(t)

                        #考虑soft评价，基于字符级的f1值计算
                        for mk2 in pred[k2]:
                            score = [0]
                            for mk1 in labels[k1]:
                                char_f1 = 2 * (len(set(mk2) & set(mk1))) / (len(mk2) + len(mk1))
                                score.append(char_f1)
                            X_e += max(score)




    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
    e_f1, e_precision, e_recall = 2 * X_e / (Y + Z), X_e / Y, X_e / Z
    print("ner hard metrics:  f1: %.4f, precision: %.4f, recall: %.4f" % (f1, precision, recall))
    print("ner soft metrics:  f1: %.4f, precision: %.4f, recall: %.4f" % (e_f1, e_precision, e_recall))


def eval_ner_cluner4(inputfile):
    X, Y, Z = 1e-10, 1e-10, 1e-10
    labels_list = []
    c_data = []
    def extract_ner(text):
        res = {}
        p1 = re.compile(r'<name>(.*?)</name>')
        p2 = re.compile(r'<organization>(.*?)</organization>')
        p3 = re.compile(r'<scene>(.*?)</scene>')
        p4 = re.compile(r'<company>(.*?)</company>')
        p5 = re.compile(r'<movie>(.*?)</movie>')
        p6 = re.compile(r'<book>(.*?)</book>')
        p7 = re.compile(r'<government>(.*?)</government>')
        p8 = re.compile(r'<position>(.*?)</position>')
        p9 = re.compile(r'<address>(.*?)</address>')
        p10 = re.compile(r'<game>(.*?)</game>')

        if p1.findall(text):
            for ent in p1.findall(text):
                res[ent] = 'name'
        if p2.findall(text):
            for ent in p2.findall(text):
                res[ent] = 'org'
        if p3.findall(text):
            for ent in p3.findall(text):
                res[ent] = 'scene'
        if p4.findall(text):
            for ent in p4.findall(text):
                res[ent] = 'com'
        if p5.findall(text):
            for ent in p5.findall(text):
                res[ent] = 'movie'
        if p6.findall(text):
            for ent in p6.findall(text):
                res[ent] = 'book'
        if p7.findall(text):
            for ent in p7.findall(text):
                res[ent] = 'gov'
        if p8.findall(text):
            for ent in p8.findall(text):
                res[ent] = 'pos'
        if p9.findall(text):
            for ent in p9.findall(text):
                res[ent] = 'loc'
        if p10.findall(text):
            for ent in p10.findall(text):
                res[ent] = 'game'
        return res

    with open(inputfile,'r',encoding='utf-8') as f:
        for line in f:
            line = json.loads(line)
            label = line['labels']
            predict = line['output']
            gold_list = extract_ner(label)
            pred_list = extract_ner(predict)
            # print(gold_list)
            # print(pred_list)
            Z += len(gold_list)
            Y += len(pred_list)
            for k1 in gold_list:
                for k2 in pred_list:
                    if k1 == k2 and gold_list[k1] == pred_list[k2]:
                        X += 1
    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
    print("ner metrics:  f1: %.4f, precision: %.4f, recall: %.4f" % (f1, precision, recall))



if __name__ =="__main__":

    #evaluate ner
    inputfile = r'E:\openlab\ChatGLM2-6B\ptuning\output\ner\model1\checkpoint-3000\generated_predictions.txt'
    eval_ner_cluner(inputfile)
