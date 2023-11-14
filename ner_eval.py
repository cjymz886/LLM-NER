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





if __name__ =="__main__":

    #evaluate ner
    inputfile = r'E:\openlab\ChatGLM2-6B\ptuning\output\ner\model1\checkpoint-3000\generated_predictions.txt'
    eval_ner_cluner(inputfile)