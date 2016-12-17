import json

def get_squad_data(size=None):
    correct_file_name = "Correct_Sentences.json"

    with open(correct_file_name) as data_file:
        correct_sentences = json.load(data_file)
    vocab = set()
    X = []
    Xq = []
    Y = []

    if size != None:
        count = 0
    for key in correct_sentences.keys():
        question = correct_sentences.get(key)
        question_tokens = question.get("question")
        ground_truths = question.get("ground_truths")       
        vocab.update(set(question_tokens))   
        for sentence in question.get("sentences"):
            vocab.update(set(sentence))
            for ground_truth in ground_truths:
                X.append(sentence)
                Xq.append(question_tokens)
                Y.append(ground_truth)
                vocab.update(set(ground_truth))
                count+=1
                if count == size:
                    break
            if count == size:
                break
        if count == size:
                break
    print(len(X), len(Xq), len(Y), len(vocab))
    return X,Xq,Y,vocab

if __name__ == '__main__':
    X,Xq,Y,vocab = get_squad_data(10)