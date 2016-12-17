import json

def get_squad_data():
    correct_file_name = "Correct_Sentences.json"

    with open(correct_file_name) as data_file:
        correct_sentences = json.load(data_file)
    vocab = set()
    X = []
    Y = []

    for key in correct_sentences.keys():
        question = correct_sentences.get(key)
        for sentence in question.get("sentences"):
            sentence_question_tokens = sentence  + question.get("question")
            ground_truths = question.get("ground_truths")
            vocab.update(set(sentence_question_tokens))
            for ground_truth in ground_truths:
                X.append(sentence_question_tokens)
                Y.append(ground_truth)
                vocab.update(set(ground_truth))

    print(len(X), len(Y), len(vocab))
    return X,Y,vocab

if __name__ == '__main__':
    X,Y,vocab = get_squad_data()