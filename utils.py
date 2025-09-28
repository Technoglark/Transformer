import pandas as pd
import gensim.models.word2vec as w2v
import re


def tokenize_text(text, type: str):
    """
    Компактная версия токенизатора
    """
    text = text.replace('<QCT>', ' ___QCT___ ')
    
    # Находим все токены: слова, числа, знаки препинания
    tokens = re.findall(r'\w+(?:\'\w+)?|\d+\.?\d*|\.\.\.|[^\w\s]', text)
    
    # Обрабатываем токены
    processed = []
    if type == 'answer':
        processed.append('<SOS>')
    for token in tokens:
        if re.match(r'^[a-zA-Z]', token):  # если начинается с буквы
            processed.append(token.lower())
        elif token == '___QCT___':
            processed.append('<QCT>')
        else:
            processed.append(token)
    if type == 'answer':
        processed.append('<EOS>')
    
    return processed


def pandas_to_list(filepath: str, column_name, type:str, subsample=None) -> list[list[str]]:
    result: list[list[str]] = []
    if subsample is not None:
        df = pd.read_csv(filepath, nrows=subsample)
    else:
        df = pd.read_csv(filepath)
    for sentence in df[column_name]:
        result.append(tokenize_text(sentence, type))
    
    return result

def make_embedding(corpus: list[list[str]], filename: str,  size):
    model = w2v.Word2Vec(sentences=corpus, vector_size=size, window=4, min_count=1, workers=4, sg=1)
    model.save(filename)


def create_corpus_indexed(corpus: list[list[str]], w2v_model):
    vocab = w2v_model.wv.key_to_index
    unk_idx = vocab.get('<unk>', 0)  

    numericalized_corpus = []
    for sentence in corpus:
        indexed_sentence = [vocab.get(word, unk_idx) for word in sentence]
        numericalized_corpus.append(indexed_sentence)

    return numericalized_corpus


def unite(corpus1: list[list[int]], corpus2: list[list[int]]) -> list[list[list[int]]]:
    return [[x, y] for x, y in zip(corpus1, corpus2) if len(x) > 0 and len(y) > 0]


import re

def to_dict(text_set: str) -> dict:
    result = {}
    
    # Находим все пары вопрос-ответ
    pattern = r'"Question" -> "([^"]+)".*?"Answers" -> {"([^"]+)"}'
    matches = re.findall(pattern, text_set, re.DOTALL)
    
    for question, answer in matches:
        result[question] = answer
    
    return result