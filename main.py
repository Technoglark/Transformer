import torch
import torch.nn as nn
from model import Transformer
from utils import tokenize_text, create_corpus_indexed
import gensim.models.word2vec as w2v
from generator import BeamGenerator

def generate(model: Transformer, sentence: str, w2v_model, device):
    corpus = [tokenize_text(sentence, type='context')]
    print(corpus)
    answers = [['<SOS>']]
    corpus_indexed = torch.tensor(create_corpus_indexed(corpus, w2v_model)).to(device)
    answers_indexed = torch.tensor(create_corpus_indexed(answers, w2v_model)).to(device)
    last_word = '<SOS>'
    counter = 0

    print(answers_indexed)

    with torch.no_grad():
        print("Start of generating...")
        while last_word != "<EOS>" and counter < 20:
            predictions = model.forward(corpus_indexed, answers_indexed)
            best_idx = torch.argmax(torch.softmax(predictions[ :, -1, : ], dim=-1), dim=-1).item()
            best_word = w2v_model.wv.index_to_key[best_idx]
            values, idxs = torch.topk(predictions[ :, -1, : ], k=5, dim=-1)
            best_words = [w2v_model.wv.index_to_key[i.item()] for i in idxs[0]]
            answers[0].append(best_word)
            answers_indexed = torch.tensor(create_corpus_indexed(answers, w2v_model)).to(device)
            last_word = best_word
            counter += 1
    return " ".join(answers[0])



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
w2v_model = w2v.Word2Vec.load('russian_embedding.model')
model = Transformer(300, len(w2v_model.wv.key_to_index), device).to(device)
model.load_state_dict(torch.load('LModel.pt', weights_only=True))


text = "Расскажи основной смысл диалога <QCT> Райан: Какая машина, по вашему мнению, лучшая для гонок Формулы-1? Джек: Я не большой поклонник этого вида спорта, но я думаю, что Феррари — это то, что нужно. Райан: Это было, но с появлением Lamborghini. Я больше никогда не буду прежней.Джек: Хорошо"
generator = BeamGenerator(model, device, k=10, max_len=50)
print(generator.generate(text, w2v_model))

'''
Нарушение авторских прав впервые в российской юридической практике стало поводом для принудительного прекращения
 предпринимательской деятельности. Адыгейский суд лишил продавца контрафакта статуса индивидуального предпринимателя. 
 Решение станет ориентиром в спорах правообладателей с пиратами, надеются участники рынка программного обеспечения. 
 Но едва ли число исков будет значительным, скептичны юристы.
'''