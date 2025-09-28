import torch
import torch.nn as nn
from model import Transformer
from utils import tokenize_text, create_corpus_indexed
import gensim.models.word2vec as w2v
from generator import BeamGenerator


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
w2v_model = w2v.Word2Vec.load('russian_embedding.model')
model = Transformer(300, len(w2v_model.wv.key_to_index), device).to(device)
model.load_state_dict(torch.load('LModel.pt', weights_only=True))


text = "Расскажи основной смысл диалога <QCT> Райан: Какая машина, по вашему мнению, лучшая для гонок Формулы-1? Джек: Я не большой поклонник этого вида спорта, но я думаю, что Феррари — это то, что нужно. Райан: Это было, но с появлением Lamborghini. Я больше никогда не буду прежней.Джек: Хорошо"
generator = BeamGenerator(model, device, k=10, max_len=50)
print(generator.generate(text, w2v_model))
