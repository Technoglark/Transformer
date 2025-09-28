import torch
from model import Transformer
from utils import tokenize_text, create_corpus_indexed
import math
import copy



class BeamGenerator():

    def __init__(self, model: Transformer, device, k=3, max_len=20):
        self.candidates: list[list] = [] # type : [[candidate, likelihood]]
        self.device = device
        self.max_len = max_len
        self.context = None
        self.k = k
        self.model = model

    def init_context(self, context) -> None:
        self.context = context

    def init_candidates(self, w2v_model) -> None:
        self.candidates = [[create_corpus_indexed([['<SOS>']], w2v_model), 0]]

    def update_candidates(self) -> None:
        with torch.no_grad():
            curr_candidates = []
            for candidate, ll in self.candidates:
                print(candidate)
                context_tensor = torch.tensor(self.context).to(self.device)
                candidate_tensor = torch.tensor(candidate).to(self.device)
                predictions = self.model.forward(context_tensor, candidate_tensor)
                values, idxs = torch.topk(torch.softmax(predictions[ :, -1, : ], dim=-1), k=self.k, dim=-1)
                for i, v in zip(idxs[0], values[0]):
                    new_candidate = copy.deepcopy(candidate)
                    new_candidate[0].append(i.item())
                    curr_candidates.append([new_candidate, (ll * (len(new_candidate[0]) - 1) + math.log(v.item())) / len(new_candidate[0])])

        self.candidates = sorted(curr_candidates, key=lambda x: x[1], reverse=True)[: self.k].copy()

    
    def continue_generating(self) -> bool:
        for candidate, ll in self.candidates:
            if candidate[0][-1] != '<EOS>':
                return True
        return False

    def to_string(self, candidate, w2v_model) -> str:
        return " ".join([w2v_model.wv.index_to_key[x] for x in candidate[0]])


    def generate(self, sentence: str, w2v_model) -> str:
        context = [tokenize_text(sentence, type='context')]
        context_indexed = torch.tensor(create_corpus_indexed(context, w2v_model)).to(self.device)
        self.init_context(context_indexed)
        self.init_candidates(w2v_model)
        self.model.eval()
        counter = 0

        while self.continue_generating() and counter < self.max_len:
            self.update_candidates(w2v_model)
            counter += 1

        result = None
        max_ll = -1e9
        for candidate, ll in self.candidates:
            if ll > max_ll:
                result = candidate
                max_ll = ll
        
        return self.to_string(result, w2v_model)
