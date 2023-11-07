import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


class EmbeddingModels:
    def __init__(self, 
                 pretrained_tokenizer='sentence-transformers/all-mpnet-base-v2', 
                 pretrained_model='sentence-transformers/all-mpnet-base-v2', 
                 device='cpu'):
        # Load model from HuggingFace Hub
        self.pretrained_tokenizer = pretrained_tokenizer
        self.pretrained_model = pretrained_model
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_tokenizer)
        self.model = AutoModel.from_pretrained(self.pretrained_model)

        self.model.eval()
        # self.device = torch.device(device)
        # self.model.to(self.device)

    # Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def get_embedding(self, sentences: list):
        # Sentences we want sentence embeddings for
        # sentences = ['This is an example sentence', 'Each sentence is converted']

        # Tokenize sentences
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        # Perform pooling
        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])

        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

        return sentence_embeddings
    