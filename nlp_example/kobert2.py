import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np
from tqdm import tqdm
import pandas as pd

#kobert
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model

#kobert
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model

#GPU 사용
device = torch.device("cuda:0")

#BERT 모델, Vocabulary 불러오기
bertmodel, vocab = get_pytorch_kobert_model()

#사이킷런에서 제공하는 train_test_split
from sklearn.model_selection import train_test_split


class Chatbot(Dataset):
    def __init__(self):
        self.data_list = []
        self.dataset = None
        self.sent_idx = None
        self.label_idx = None
        self.bert_tokenizer = None
        self.max_len = None
        self.pad = None
        self.pair = None
        self.transform = nlp.data.BERTSentenceTransform(self.bert_tokenizer, max_seq_length=self.max_len, pad=self.pad, pair=self.pair)
        self.sentences = []
        self.labels = []
        self.max_len = 64
        self.batch_size = 64
        self.warmup_ratio = 0.1
        self.num_epochs = 5
        self.max_grad_norm = 1
        self.log_interval = 200
        self.learning_rate = 5e-5

    def hook(self):
        self.preprocessing()
        self.train()
        self.test()

    def preprocessing(self):
        chatbot_data = pd.read_excel('./data/한국어_단발성_대화_데이터셋.xlsx')
        chatbot_data.loc[(chatbot_data['Emotion'] == "공포"), 'Emotion'] = 0  # 공포 => 0
        chatbot_data.loc[(chatbot_data['Emotion'] == "놀람"), 'Emotion'] = 1  # 놀람 => 1
        chatbot_data.loc[(chatbot_data['Emotion'] == "분노"), 'Emotion'] = 2  # 분노 => 2
        chatbot_data.loc[(chatbot_data['Emotion'] == "슬픔"), 'Emotion'] = 3  # 슬픔 => 3
        chatbot_data.loc[(chatbot_data['Emotion'] == "중립"), 'Emotion'] = 4  # 중립 => 4
        chatbot_data.loc[(chatbot_data['Emotion'] == "행복"), 'Emotion'] = 5  # 행복 => 5
        chatbot_data.loc[(chatbot_data['Emotion'] == "혐오"), 'Emotion'] = 6  # 혐오 => 6

        for q, label in zip(chatbot_data['Sentence'], chatbot_data['Emotion']):
            data = []
            data.append(q)
            data.append(str(label))
            self.data_list.append(data)

    def train(self):
        dataset_train = train_test_split(self.data_list, test_size=0.25, random_state=0)
        return dataset_train

    def test(self):
        dataset_test = train_test_split(self.data_list, test_size=0.25, random_state=0)
        return dataset_test

    def bert_dataset(self):
        transform = nlp.data.BERTSentenceTransform(self.bert_tokenizer, max_seq_length=self.max_len, pad=self.pad, pair=self.pair)
        sentences = [self.transform([i[self.sent_idx]]) for i in self.dataset]
        labels = [np.int32(i[self.label_idx]) for i in self.dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i]))

    def __len__(self):
        return (len(self.labels))

    def tokenizer(self):
        # 토큰화
        tokenizer = get_tokenizer()
        tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

        data_train = self.bert_dataset(self.train(), 0, 1, tok, self.max_len, True, False)
        data_test = self.bert_dataset(self.test(), 0, 1, tok, self.max_len, True, False)

        print(data_train[0])
        print(data_test[0])

if __name__ == '__main__':
    Chatbot().hook()
