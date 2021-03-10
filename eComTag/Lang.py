class Lang:
    def __init__(self):
        self.word2count = {}

    def add_funs(self, init_index2word):
        self.init_index2word = init_index2word
        self.word2count = {str(v): 1 for k, v in init_index2word.items()}
        self.word2index = {str(v): int(k) for k, v in init_index2word.items()}
        self.index2word = init_index2word
        self.n_words = len(init_index2word)  # Count default tokens

    def index_words(self, sentence):
        for word in sentence:
            self.index_word(word.strip())

    def index_word(self, word):
        if word not in self.word2count:
            # self.word2index[word] = self.n_words
            self.word2count[word] = 1
            # self.index2word[self.n_words] = word
            # self.n_words += 1
        else:
            self.word2count[word] += 1