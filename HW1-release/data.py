"""
author-gh: @adithya8
editor-gh: ykl7
"""

import collections

import numpy as np
import torch
import random

np.random.seed(1234)
torch.manual_seed(1234)

# Read the data into a list of strings.
def read_data(filename):
    with open(filename) as file:
        text = file.read()
        data = [token.lower() for token in text.strip().split(" ")]
    return data

def build_dataset(words, vocab_size):
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocab_size - 1))
    # token_to_id dictionary, id_to_taken reverse_dictionary
    vocab_token_to_id = dict()
    for word, _ in count:
        vocab_token_to_id[word] = len(vocab_token_to_id)
    data = list()
    unk_count = 0
    for word in words:
        if word in vocab_token_to_id:
            index = vocab_token_to_id[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    vocab_id_to_token = dict(zip(vocab_token_to_id.values(), vocab_token_to_id.keys()))
    return data, count, vocab_token_to_id, vocab_id_to_token

class Dataset:
    def __init__(self, data, batch_size=128, num_skips=8, skip_window=4):
        """
        @data_index: the index of a word. You can access a word using data[data_index]
        @batch_size: the number of instances in one batch
        @num_skips: the number of samples you want to draw in a window 
                (In the below example, it was 2)
        @skip_windows: decides how many words to consider left and right from a context word. 
                    (So, skip_windows*2+1 = window_size)
        """

        self.data_index = 0
        self.data = data
        assert batch_size % num_skips == 0
        assert num_skips <= 2 * skip_window

        self.batch_size = batch_size
        self.num_skips = num_skips
        self.skip_window = skip_window
    
    def reset_index(self, idx=0):
        self.data_index=idx

    def generate_batch(self):
        """
        Write the code generate a training batch

        batch will contain word ids for context words. Dimension is [batch_size].
        labels will contain word ids for predicting(target) words. Dimension is [batch_size, 1].
        """

        #batch 증명한 단어
        center_word = np.ndarray(shape=(self.batch_size), dtype=np.int32)
        #lables
        context_word = np.ndarray(shape=(self.batch_size, 1), dtype=np.int32)
        
        # stride: for the rolling window
        stride = 1

        #스킵 윈도우 타겟 스킵윈도우 skip
        span = 2 * self.skip_window + 1
        buffer = collections.deque(maxlen=span)

        # batch 파일 만들고, 2번째는 로스 포뮬라 코드 짜기, 보고서 작성
        ### TODO(students): start

        # for _ in range(span):
        #     buffer.append(self.data[self.data_index])
        #     self.data_index = (self.data_index + stride) % len(self.data)
        #
        # for i in range(self.batch_size):
        #     # target label at the center of the buffer
        #     target = self.skip_window
        #     #피해진 타겟을 말한는거
        #     no_target = [self.skip_window]
        #     buffer.append(self.data[self.data_index])
        #     self.data_index = (self.data_index + 1) % len(self.data)
        # return center_word, context_word

        self.data_index = self.data_index + self.skip_window
        if self.data_index == 0:
            print("-")
        else:
            self.data_index
        self.data_index = self.data_index % len(self.data)
        # Used to keep track of the number of words in the batch so far
        batch_size = 0
        index = 1
        while batch_size < self.batch_size:
            context_word[batch_size:batch_size + self.num_skips] = self.data[self.data_index]
            #index를 더해서 기존에 데이터를 더한다
            Add_data_index = self.data_index + index

            arry_window = self.data[self.data_index - self.skip_window:self.data_index] + self.data[Add_data_index: Add_data_index + self.skip_window]
            # 문맥 단어의 무작위 샘플링입니다. num_skips는 창 크기보다 작을 수 있습니다.
            sample_window = np.random.choice(arry_window, size=self.num_skips, replace=False)
            center_word[batch_size:batch_size + self.num_skips] = sample_window

            batch_size = batch_size + self.num_skips
            self.data_index = self.data_index + stride

        return center_word, context_word

        ### TODO(students): end
        return torch.LongTensor(center_word), torch.LongTensor(context_word)