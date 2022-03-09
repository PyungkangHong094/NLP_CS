"""
author-gh: @adithya8
editor-gh: ykl7
"""

import math 

import numpy as np
import torch
import torch.nn as nn

sigmoid = lambda x: 1/(1 + torch.exp(-x))

class WordVec(nn.Module):
    def __init__(self, V, embedding_dim, loss_func, counts):
        super(WordVec, self).__init__()
        self.center_embeddings = nn.Embedding(num_embeddings=V, embedding_dim=embedding_dim)
        self.center_embeddings.weight.data.normal_(mean=0, std=1/math.sqrt(embedding_dim))
        self.center_embeddings.weight.data[self.center_embeddings.weight.data<-1] = -1
        self.center_embeddings.weight.data[self.center_embeddings.weight.data>1] = 1

        self.context_embeddings = nn.Embedding(num_embeddings=V, embedding_dim=embedding_dim)
        self.context_embeddings.weight.data.normal_(mean=0, std=1/math.sqrt(embedding_dim))
        self.context_embeddings.weight.data[self.context_embeddings.weight.data<-1] = -1 + 1e-10
        self.context_embeddings.weight.data[self.context_embeddings.weight.data>1] = 1 - 1e-10
        
        self.loss_func = loss_func
        self.counts = counts

    def forward(self, center_word, context_word):

        if self.loss_func == "nll":
            return self.negative_log_likelihood_loss(center_word, context_word)
        elif self.loss_func == "neg":
            return self.negative_sampling(center_word, context_word)
        else:
            raise Exception("No implementation found for %s"%(self.loss_func))

    def negative_log_likelihood_loss(self, center_word, context_word):
        ### TODO(students): start

        Context = self.context_embeddings(context_word)
        Center = self.center_embeddings(center_word)

        trans_p = torch.matmul(torch.transpose(Context, 0, 1), torch.transpose(Center, 0, 1))
        B = torch.logsumexp(trans_p, 1)
        A = torch.matmul(torch.transpose(Context, 0, 1), torch.transpose(Center, 0, 1))
        loss = torch.sub(B, A)
        loss = loss.mean()
        return loss
    
    def negative_sampling(self, center_word, context_word):
        k = 1
        data_tensor = self.center_embeddings(center_word)
        conv_tensor = self.context_embeddings(context_word)

        first_part = -torch.log(torch.sigmoid(torch.matmul(torch.transpose(conv_tensor, 0, 1),data_tensor)))
        second_part = torch.sum(torch.sigmoid((torch.matmul(torch.multinomial(self.counts, k), 1), data_tensor)))
        # torch.log(second_part)
        loss = torch.add(first_part,second_part )
        loss = loss.mean()



        # # second_part = torch.sum(torch.log(torch.matmul(torch.neg(k_samples),vector_tensor.T)),dim=1)
        # loss = torch.sub(first_part, second_part)
        return loss


    def print_closest(self, validation_words, reverse_dictionary, top_k=8):
        print('Printing closest words')
        embeddings = torch.zeros(self.center_embeddings.weight.shape).copy_(self.center_embeddings.weight)
        embeddings = embeddings.data.cpu().numpy()

        validation_ids = validation_words
        norm = np.sqrt(np.sum(np.square(embeddings),axis=1,keepdims=True))
        normalized_embeddings = embeddings/norm
        validation_embeddings = normalized_embeddings[validation_ids]
        similarity = np.matmul(validation_embeddings, normalized_embeddings.T)
        for i in range(len(validation_ids)):
            word = reverse_dictionary[validation_words[i]]
            nearest = (-similarity[i, :]).argsort()[1:top_k+1]
            print(word, [reverse_dictionary[nearest[k]] for k in range(top_k)])