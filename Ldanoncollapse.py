"(C) Copyright 2016, Hesam Amoualian inspired by Georgios Balikas code"
# References :
# D.M. Blei, A. Ng, M.I. Jordan. Latent Dirichlet Allocation. NIPS, 2002
# W.M. Darling. A Theoretical and Practical Implementation Tutorial on Topic Modeling and Gibbs Sampling. School of Computer Science University of Guelph. December 1, 2011
# H. Amoualian et al, Streaming-LDA: A Copula-based Approach to Modeling Topic Dependencies in Document Streams, KDD, 2016
# use python3 for running and write Python3 HDPcode.py
# needs to have toy_dataset.text and vocabulary.py in same path



from pylab import *
import numpy, codecs
from scipy.special import gamma, gammaln
from datetime import datetime
import vocabulary
import time
import math
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from itertools import groupby
          


class lda_gibbs_sampling:
    def __init__(self, K=20, alpha=0.5, beta=0.5, docs= None, V= None):
        self.K = K
        self.alpha = numpy.ones(K)*alpha # parameter of topics prior
        self.beta = numpy.ones(V)*beta   # parameter of words prior
        self.docs = docs #a list of documents which include the words
        self.V = V # number of different words in the vocabulary
        self.z_m_n = {} # topic assignements for documents
        self.n_m_z = numpy.zeros((len(self.docs), K))     # number of words assigned to topic z in document m
        self.n_z_t = numpy.zeros((K, V))+beta  # number of times a word v is assigned to a topic z
        self.theta = numpy.zeros((len(self.docs), K)) # topic distribution for each document
        self.phi = numpy.zeros((K, V))  # topic-words distribution for whole of corpus
        self.n_z = numpy.zeros(K) + V * beta  # total number of words assigned to a topic z
        self.pers=[] # Array for keeping perplexities over iterations
	

        for m, doc in enumerate(docs):         # Initialization
            for n,w in enumerate(doc):
                z = numpy.random.randint(0, K) # Randomly assign a topic to a word and increase the counting array
                self.n_m_z[m, z] += 1
                self.n_z_t[z,w] += 1
                self.z_m_n[(m,n)]=z
                self.n_z[z] += 1


    def inference(self,iteration):
        for m, doc in enumerate(self.docs):
            if m<901 :  # assign 900 documents through 1100 documents as train dataset
                         self.theta[m]=numpy.random.dirichlet(self.n_m_z[m]+self.alpha, 1)  #sample Theta for each document using uncollapsed gibbs

                         for n,w in enumerate(doc):  # update arrays for each word of a document
                           
                           z=self.z_m_n[(m,n)]
                           self.n_m_z[m,z] -=1
                           self.n_z_t[z,w] -=1
                           self.n_z[z] -= 1
                           self.phi[:,w]=self.n_z_t[:,w] / self.n_z
                          
                           p_z = self.theta[m]*self.phi[:,w]
                           new_z = numpy.random.multinomial(1, p_z/p_z.sum()).argmax()   #sample Z using multinomial distribution of equation 7 of reference 3
                           self.n_m_z[m,new_z] +=1
                           self.n_z_t[new_z,w] +=1
                           self.n_z[new_z] += 1
                           self.z_m_n[(m,n)]=new_z
            


            else:  # assign 200 documents through 1100 documents as test dataset
                     self.theta[m]=numpy.random.dirichlet(self.n_m_z[m]+self.alpha, 1)
                     for n,w in enumerate(doc):
                        
                        z=self.z_m_n[(m,n)]
                        self.n_m_z[m,z] -=1
                        
                        O=self.n_z_t[:,w] / self.n_z  # use the previous phi for sampling z
                        self.n_z_t[z,w] -=1
                        self.n_z[z] -= 1

                        p_z = self.theta[m] * O
                        new_z = numpy.random.multinomial(1, p_z/p_z.sum()).argmax()
                                            
                        self.n_m_z[m,new_z] +=1
                        self.z_m_n[(m,n)]=new_z
                        self.n_z_t[new_z,w] +=1
                        self.n_z[new_z] += 1
                        O=[]


        per=0
        b=0
        c=0
        self.phi=self.n_z_t/ self.n_z[:, numpy.newaxis]

        for m, doc in enumerate(self.docs):  # find perplexity over whole of the words of test set
            if m>901:
                b+=len(doc)
                
                for n, w in enumerate(doc):
                    l=0
                    for i in range(self.K):
                        l+=(self.theta[m,i])*self.phi[i,w]
                    c+=numpy.log(l)
 
        per=numpy.exp(-c/b)
        print ('perpelixity:', per)

    def worddist(self):
       
        """topic-word distribution"""
        return self.phi


if __name__ == "__main__":
    corpus = codecs.open("toy_dataset.txt", 'r', encoding='utf8').read().splitlines()
    iterations = 50
    voca = vocabulary.Vocabulary(excluds_stopwords=False)
    docs = [voca.doc_to_ids(doc) for doc in corpus]

    lda = lda_gibbs_sampling(K=20, alpha=0.5, beta=0.5, docs=docs, V=voca.size())
    for i in range(iterations):
        print ('iteration:', i)
        lda.inference(i)
	
    d = lda.worddist()
    for i in range(20):
        ind = numpy.argpartition(d[i], -10)[-10:]
        for j in ind:
            print (voca[j],' ',end=""),
        print ()
        
        
        
