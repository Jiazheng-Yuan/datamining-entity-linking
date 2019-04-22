import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn import svm
import math
from scipy import spatial
import pickle
from sklearn.model_selection import GridSearchCV


import collections
class SupModel:
    def __init__(self):
        self.dict = {}
        self.w = None
        self.clf = None
        self.b = None

    def cosine_similarity(self,entity1,entity2):
        vec1 = entity1.split(" ")
        vec2 = entity2.split(" ")
        def gen_map(vec):
            mapp = collections.defaultdict(int)
            for term in vec:
                mapp[term] += 1
            return mapp
        vec1 = gen_map(vec1)
        vec2 = gen_map(vec2)

        intersection = set(vec1.keys()) & set(vec2.keys())
        sum = 0

        for term in intersection:
            sum1 = 0
            sum2 = 0
            if term in vec1:
                sum1 += vec1[term]
            if term in vec2:
                sum2 += vec2[term]
            sum += sum1 * sum2
        len1 = 0
        len2 = 0
        for key,val in vec1.items():
            len1 += val ** 2
        for key,val in vec2.items():
            len2 += val ** 2
        return sum / (math.sqrt(len1)*math.sqrt(len2))



        pass

    def fit(self, dataset):
        X = []
        Y = []
        for mention in dataset.mentions:
            entry = []
            flag = False
            if len(mention.candidates) == 0:
                continue
            for i in range(len(mention.candidates)):
                if mention.gt.name == mention.candidates[i].name:
                    Y.append(i)
                    flag = True
                    break

            if not flag:
                continue
            for i in range(8):
                if i >= len(mention.candidates):
                    entry.append(0)
                    entry.append(0)
                    continue
                word = mention.candidates[i].name
                entry.append(mention.candidates[i].prob)
                entry.append(self.cosine_similarity(mention.surface,word))
            X.append(entry)
        clf = svm.SVC(gamma="scale",C=10)
        clf.fit(X,Y)
        self.clf = clf




        # fill this function if your model requires training

    def predict(self, dataset):
        pred_cids = []
        for mention in dataset.mentions:
            entry = []
            for i in range(8):
                if i >= len(mention.candidates):
                    entry.append(0)
                    entry.append(0)
                    continue
                word = mention.candidates[i].name
                entry.append(mention.candidates[i].prob)
                entry.append(self.cosine_similarity(mention.surface,word))
            result = self.clf.predict([entry])
            pred_cids.append(mention.candidates[result[0]].id if mention.candidates else 'NIL')
        return pred_cids

class PriorModel:
    def fit(self,dataset):
        pass
    def predict(self,dataset):
        pred_cids = []
        for mention in dataset.mentions:
            pred_cids.append(mention.candidates[0].id if mention.candidates else 'NIL')
        return pred_cids

class RandomModel:
    def __init__(self):
        pass

    def fit(self, dataset):
        # fill this function if your model requires training
        pass

    def predict(self, dataset):
        pred_cids = []
        for mention in dataset.mentions:
            pred_cids.append(random.choice(mention.candidates).id if mention.candidates else 'NIL')
        return pred_cids


class Step3Model:
    def __init__(self):
        self.clf = None
        with open("../data/embeddings/ent2embed.pk", "rb") as rf:
            ent2embed = pickle.load(rf)
        self.ent2embed = ent2embed
        with open("../data/embeddings/word2embed.pk", "rb") as rf:
            word2embed = pickle.load(rf)
        self.word2embed = ent2embed

    def cosine_similarity(self,entity1,entity2):
        vec1 = entity1.split(" ")
        vec2 = entity2.split(" ")
        def gen_map(vec):
            mapp = collections.defaultdict(int)
            for term in vec:
                mapp[term] += 1
            return mapp
        vec1 = gen_map(vec1)
        vec2 = gen_map(vec2)

        intersection = set(vec1.keys()) & set(vec2.keys())
        sum = 0

        for term in intersection:
            sum1 = 0
            sum2 = 0
            if term in vec1:
                sum1 += vec1[term]
            if term in vec2:
                sum2 += vec2[term]
            sum += sum1 * sum2
        len1 = 0
        len2 = 0
        for key,val in vec1.items():
            len1 += val ** 2
        for key,val in vec2.items():
            len2 += val ** 2
        return sum / (math.sqrt(len1)*math.sqrt(len2))



        pass

    def fit(self, dataset):
        X = []
        Y = []
        icounter = 0
        for mention in dataset.mentions:

            #print("here")

            # if len(mention.candidates) == 0:
            #     continue
            # skip = True
            # for i in range(len(mention.candidates)):
            #     if mention.gt.name == mention.candidates[i].name:
            #
            #         skip = False
            #         break
            # if skip:
            #     continue

            #if counter == 17:
            #    print("here")
            context_embedding = np.zeros(300)
            for sentence in mention.contexts:
                for word in sentence:
                    if word in self.word2embed:
                        context_embedding += self.word2embed[word]
            context_embedding = context_embedding.tolist()
            counter = 0
            for candidate in mention.candidates:

                entry = []
                candidate_name = "_".join(candidate.name.split(" "))
                if candidate_name in self.ent2embed:
                    entry +=self.ent2embed[candidate_name].tolist()
                else:
                    entry += [0 for i in range(300)]
                entry += context_embedding
                Y.append(1 if candidate.name == mention.gt.name else 0)
                #entry.append(context_embedding)
                word = candidate_name
                entity = "_".join(mention.surface.split(" "))
                if entity not in self.ent2embed or \
                        word not in self.ent2embed:
                    entry.append(0)
                else:
                    if np.count_nonzero(self.ent2embed[entity]) == 0 or \
                            np.count_nonzero(self.ent2embed[word]) == 0:
                        entry.append(0)
                    else:
                        entry.append(spatial.distance.cosine(self.ent2embed[entity], self.ent2embed[word]))

                    #entry = np.array(entry)


                entry.append(self.cosine_similarity(mention.surface, word))
            #print(counter,len(entry))
                X.append(entry)
            #break
        #X = np.array(X)
        clf = svm.SVC(gamma='scale',C=10)


        clf.fit(X, Y)
        #clf = LogisticRegression(random_state=0, solver='liblinear',max_iter=4000).fit(X, Y)
        self.clf = clf

        # fill this function if your model requires training
        pass
    def predict(self, dataset):
        pred_cids = []

        for mention in dataset.mentions:

            if len(mention.candidates) == 0:
                pred_cids.append("NIL")
                continue

            #if counter == 17:
            #    print("here")
            context_embedding = np.zeros(300)
            for sentence in mention.contexts:
                for word in sentence:
                    if word in self.word2embed:
                        context_embedding += self.word2embed[word]
            context_embedding = context_embedding.tolist()
            Found = False
            for candidate in mention.candidates:
                entry = []
                candidate_name = "_".join(candidate.name.split(" "))
                if candidate_name in self.ent2embed:
                    entry +=self.ent2embed[candidate_name].tolist()
                else:
                    entry += [0 for i in range(300)]
                #Y.append(1 if candidate.name == mention.gt.name else 0)
                #entry.append(context_embedding)
                word = candidate_name
                entity = "_".join(mention.surface.split(" "))
                entry += context_embedding
                if entity not in self.ent2embed or \
                        word not in self.ent2embed:
                    entry.append(0)
                else:
                    if np.count_nonzero(self.ent2embed[entity]) == 0 or \
                            np.count_nonzero(self.ent2embed[word]) == 0:
                        entry.append(0)
                    else:
                        entry.append(spatial.distance.cosine(self.ent2embed[entity], self.ent2embed[word]))

                    #entry = np.array(entry)


                entry.append(self.cosine_similarity(mention.surface, word))
                result = self.clf.predict([entry])
                #found = False
                if result[0] == 1:
                    pred_cids.append(candidate.id)
                    Found = True
                    #print("here")
                #    found = True
                    break
                #if not found:
            if not Found:
                pred_cids.append(mention.candidates[0].id)

        return pred_cids


class Step3Model_original:
    def __init__(self):
        self.clf = None
        with open("../data/embeddings/ent2embed.pk", "rb") as rf:
            ent2embed = pickle.load(rf)
        self.ent2embed = ent2embed
        with open("../data/embeddings/word2embed.pk", "rb") as rf:
            word2embed = pickle.load(rf)
        self.word2embed = ent2embed

    def cosine_similarity(self,entity1,entity2):
        vec1 = entity1.split(" ")
        vec2 = entity2.split(" ")
        def gen_map(vec):
            mapp = collections.defaultdict(int)
            for term in vec:
                mapp[term] += 1
            return mapp
        vec1 = gen_map(vec1)
        vec2 = gen_map(vec2)

        intersection = set(vec1.keys()) & set(vec2.keys())
        sum = 0

        for term in intersection:
            sum1 = 0
            sum2 = 0
            if term in vec1:
                sum1 += vec1[term]
            if term in vec2:
                sum2 += vec2[term]
            sum += sum1 * sum2
        len1 = 0
        len2 = 0
        for key,val in vec1.items():
            len1 += val ** 2
        for key,val in vec2.items():
            len2 += val ** 2
        return sum / (math.sqrt(len1)*math.sqrt(len2))



        pass

    def fit(self, dataset,idset):
        X = []
        Y = []
        counter = 0
        for mention in dataset.mentions:
            counter += 1
            #if counter > 10000:
            #    break
            entry = []
            flag = False
            for i in range(len(mention.candidates)):
                if mention.gt.name == mention.candidates[i].name:
                    Y.append(i)
                    flag = True
                    break
            if not flag:
                continue
            #if counter == 17:
            #    print("here")
            for i in range(8):

                if i >= len(mention.candidates):
                    entry.append(0)
                    entry.append(0)
                    entry += [0 for i in range(301)]

                else:
                    word = mention.candidates[i].name

                    entry.append(mention.candidates[i].prob)
                    entry.append(self.cosine_similarity(mention.surface, word))
                    if "_".join(word.split(" ")) in self.ent2embed:
                        embedding_candidate = self.ent2embed["_".join(word.split(" "))]
                        entry += embedding_candidate.tolist()
                    else:
                        entry += [0 for i in range(300)]




                    word = "_".join(word.split(" "))
                    entity = "_".join(mention.surface.split(" "))
                    if i >= len(mention.candidates) or entity not in self.ent2embed or \
                             word not in self.ent2embed:
                        entry.append(0)
                    else:
                        if np.count_nonzero(self.ent2embed[entity]) == 0 or \
                                np.count_nonzero(self.ent2embed[word]) == 0:
                            entry.append(0)
                        else:
                            entry.append(spatial.distance.cosine(self.ent2embed[entity],self.ent2embed[word]))
                    #entry = np.array(entry)
            context_embedding = np.zeros(300)
            for sentence in mention.contexts:
                for word in sentence:
                    if word in self.word2embed:
                        context_embedding += self.word2embed[word]
            context_embedding = context_embedding.tolist()
            entry += context_embedding
            #print(counter,len(entry))
            X.append(entry)
            #break
        #X = np.array(X)
        clf = svm.SVC(gamma="scale",C=10)
        clf.fit(X, Y)
        self.clf = clf

        # fill this function if your model requires training
        pass
    def predict(self, dataset):
        pred_cids = []
        for mention in dataset.mentions:
            entry = []


            for i in range(8):

                if i >= len(mention.candidates):
                    entry.append(0)
                    entry.append(0)
                    entry += [0 for i in range(301)]

                else:
                    word = mention.candidates[i].name

                    entry.append(mention.candidates[i].prob)
                    entry.append(self.cosine_similarity(mention.surface, word))
                    if "_".join(word.split(" ")) in self.ent2embed:
                        embedding_candidate = self.ent2embed["_".join(word.split(" "))]
                        entry += embedding_candidate.tolist()
                    else:
                        entry += [0 for i in range(300)]




                    word = "_".join(word.split(" "))
                    entity = "_".join(mention.surface.split(" "))
                    if i >= len(mention.candidates) or entity not in self.ent2embed or \
                             word not in self.ent2embed:
                        entry.append(0)
                    else:
                        if np.count_nonzero(self.ent2embed[entity]) == 0 or \
                                np.count_nonzero(self.ent2embed[word]) == 0:
                            entry.append(0)
                        else:
                            entry.append(spatial.distance.cosine(self.ent2embed[entity],self.ent2embed[word]))
                    #entry = np.array(entry)
            context_embedding = np.zeros(300)
            for sentence in mention.contexts:
                for word in sentence:
                    if word in self.word2embed:
                        context_embedding += self.word2embed[word]
            context_embedding = context_embedding.tolist()
            entry += context_embedding

            result = self.clf.predict([entry])
            if len(mention.candidates) > 0 :
                #print("here")
                if len(result) < 1 or result[0] >= len(mention.candidates):
                    if result[0]!= 0:
                        print(result)
                    pred_cids.append(mention.candidates[0].id)
                else:
                    pred_cids.append(mention.candidates[result[0]].id)
            else:
                pred_cids.append('NIL')
        return pred_cids

