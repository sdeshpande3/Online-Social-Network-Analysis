#!/usr/bin/env python
# coding: utf-8

# In[14]:


from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen
import pickle
import re
import numpy as np


# In[15]:


def get_tweets(name):
    """
    Load stored tweets.
    List of strings, one per tweet.
    """
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


# In[16]:


def download_afin():

    url = urlopen('http://www2.compute.dtu.dk/~faan/data/AFINN.zip')
    file = ZipFile(BytesIO(url.read()))
    afin_f = file.open('AFINN/AFINN-111.txt')
    return afin_f


# In[17]:


def read_data(file):

    afinn = {}
    for l in file:
        line = l.strip().split()
        if len(line) == 2:
            afinn[line[0].decode("utf-8")] = int(line[1])
    return afinn


# In[18]:


def tweet_sentiment(terms, afinn, verbose=False):

    pos = 0
    neg = 0
    for t in terms:
        if t in afinn:
            if verbose:
                print('\t%s=%d' % (t, afinn[t]))
            if afinn[t] > 0:
                pos += afinn[t]
            else:
                neg += -1 * afinn[t]
    return pos, neg


# In[19]:


def tokenize(doc, keep_internal_punct = False):
    convert = doc.lower();
    if keep_internal_punct == False:
        return np.array(re.findall("[\w_]+", convert))
    else:
        return np.array(re.findall('[\w_][^\s]*[\w_]|[\w_]', convert))


# In[20]:


def token_features(tokens, feats):
    cnt = Counter(tokens)
    for i in cnt:
        feats["token="+i] = cnt[i]


# In[21]:


def token_pair_features(tokens, feats, k=3):
    windows = []
    def window_creator(list,degree):
        for ws in range(len(tokens) - degree + 1):
            yield [list[ws+l] for l in range(degree)]

    window_generator = window_creator(tokens,k)

    for window in window_generator:
        subseq = [c[0]+"__"+c[1] for c in combinations(window,2)]
        for sub in subseq:
            if "token_pair="+sub not in feats:
                feats["token_pair="+sub] = 1
            elif "token_pair="+sub in feats:
                feats["token_pair="+sub] = feats["token_pair="+sub] + 1


# In[22]:


def lexicon_features(tokens, feats):
    feats['pos_words'] = 0
    feats['neg_words'] = 0

    for token in tokens :
        if(token.lower() in pos_words):
            feats['pos_words'] += 1
        elif(token.lower() in neg_words):
            feats['neg_words'] += 1


# In[23]:


def featurize(tokens, feature_fns):
    feats = {}
    for features in feature_fns:
        features(tokens,feats)

    result = [(x, y) for x, y in feats.items()]

    return sorted(result,key=lambda x:x[0])


# In[24]:


def sentiment(tweets,tokens,afin):

    posv = []
    negv = []
    combo = []
    for tk, tw in zip(tokens, tweets):
        pos, neg = tweet_sentiment(tk, afin)
        if neg == pos:
            combo.append((tw['text'], pos, neg))
        elif neg > pos:
            negv.append((tw['text'], pos, neg))
        elif pos > neg:
            posv.append((tw['text'], pos, neg))
    return posv, negv, combo


# In[25]:


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f)


# In[26]:


def main():

    tweets = get_tweets('tweets')
    print("got tweets!")
    afin = download_afin()
    read = read_data(afin)
    print("afin data downloaded and read!")
    tokens = [tokenize(t['text']) for t in tweets]
    positives, negatives, combined =sentiment(tweets,tokens,read)
    print("There are %d positive tweets, %d negative tweets and %d neutral tweets." 
          % (len(positives), len(negatives),len(combined)))
    classify = (positives,negatives,combined)
    save_obj(classify, 'classify')
    print("classify saved!")

if __name__ == '__main__':
    main()

