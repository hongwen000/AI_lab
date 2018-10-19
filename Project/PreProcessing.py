#!/usr/bin/env python
# coding: utf-8

# In[31]:


import csv
import re as regex, string
from typing import List
from enchant.checker import SpellChecker
import numba
from tqdm import tnrange, tqdm_notebook
from time import sleep
import nltk


# In[32]:


# import logging
# import sys
# logging.basicConfig(format='[%(funcName)s]: %(message)s',
#                      level=logging.INFO, stream=sys.stdout)
# logger = logging.getLogger("logger")
# logger.setLevel(logging.DEBUG)
# logger.info('Hello world!')
# logger.debug("Hello deubg")





# ## 分词

# In[34]:


from nltk.tokenize import word_tokenize
def split_words(fdata: List[List])->List[List]:
    ret = []
    for row in tqdm_notebook(fdata):
        words = word_tokenize(row)
        ret.append(words)
    return ret


# In[35]:




# In[36]:





# ## 去除标点、特殊符号、HTML标签等非英文内容

# In[37]:


def remove_punc(row_of_words: List[List])->List[List]:
    ret = []
    for row in tqdm_notebook(row_of_words):
        words = [word for word in row if word.isalpha()]
        ret.append(words)
    return ret


# In[38]:





# In[40]:


print(sum(len(row) for row in passage))


# ## 去除停用词

# In[ ]:


from nltk.corpus import stopwords
def remove_stop_words(passage):
    stop_words = set(stopwords.words('english'))
    stop_words.add('us')
    passage = [list(filter(lambda w: w.lower() not in stop_words, row)) for row in passage]
    return passage




# In[44]:


import autocorrect

def words_spell_check(fdata)->List[List]:
    ret = []
    err = 0
    cnt = 0
    for row in tqdm_notebook(fdata):
        corrected_row = []
        for word in row:
            suggest = autocorrect.spell(word)
            if word != suggest:
                err += 1
            cnt += 1
            corrected_row.append(word)
        ret.append(corrected_row)
    print("There are {} errors in {} words, error rate : {}".format(err, cnt, err/cnt))
    return ret

import os, time, random
def correct_words(passage):
    print('Run task (%s)...' % (os.getpid()))
    ret = [[autocorrect.spell(word) for word in row] for row in passage]
    return ret


# In[45]:


from multiprocessing import Pool
from typing import List, NoReturn, Callable
def list_multiprocess(lst: List, func: Callable[[List],List], n: int)-> List:
    if len(lst) < n:
        return func(lst)
    p = Pool(n)
    lists = []
    psize = int(len(lst) / n)
    for i in range(n - 1):
        lists.append(lst[i * psize: (i+1) * psize])
    lists.append(lst[(n-1) * psize:])
    ret = []
    for i in range(n):
        ret.append(p.apply_async(func, args=(lists[i],)))
    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    last = []
    for i in ret:
        last += i.get()
    print('All subprocesses done.')
    return last


# In[46]:




# ## 词性标注

# In[48]:


from nltk.corpus import wordnet
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None




# In[49]:


from pattern.en import lemma
lemma('countries')


# In[50]:


from nltk.stem import WordNetLemmatizer
def lemma_passage(passage):
    ret = list(range(len(passage)))
    lemmatizer = WordNetLemmatizer()
    for i, row in tqdm_notebook(enumerate(passage)):
        nrow = []
        for w, pos in nltk.pos_tag(row):
            wordnet_pos = get_wordnet_pos(pos) or wordnet.NOUN
            nrow.append(lemmatizer.lemmatize(w, pos=wordnet_pos))
        ret[i] = nrow
    return ret


# In[51]:





# In[73]:


# In[ ]:


# In[33]:

def PreProcess(filen:str, slice:int) -> List[List]:
    # ## 读取文件
    filen = 'data/5/testData.txt'
    fdata = list(open(filen))
    fdata = fdata[0:1000]
    # ## 分词
    passage = split_words(fdata)
    print("#word After spliting: ", sum(len(row) for row in passage))
    # ## 去除非单词
    passage = remove_punc(passage)
    print("#word After removing punc: ", sum(len(row) for row in passage))
    # ## 拼写检查
    passage = list_multiprocess(passage, words_spell_check, 6)
    # ## 词形还原
    passage = lemma_passage(passage)
    print("#word After lemma: ", sum(len(row) for row in passage))
    # ## 去除停用词
    passage_compress = remove_stop_words(passage)
    print("#word After removing stopwords: ", sum(len(row) for row in passage))
    with open(os.path.splitext(filen)[0] + "clean.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(passage)
    with open(os.path.splitext(filen)[0] + "clean_compress.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(passage_compress)

