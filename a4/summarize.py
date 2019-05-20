#!/usr/bin/env python
# coding: utf-8

# In[39]:


import pickle


# In[40]:


def load_file(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


# In[41]:


def create_info():
    file = open('summary.txt', 'w')
    file.write("*****Number of users collected:*****\n")
    file.write('\n')
    users = load_file('information')
    file.write("There are %d initial users collected.\n" % (len(users)))
    file.write("All friends of these users are also collected for future analysis.\n")
    for user in users:
        file.write("%s has %d friends.\n" % (user['screen_name'], len(user['friend_id'])))
    file.write('\n')


# In[42]:


def create_tweets():
    file = open('summary.txt', 'a')
    file.write('*****Number of messages collected:*****\n')
    file.write('\n')
    tweets = load_file('tweets')
    file.write('For sentiment analysis, we collected %d tweets (max. tweets allowed)\n' % len(tweets))
    file.write('\n')


# In[46]:


def create_cluster():
    file = open('summary.txt', 'a')
    file.write('*****Number of communities discovered:*****\n')
    file.write('\n')
    file.write('We cluster all initial users and their friends in to different communities and exclude users that followed by less than two initial users and outliers.\n')
    file.write('Outliers are those points that clustered as singleton.\n')
    clusters = load_file('clusters')
    file.write("There are %d communities\n" % (len(clusters)))
#     print(len(pruned))
    file.write('\n')
    total = 0
    for cluster in clusters:
        total += len(cluster.nodes())
    avg = total / len(clusters)
    file.write('Total number of users in the network: %d\n' % (total))
    file.write('*****Average number of users per community:*****\n %d\n' % (avg))
    file.write('\n')


# In[47]:


def create_classify():
    file = open('summary.txt', 'a')
    file.write('*****Number of instances per class found:*****\n')
    file.write('\n')
    file.write('There are three classes for sentiment analysis.\n')
    (positives,negatives,combine)  = load_file('classify')
    file.write('\nTotal positive, negative and combine tokens: %d positive instances , %d negative instances and %d combine instances'
    %(len(positives), len(negatives), len(combine)))
    print('\n')
    file.write('\n\n*****One example from each class:*****\n')
    for tweet, pos, neg in sorted(positives, key=lambda x: x[1], reverse=False):
        positive = (pos,neg,tweet)
    
    for tweet, pos, neg in sorted(negatives, key=lambda x: x[2], reverse=False):
        negative = (neg,pos,tweet)
    
    for tweet, pos, neg in sorted(combine, key=lambda x: x[2], reverse=True):
        combined = (pos,neg,tweet)
        
    file.write('\nPOSITIVE:')
    content1 = positive[2].encode('utf-8', 'ignore') 
    file.write(str(content1))
    file.write('\n')
    
    file.write('\nNEGATIVE:')
    content2 = negative[2].encode('utf-8', 'ignore') 
    file.write(str(content2))
    file.write('\n')
    
    file.write('\nNEUTRAL:')
    content3 = combined[2].encode('utf-8', 'ignore') 
    file.write(str(content3))
    
    file.close()


# In[48]:


def main():
    create_info()
    create_tweets()
    create_cluster()
    create_classify()
    print('Written!')

if __name__ == '__main__':
    main()


# In[ ]:





# In[ ]:





# In[ ]:




