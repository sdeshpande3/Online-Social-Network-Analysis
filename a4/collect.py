#!/usr/bin/env python
# coding: utf-8

# In[4]:


# Imports you'll need.
from collections import Counter
import matplotlib.pyplot as plt
import networkx as nx
import sys
import time
from TwitterAPI import TwitterAPI
import pickle
import os


# In[5]:


def get_twitter():
    """ Construct an instance of TwitterAPI using the tokens you entered above.
    Returns:
      An instance of TwitterAPI.
    """
    consumer_key = 'Id0pbUg2MQPQlLp1HqVTE5XdB'
    consumer_secret = 'OJNSZfaplR2YsTUCsXnDfUfvI2JcPivlLiNDdjMIyKooaKZQAO'
    access_token = '1086684450266771458-750WadbMNdZVuRNJ2iBUeSMTfBYNTW'
    access_token_secret = 'GeNLScNFJTGOeOh7wqJit2kurRvtqSiP7I2QIcfhZrnrX'
    return TwitterAPI(consumer_key, consumer_secret, access_token, access_token_secret)


# In[6]:


def read_screen_names(filename):
    """
    Read a text file containing Twitter screen_names, one per line.

    Params:
        filename....Name of the file to read.
    Returns:
        A list of strings, one per screen_name, in the order they are listed
        in the file.
    """
    x = open(filename, "r")
    return(x.read().split())


# In[7]:


def robust_request(twitter, resource, params, max_tries=5):
    """ If a Twitter request fails, sleep for 15 minutes.
    Do this at most max_tries times before quitting.
    Args:
      twitter .... A TwitterAPI object.
      resource ... A resource string to request; e.g., "friends/ids"
      params ..... A parameter dict for the request, e.g., to specify
                   parameters like screen_name or count.
      max_tries .. The maximum number of tries to attempt.
    Returns:
      A TwitterResponse object, or None if failed.
    """
    for i in range(max_tries):
        request = twitter.request(resource, params)
        if request.status_code == 200:
            return request
        else:
            print('Got error %s \nsleeping for 15 minutes.' % request.text)
            sys.stderr.flush()
            time.sleep(61 * 15)


# In[8]:


def get_users_info(twitter, screen_names):
    """
    Retrieve the Twitter user objects for each screen_name.
    
    Params:
        twitter........The TwitterAPI object.
        screen_names...A list of strings, one per screen_name
    Returns:
        A list of dicts, one per user, containing all the user information
        (screen_name, id, friend_id)
    """

    users_info = []
    for sname in screen_names:
        request = robust_request(twitter, 'users/lookup', {'screen_name': sname}, max_tries=5)
        user = [i for i in request]
        friends = []
        request = robust_request(twitter, 'friends/ids', {'screen_name': sname, 'count': 5000}, max_tries=5)
        friends = sorted([str(i) for i in request])
        b = {'screen_name': user[0]['screen_name'],
             'id': str(user[0]['id']),
             'friend_id': friends}
        users_info.append(b)
    return users_info


# In[9]:


def get_tweets(twitter, screen_name):
    """
    Retrieve tweets of the user.
    params:
        twiiter......The TwitterAPI object.
        screen_name..The user to collect tweets from.
        num_tweets...The number of tweets to collect.
    returns:
        A list of strings, one per tweet.
    """
    tweets = []
    resource = 'search/tweets'
    for s in screen_name:
        request=robust_request(twitter,resource, {'q': s, 'lang':'en', 'count': 100})
        for t in request:
            tweets.append(t)
    print(tweets)
    return tweets


# In[10]:


def save_obj(obj, name):
    """
    store, list of dicts
    """    
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f)


# In[11]:


def main():
    print("import done.")
    twitter = get_twitter()
    print('Established Twitter connection.')
    screen_names = read_screen_names('names.txt')
    print('Read screen names:\n%s' % screen_names)
    users_info = get_users_info(twitter, screen_names)
    save_obj(users_info, 'information')
    print("information saved!")
#     print(screen_names[2])
    tweets = get_tweets(twitter, screen_names)
    save_obj(tweets, 'tweets')
#     print("%d tweets available of %s" % (len(tweets)))
    print("tweets saved!")

if __name__ == '__main__':
    main()


# In[ ]:




