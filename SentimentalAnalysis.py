from nltk.corpus import stopwords, twitter_samples
import numpy as np
import pandas as pd
import nltk
import string
from nltk.tokenize import TweetTokenizer
import re
from nltk.stem import PorterStemmer          

nltk.download('stopwords')
nltk.download('twitter_samples')

class Twitter_Classifier():
    def __init__(self):
        pos = twitter_samples.strings('positive_tweets.json')
        neg = twitter_samples.strings('negative_tweets.json')
        self.x = pos + neg
        self.y = np.append(np.ones(len(pos)), np.zeros(len(neg)))
        self.freqs = self.count_tweets()
        self.logprior, self.loglikelihood = self.NB_train()

    def process_tweet(self, tweet):
            tweet2 = re.sub(r'^RT[\s]+', '', tweet)
            # remove hyperlinks
            tweet2 = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet2)
            # remove hashtags
            # only removing the hash # sign from the word
            tweet2 = re.sub(r'#', '', tweet2)
            # instantiate tokenizer class
            tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                               reduce_len=True)
            # tokenize tweets
            tweet_tokens = tokenizer.tokenize(tweet2)
            stopwords_english = stopwords.words('english')
            tweets_clean = []
            for word in tweet_tokens: # Go through every word in your tokens list
                if (word not in stopwords_english and word not in string.punctuation):  # remove stop words and punctuation
                    tweets_clean.append(word) 
            # Instantiate stemming class
            stemmer = PorterStemmer() 

            # Create an empty list to store the stems
            tweets_stem = [] 

            for word in tweets_clean:
                stem_word = stemmer.stem(word)  # stemming word
                tweets_stem.append(stem_word)  # append to the list
            return tweets_stem

    def count_tweets(self):
        result = {}
        for ys, tweet in zip(self.y, self.x):
            for word in self.process_tweet(tweet):
                # define the key, which is the word and label tuple
                pair = (word, ys)

                # if the key exists in the dictionary, increment the count
                if pair in result:
                    result[pair] += 1

                # else, if the key is new, add it to the dictionary and set the count to 1
                else:
                    result[pair] = 1
        return result

    def NB_train(self):
        loglikelihood = {}
        logprior = 0

        # calculate V, the number of unique words in the vocabulary
        vocab = set([pair[0] for pair in self.freqs.keys()])
        V = len(vocab)

        # calculate N_pos and N_neg
        N_pos = N_neg = 0
        for pair in self.freqs.keys():
            # if the label is positive (greater than zero)
            if pair[1] > 0:

                # Increment the number of positive words by the count for this (word, label) pair
                N_pos += self.freqs[pair]

            # else, the label is negative
            else:

                # increment the number of negative words by the count for this (word,label) pair
                N_neg += self.freqs[pair]

        # Calculate D, the number of documents
        D = len(self.x)

        # Calculate D_pos, the number of positive documents (*hint: use sum(<np_array>))
        D_pos = sum(self.y)

        # Calculate D_neg, the number of negative documents (*hint: compute using D and D_pos)
        D_neg = D - D_pos

        # Calculate logprior
        logprior = np.log(D_pos) - np.log(D_neg)

        # For each word in the vocabulary...
        for word in vocab:
            # get the positive and negative frequency of the word
            freq_pos = self.freqs.get((word, 1),0)
            freq_neg = self.freqs.get((word, 0),0)

            # calculate the probability that each word is positive, and negative
            p_w_pos = (freq_pos + 1)/(V + N_pos)
            p_w_neg = (freq_neg + 1)/(V + N_neg)

            # calculate the log likelihood of the word
            loglikelihood[word] = np.log(p_w_pos) - np.log(p_w_neg)

        return logprior, loglikelihood


    def NB_predict(self, tweet):
        
        # process the tweet to get a list of words
        word_l = self.process_tweet(tweet)
        pos,neg = 0,0

        # initialize probability to zero
        p = 0

        # add the logprior
        p += self.logprior

        for word in word_l:

            # check if the word exists in the loglikelihood dictionary
            if word in self.loglikelihood:
                # add the log likelihood of that word to the probability
                p += self.loglikelihood[word]
                if self.loglikelihood[word] > 0:
                    pos += 1
                elif self.loglikelihood[word] < 0:
                    neg += 1
                else:
                    pass 
        return p,pos,neg
