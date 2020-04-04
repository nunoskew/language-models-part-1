from collections import Counter
import numpy as np

def compute_ngrams(text,n):
    if len(text)!=n:
        return [tuple(text[i:(i+n)]) for i in range(len(text)-n+1)]
    else:
        return [tuple(text)]

def compute_ngram_counts(text,n):
    if n>1:
        ngram_counts = Counter(compute_ngrams(text,n))
        context_counts = Counter(compute_ngrams(text,n-1))
    else:
        print ('N needs to be >1')
        return 0
    return ngram_counts,context_counts

def compute_ngram_probability(ngram,ngram_counts,context_counts,verbose=False):
    if verbose:
        print('ngram: ',ngram)
        print('ngram counts: ',ngram_counts[ngram])
        print('context: ',ngram[:len(ngram)-1])
        print('context counts: ',context_counts[ngram[:len(ngram)-1]])
    return ngram_counts[ngram]/context_counts[ngram[:len(ngram)-1]]

def compute_sentence_ngram_probability(s,ngram_counts,context_counts):
    s = s.split(' ')
    ngrams = compute_ngrams(s,len(list(ngram_counts.keys())[0]))
    return np.exp(np.sum([np.log(compute_ngram_probability(ngrams[i],ngram_counts,context_counts)) for i in range(len(ngrams))]))
# understand perplexity of a model
def compute_perplexity(text,ngram_counts,context_counts):
    ngrams = compute_ngrams(text,len(list(ngram_counts.keys())[0]))
    return np.exp(-(1/len(text))*np.sum([np.log(compute_ngram_probability(ngrams[i],ngram_counts,context_counts)) for i in range(len(ngrams))]))

def unk_encode_text_train_test(min_count,train_text,test_text=[]):
    train_unigram_counts = Counter(train_text)
    train_rare_words = [w for w in list(train_unigram_counts.keys()) if train_unigram_counts[w]<=min_count]
    train_text = ['<UNK>' if w in train_rare_words else w for w in train_text ]
    test_text = ['<UNK>' if w not in list(train_unigram_counts.keys()) else w for w in test_text]
    return train_text,test_text

def laplace_bigram_smoothing_counts(bigram_counts,unigram_counts,k=1):
    vocabulary = [l[0] for l in list(unigram_counts.keys())]
    all_bigrams  = list(itertools.product(vocabulary,vocabulary))
    all_bigram_counts = Counter(all_bigrams)
    all_bigram_counts.update(Counter(dict.fromkeys(all_bigram_counts,k-1)))
    all_bigram_counts.update(bigram_counts)
    unigram_counts.update(Counter(dict.fromkeys(unigram_counts,len(vocabulary))))
    return all_bigram_counts,unigram_counts

def sample_word_from_context(context_word,ngram_counts):
    ngrams = list(ngram_counts.keys())
    context_ngrams = np.array([ngram for ngram in ngrams if ngram[0]==context_word])
    next_possible_words = np.array([ngram[1] for ngram in context_ngrams])
    context_probs = np.array([ngram_counts[ngram] for ngram in ngrams if ngram[0]==context_word],dtype='float')
    context_probs/=np.sum(context_probs)
    return np.random.choice(a=next_possible_words,size=1,p=context_probs)[0]

def generate_random_sentence(seed_word,ngram_counts,stop_word='.',sentence_max_length=300,):
    sentence = seed_word
    next_word = sample_word_from_context(context_word=seed_word,ngram_counts=ngram_counts)
    sentence += ' '+next_word
    while next_word not in ('.','!','?') and len(sentence.split(' '))<sentence_max_length:
        next_word = sample_word_from_context(context_word=next_word,ngram_counts=ngram_counts)
        sentence += ' '+next_word
    print("Random Sentence:",sentence)
