import re
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
from nltk import pos_tag

tknzr = TweetTokenizer(reduce_len=True, strip_handles=True)
lemm = WordNetLemmatizer()

eng_stopword_set = set(stopwords.words('english'))
url_pattern = r'(([^\s\[\]\(\)\{\}])+\w+\.\w{2,}([^\s\[\]\(\)\{\}])+)|(https?[^\s]*)'

def stop_filter(tokens: [str]) -> [str]:
    return [tok for tok in tokens if len(tok) > 1 and tok.lower() not in eng_stopword_set]

def get_wordnet_pos(pos_tag):
    pos_tag = pos_tag[0].upper()
    tag_dict = {
        "J": wordnet.ADJ,
        "N": wordnet.NOUN,
        "V": wordnet.VERB,
        "R": wordnet.ADV
    }

    return tag_dict.get(pos_tag, wordnet.NOUN)

def lemm_stem(tokens: [str]) -> [str]:
    pos_tagged = pos_tag(tokens)
    return [lemm.lemmatize(token, get_wordnet_pos(pos)) for token, pos in pos_tagged]

def preprocess_message(message: str) -> str:
    # remove urls
    message = re.sub(url_pattern, 'external_link', message)
    # filter non ascii chars
    message = ''.join([c for c in message if c.isascii()])
    # tokenize
    message = tknzr.tokenize(message)
    # remove stop words
    message = stop_filter(message)
    # lemmatize
    message = lemm_stem(message)
    # recombine
    return ' '.join(message)