import nltk
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import unicodedata
import string
nltk.download('punkt');nltk.download('averaged_perceptron_tagger');nltk.download('wordnet');nltk.download('stopwords');

def get_wordnet_pos(word):
 tag = nltk.pos_tag([word])[0][1][0].upper()
 tag_dict ={"J": wordnet.ADJ,
  "N": wordnet.NOUN,
  "V": wordnet.VERB,
  "R": wordnet.ADV
 }
 return tag_dict.get(tag,wordnet.NOUN)

def lemmatization(paragraph):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(w,get_wordnet_pos(w)) for w in nltk.word_tokenize(paragraph)]

def stop_word_filter(word_tokens):
    stop_words = set(stopwords.words('english'))
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    filtered_sentence = []
    for w in word_tokens:
        if w not in stop_words:
            filtered_sentence.append(w)
    return filtered_sentence

def strip_accents(text):

    return ''.join(c for c in unicodedata.normalize('NFKD', text) if unicodedata.category(c) != 'Mn')

def modify_phrase(ph):
    tr = str.maketrans(string.punctuation, " "*32)
    ph = ph.lower()
    return ph.translate(tr).strip()
