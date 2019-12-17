import pickle
import sa_preprocess as sap
import numpy as np
import os
from gensim.models import Word2Vec

def sentence_w2v(input_df, model):

    # Reading the Word2Vec embedding file.
    # Lowering the input
    input_df = input_df.lower()
    # Removing punctuation
    input_df = sap.modify_phrase(input_df)
    # Lemmatization (Try google, it normally packs with stop word.)
    templist = sap.lemmatization(input_df)
    # Removing stop words (Try google, it normally packs with lemmatization)
    sentences = sap.stop_word_filter(templist)
    # If after filter, there is no word for evaluation, returns and ask for new input.
    if (len(sentences)==0):
        return (input_df,0)

    # Stripping off unicode data (special character like è, à)
    for i in range(len(sentences)):
        sentences[i] = sap.strip_accents(sentences[i])
    # Accepting a maximum of 550 words, more than that will be removed.
    maxLen=550
    # Creating a new numpy array to throw in vectors, in which it will be initially 550 words by 300 (dimension of vector is 500)
    X_vector = np.zeros((maxLen,300))

    # Assigning all vector to NAN
    X_vector[:] = np.nan

    # Assigning models vectors word by word and appending it into the X_vector that was initiated early on.
    j = 0
    for w in sentences:
        try:
            vector = model.wv[w]
            for y in range(vector.shape[0]):
                X_vector[j,y] = float(vector[y])
            j=j+1
        except Exception as e:
            continue
    return (X_vector,1)

def sentiment_analysis(emb, logisticRegr):
    sentence = input("Enter a sentence that you would like to predict its sentiment: ")
    X_Vector,type_return = sentence_w2v(sentence,emb)

    if (type_return == 0):
        print("Sorry, there is nothing to predict after filtering is done.")
        return 0
    else:
        Vec = np.zeros((1,300))
        Vec[0] = np.array(np.nanmean(X_Vector.astype('float64'), axis=0))
        if(np.isnan(Vec).any()):
            print("Sorry, there is nothing to predict after filtering is done.")
            return 0
        predicted = logisticRegr.predict(Vec)
        print("\n")
        print("======================================================================")
        print("| Sentence Inputted is predicted as", predicted,end='')
        if(predicted==0):
            print(", which represents [Negative] |")
        else:
            print(", which represents [Positive] |")
        print("======================================================================")

        return 0

if __name__== "__main__":

    emb = input("Please enter embedding file name: ")
    # If you wish to set it permanently, comment the input and then put emb with the location f of embedding file
    emb_model = emb
    emb_model = Word2Vec.load(emb_model)

    model = input("Please enter Sentiment Analysis Pre-trained Model File name: ")
    # If you wish to set it permanently, comment the input and then put model with the location f of model file
    sa_model = model
    logisticRegr = pickle.load(open(sa_model, 'rb'))
    loop = 1
    while(loop==1):
        sentiment_analysis(emb_model,logisticRegr)
        loop = int(input("\nDo you wish to continue? If you do, kindly insert 1, insert 0 if you wish to exit: \n"))
