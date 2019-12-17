# Sentiment Analysis Using Python

A program that is able to predicts its sentiment (Negative or Positive) with the help of embeddings.

Steps to run the program
1. Clone to repository
2. Download the embedding file from [this link](https://drive.google.com/open?id=1LuIupesbpQzQ5YCX8jQN5cGtqx80dW_e), why? [Github does not allow file larger than 100MB to be uploaded.]
3. Move the downloaded file to the repository file location
4. Run <b>sa_preprocess.py</b>
   - It may takes a minute or two before first line was inputted, because it was importing all used library.
5. Follow instruction given in the file.

Files Explanation
1. sa_main.py
   - Main file to run and acts as function caller.
2. sa_preprocess.py
   - Pre-processing file (Filtering function), waiting to be called by sa_main.py
3. amazon_rt_sentiment.model
   - Sentiment Analysis Pre-trained Model File (Trained with rotten tomatoes sentiment file)
4. amazon_embedding_v3[xxx]
   - Embedding file, trained with amazon product data, all 3 file must be place in same place because it runs by default combining.
