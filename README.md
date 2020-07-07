## Sentiment-Analysis based Classification model

Sentiment Analysis is a common NLP task that Data Scientists need to perform. This is a straightforward guide to creating a barebones movie review classifier in Python.

### Data overview
For this analysis we’ll be using a dataset of 50,000 movie reviews taken from IMDb. The data was compiled by Andrew Maas
The data is split evenly with 25k reviews intended for training and 25k for testing your classifier. Moreover, each set has 12.5k positive and 12.5k negative reviews.
I am mentioning the all the steps for completion of the process and you can check the code related to all the steps in my repository.
IMDb lets users rate movies on a scale from 1 to 10. To label these reviews the curator of the data labeled anything with ≤ 4 stars as negative and anything with ≥ 7 stars as positive. Reviews with 5 or 6 stars were left out.

### Download and Combine Movie Reviews
If you haven’t yet, go to [IMDb Reviews](http://ai.stanford.edu/~amaas/data/sentiment/) and click on “Large Movie Review Dataset v1.0”. Once that is complete you’ll have a file called aclImdb_v1.tar.gz in your downloads folder.

**Shortcut:** If you want to get straight to the data analysis and/or aren’t super comfortable with the terminal, I’ve put a tar file of the final directory that this step creates here: [Merged Movie Data](https://github.com/aaronkub/machine-learning-examples/blob/master/imdb-sentiment-analysis/movie_data.tar.gz). Double clicking this file should be sufficient to unpack it (at least on a Mac), otherwise gunzip -c movie_data.tar.gz | tar xopf — in a terminal will do it.Here I've used the tar file directly so I'll continue the process from downloading the raw data.

### Read into Python

For most of what we want to do in this walkthrough we’ll only need our reviews to be in a Python 'list' . Make sure to point 'open' to the directory where you put the movie data.

### Clean and Preprocess
The raw text is pretty messy for these reviews so before we can do any analytics we need to clean things up.  

**Note:** Understanding and being able to use regular expressions is a prerequisite for doing any Natural Language Processing task. If you’re unfamiliar with them perhaps start here: [Regex Tutorial](https://medium.com/factory-mind/regex-tutorial-a-simple-cheatsheet-by-examples-649dc1c3f285) 

### Text Processing
For our first iteration we did very basic text processing like removing punctuation and HTML tags and making everything lower-case. We can clean things up further by removing stop words and normalizing the text.
To make these transformations we’ll use libraries from the [Natural Language Toolkit](https://www.nltk.org/) (NLTK). This is a very popular NLP library for Python.

### Removing Stop Words
Stop words are the very common words like ‘if’, ‘but’, ‘we’, ‘he’, ‘she’, and ‘they’. We can usually remove these words without changing the semantics of a text and doing so often (but not always) improves the performance of a model. Removing these stop words becomes a lot more useful when we start using longer word sequences as model features (see n-grams below).

**Note:** In practice, an easier way to remove stop words is to just use the stop_words argument with any of scikit-learn’s ‘Vectorizer’ classes. If you want to use NLTK’s full list of stop words you can do stop_words='english’. In practice I’ve found that using NLTK’s list actually decreases my performance because its too expansive, so I usually supply my own list of words. For example, stop_words=['in','of','at','a','the'] .

### Normalization
A common next step in text preprocessing is to _normalize_ the words in your corpus by trying to convert all of the different forms of a given word into one. Two methods that exist for this are _Stemming_ and _Lemmatization_.

### Stemming

Stemming is considered to be the more crude/brute-force approach to normalization (although this doesn’t necessarily mean that it will perform worse). There’s several algorithms, but in general they all use basic rules to chop off the ends of words.
NLTK has several stemming algorithm implementations. We’ll use the Porter stemmer here but you can explore all of the options with examples here: [NLTK Stemmers](http://www.nltk.org/howto/stem.html)

### Lemmatization

Lemmatization works by identifying the part-of-speech of a given word and then applying more complex rules to transform the word into its true root.

### n-grams
Last time we used only single word features in our model, which we call 1-grams or unigrams. We can potentially add more predictive power to our model by adding two or three word sequences (bigrams or trigrams) as well. For example, if a review had the three word sequence “didn’t love movie” we would only consider these words individually with a unigram-only model and probably not capture that this is actually a negative sentiment because the word ‘love’ by itself is going to be highly correlated with a positive review.
The scikit-learn library makes this really easy to play around with. Just use the ngram_range argument with any of the ‘Vectorizer’ classes.

**Note:** There’s technically no limit on the size that n can be for your model, but there are several things to consider. First, increasing the number of grams will not necessarily give you better performance. Second, the size of your matrix grows exponentially as you increment n, so if you have a large corpus that is comprised of large documents your model may take a very long time to train.

### Representations

In part 1 we represented each review as a binary vector (1s and 0s) with a slot/column for every unique word in our corpus, where 1 represents that a given word was in the review.
While this simple approach can work very well, there are ways that we can encode more information into the vector.

### Word Counts

Instead of simply noting whether a word appears in the review or not, we can include the number of times a given word appears. This can give our sentiment classifier a lot more predictive power. For example, if a movie reviewer says ‘amazing’ or ‘terrible’ multiple times in a review it is considerably more probable that the review is positive or negative, respectively.

### TF-IDF

Another common way to represent each document in a corpus is to use the [tf-idf statistic](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) (term frequency-inverse document frequency) for each word, which is a weighting factor that we can use in place of binary or word count representations.
There are several ways to do tf-idf transformation but in a nutshell, tf-idf aims to represent the number of times a given word appears in a document (a movie review in our case) relative to the number of documents in the corpus that the word appears in — where words that appear in many documents have a value closer to zero and words that appear in less documents have values closer to 1.
**Note:** Now that we’ve gone over n-grams, when I refer to ‘words’ I really mean any n-gram (sequence of words) if the model is using an n greater than one.

### Algorithms

So far we’ve chosen to represent each review as a very sparse vector (lots of zeros!) with a slot for every unique n-gram in the corpus (minus n-grams that appear too often or not often enough). "Linear classifiers" typically perform better than other algorithms on data that is represented in this way.

### Support Vector Machines (SVM)

Recall that linear classifiers tend to work well on very sparse datasets (like the one we have). Another algorithm that can produce great results with a quick training time are Support Vector Machines with a linear kernel.
There are many great explanations of Support Vector Machines that do a much better job than I could. If you’re interested in learning more, this is a great tutorial: [SVM Tutorial](https://blog.statsbot.co/support-vector-machines-tutorial-c1618e635e93)

### Final Model

The goal of this post was to give you a toolbox of things to try and mix together when trying to find the right model + data transformation for your project. I found that removing a small set of stop words along with an n-gram range from 1 to 3 and a linear support vector classifier gave me the best results.
**Note:** I've uploaded all the executed code in this repository corresponding to the order of steps and concepts mentioned here.

## Summary

I’ve gone over several options for transforming text that can improve the accuracy of an NLP model. Which combination of these techniques will yield the best results will depend on the task, data representation, and algorithms you choose. It’s always a good idea to try out many different combinations to see what works.
I’m very confidant a higher accuracy on this data can be attained with a different combination of the things outlined in this post. I’ll leave that for a more ambitious reader. :) Please comment with your results and method!. 
