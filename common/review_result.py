from nltk.sentiment import vader
from nltk.corpus import sentiwordnet as swn
from string import punctuation
from nltk.corpus import stopwords

sia = vader.SentimentIntensityAnalyzer()

def getVaderSentiment(review):
    if(review is not None):
        return sia.polarity_scores(review)['compound']
    else:
        return 0

def getSentimentAnalyzed(sentimentReviewer, posReviews, negReviews):
    positiveReviewResult = [sentimentReviewer(review) for review in posReviews]
    print('Positive review processed')
    negativeReviewResult = [sentimentReviewer(review) for review in negReviews]
    print('Negative review processed')

    return {'positive_review_results': positiveReviewResult, 'negative_review_results': negativeReviewResult}

def getSuperNaiveSentiment(review):
    error = 0;
    reviewPolarity = 0.0
    #print('Review comment : {0}'.format(review.lower().split()))
    for word in review.lower().split():
        weight=0
        try:
            common_meanings = list(swn.senti_synsets(word))

            if(len(common_meanings)>0):
                common_meaning = common_meanings[0]
                # print('Word: {0}, Positive: {1}, Negative: {2}'
                #       .format(word,common_meaning.pos_score(), common_meaning.neg_score()))
                if(common_meaning.pos_score()>common_meaning.neg_score()):
                    weight = weight + common_meaning.pos_score()
                elif(common_meaning.neg_score()>common_meaning.pos_score()):
                    weight = weight - common_meaning.neg_score()
        except RuntimeError:
            error = error + 1

        reviewPolarity = reviewPolarity + weight
    return reviewPolarity

def getStopwords():
    #print(stopwords)
    stopwordList = stopwords.words('english')
    punctuationList = list(punctuation)

    return set(stopwordList + punctuationList)

def getNaiveSentiment(review):
    stopwords = getStopwords()
    numException = 0
    reviewPolarity=0.0

    words = review.lower().split()

    for word in words:
        if(word in stopwords):
            continue
        wordMeanings = list(swn.senti_synsets(word))
        weight = 0
        numMeanings = 0
        if(wordMeanings != None and len(wordMeanings) > 0):
            for wordMeaning in wordMeanings:
                if(wordMeaning.pos_score() > wordMeaning.neg_score()):
                    weight = weight + (wordMeaning.pos_score() - wordMeaning.neg_score())
                    numMeanings = numMeanings + 1
                elif(wordMeaning.neg_score() > wordMeaning.pos_score()):
                    weight = weight - (wordMeaning.neg_score() - wordMeaning.pos_score())
                    numMeanings = numMeanings + 1
            if(numMeanings>0):
                reviewPolarity = reviewPolarity + (weight/numMeanings)
        else:
            numException = numException + 1
    return reviewPolarity