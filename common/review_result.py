from nltk.sentiment import vader

def getVaderSentiment(review):
    sia = vader.SentimentIntensityAnalyzer()
    if(review is not None):
        return sia.polarity_scores(review)['compound']
    else:
        return 0

def getSentimentAnalyzed(sentimentReviewer, posReviews, negReviews):
    positiveReviewResult = [sentimentReviewer(review) for review in posReviews]
    negativeReviewResult = [sentimentReviewer(review) for review in negReviews]

    return {'positive_review_results': positiveReviewResult, 'negative_review_results': negativeReviewResult}
