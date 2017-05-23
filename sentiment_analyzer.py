#!/usr/bin/python 3
# Comments::
#

from nltk.sentiment import vader
from common import review_result as reviewer

sia = vader.SentimentIntensityAnalyzer()

posFile = 'Data/rt-polarity.pos'
negFile = 'Data/rt-polarity.neg'

fileWithCompute = 'Data/polarity-compute.txt'

posReviews = None
negReviews = None

with open(posFile, 'r') as posfl:
    posReviews = posfl.readlines()


with open(negFile, 'r') as negfl:
    negReviews = negfl.readlines()

with open(fileWithCompute, 'w') as wrFile :
    for review in posReviews :
        print('{:-<65} {:>8}'.format(review[:45].replace('\n','') + ' ', sia.polarity_scores(review)['compound']))
        wrFile.write('{:-<65} {:>8}\n'.format(review[:45].replace('\n','') + ' ', sia.polarity_scores(review)['compound']))
    print('{:=<128}'.format(''))
    wrFile.write('{:=<128}\n'.format(''))

    for review in negReviews :
        print('{:-<65} {:>8}'.format(review[:45].replace('\n','') + ' ', sia.polarity_scores(review)['compound']))
        wrFile.write('{:-<65} {:>8}\n'.format(review[:45].replace('\n','') + ' ', sia.polarity_scores(review)['compound']))

    wrFile.write('===============:: End of Program ::==================\n')


#reviewResults = reviewer.getSentimentAnalyzed(reviewer.getVaderSentiment,
#                                              posReviews=posReviews, negReviews=negReviews)

# reviewResults = reviewer.getSentimentAnalyzed(reviewer.getSuperNaiveSentiment,
#                                               posReviews=posReviews, negReviews=negReviews)

reviewResults = reviewer.getSentimentAnalyzed(reviewer.getNaiveSentiment,
                                              posReviews=posReviews, negReviews=negReviews)

print(reviewResults.keys())
print('Length of positive reviews:{0}\nLength of negative reviews:{1}'
      .format(len(reviewResults['positive_review_results']),
                  len(reviewResults['negative_review_results'])))

print('-'*55)
posActPercentage = float(sum(x>0 for x in reviewResults['positive_review_results'])
                         /len(reviewResults['positive_review_results']))*100

negActPercentage = float(sum(x>0 for x in reviewResults['negative_review_results'])
                         /len(reviewResults['negative_review_results']))*100

print("Positive accuracy {:6.2f}%\nNegative accuracy {:6.2f}%"
      .format(posActPercentage, negActPercentage))
print('===============:: End of Program ::==================')
