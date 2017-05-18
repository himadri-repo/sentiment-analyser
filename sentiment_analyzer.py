#!/usr/bin/python 3
# Comments::
#

from nltk.sentiment import vader

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

    print('===============:: End of Program ::==================')
    wrFile.write('===============:: End of Program ::==================\n')