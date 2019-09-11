from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyser = SentimentIntensityAnalyzer()


def sentiment_analyzer_scores(sentence):
    score = analyser.polarity_scores(sentence)
    print(f"{sentence:-<40} {str(score)}")

# https://github.com/cjhutto/vaderSentiment?source=post_page---------------------------#about-the-scoring
sentiment_analyzer_scores("The code is super cool.")
sentiment_analyzer_scores("The phone really sucks")
sentiment_analyzer_scores("Today is SUX")
sentiment_analyzer_scores("The phone absolutely :( terribly SUCKS!!")
sentiment_analyzer_scores("You are super duper paratrooper")