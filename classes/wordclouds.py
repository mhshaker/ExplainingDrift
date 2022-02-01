from gensim.utils import simple_preprocess
from wordcloud import WordCloud, STOPWORDS
from collections import Counter

class Wordclouds:

    # https://radimrehurek.com/gensim/utils.html#gensim.utils.simple_preprocess
    def get_tokens(indexes, texts, stopwords = set(STOPWORDS)):
        tokens = []
        for index in indexes:
            tokens += simple_preprocess(texts[index])
        tokens = [w for w in tokens if w not in stopwords]
        return Counter(tokens)

    def remove_tokens(counter, counter_remove, factor=2):
        dict_ = {}
        for token in counter.keys():
            if(token in counter_remove):
                if(counter[token] >= counter_remove[token] * factor):
                    dict_[token] = counter[token]
        return dict_

    def print_wordcould(counts):
        font_path='/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf' #  fc-list | grep 'NotoSans-Bold'
        wordcloud = WordCloud(background_color="white", font_path=font_path, colormap='Dark2', width=1200, height=800).generate_from_frequencies(counts)
        plt.imshow(wordcloud)
        plt.axis("off")