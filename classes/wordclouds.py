# Class to extract tokens out of texts and to create wordclouds.
# Author: https://github.com/adibaba

from gensim.utils import simple_preprocess
from wordcloud import WordCloud, STOPWORDS
from collections import Counter
import matplotlib.pyplot as plt
import os.path

class Wordcloud:

    # https://radimrehurek.com/gensim/utils.html#gensim.utils.simple_preprocess
    def get_tokens(self, texts, stopwords=STOPWORDS, min_len=2, max_len=15):
        tokens = []
        for text in texts:
            tokens += simple_preprocess(text, min_len=min_len, max_len=max_len)
        tokens = [w for w in tokens if w not in stopwords]
        return Counter(tokens)

    def remove_tokens(self, counter, counter_remove, factor=2):
        dict_ = {}
        for token in counter.keys():
            if(token in counter_remove):
                if(counter[token] >= counter_remove[token] * factor):
                    dict_[token] = counter[token]
        return Counter(dict_)

    # Parameters: https://amueller.github.io/word_cloud/generated/wordcloud.WordCloud.html
    def get_wordcloud(self, counts, parameters={}):
        # Implementation: https://amueller.github.io/word_cloud/_modules/wordcloud/wordcloud.html

        # width  : int (default=400) | Width of the canvas.
        # height : int (default=200) | Height of the canvas.
        if not 'width' in parameters:
            parameters['width']  = 400 # int(16 / 4 * 100)
        if not 'height' in parameters:
            parameters['height'] = 225 # int( 9 / 4 * 100)

        # scale : float (default=1)
        # "Larger canvases with make the code significantly slower. If you need a
        #  large word cloud, try a lower canvas size, and set the scale parameter."
        if not 'scale' in parameters:
            parameters['scale'] = 4

        # background_color : color value (default="black")
        if not 'background_color' in parameters:
            parameters['background_color'] = 'white'

        # Readable colors on white background
        # https://matplotlib.org/stable/gallery/color/colormap_reference.html
        if not 'colormap' in parameters:
            parameters['colormap'] = 'Dark2'

        # Do not use non-existing fonts
        if 'font_path' in parameters:
            if not os.path.isfile(parameters['font_path']):
                del parameters['font_path']

        # Default font
        # Check if installed on debian: fc-list | grep 'RobotoCondensed-Medium.ttf'
        # Install on debian: sudo apt-get -y install fonts-roboto
        # Download font: https://fonts.google.com/specimen/Roboto+Condensed
        if not 'font_path' in parameters:
            font_path = '/usr/share/fonts/truetype/roboto/unhinted/RobotoCondensed-Medium.ttf'
            if os.path.isfile(font_path):
                parameters['font_path'] = font_path

        return WordCloud(**parameters).generate_from_frequencies(counts)
    
    def plot(self, wordcloud, file_name="wordcloud", figsize=[4.0, 2.25], dpi=150, axs=-1, axs_index=-1):
        # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.figure.html
        # matplotlib.pyplot.figure: figsize(float, float), default: rcParams["figure.figsize"] (default: [6.4, 4.8])
        # print(16/4, 9/4) # 4.0 2.25 # small for jupyter notebools
        # print(16/2, 9/2) # 8.0 4.5  # large for external use
        if axs_index==-1:
            plt.rcParams['figure.figsize'] = figsize

            # matplotlib.pyplot.figure: dpifloat, default: rcParams["figure.dpi"] (default: 100.0)
            plt.rcParams['figure.dpi'] = dpi

            # https://amueller.github.io/word_cloud/auto_examples/simple.html#sphx-glr-auto-examples-simple-py
            plt.figure()
            plt.imshow(wordcloud, interpolation="bilinear")
            plt.axis("off")
            plt.savefig(f"./Results/{file_name}.png")
        else:
            # axs[axs_index].rcParams['figure.figsize'] = figsize
            # axs[axs_index].rcParams['figure.dpi'] = dpi
            axs[axs_index[0]][axs_index[1]].imshow(wordcloud, interpolation="bilinear")
            axs[axs_index[0]][axs_index[1]].axis("off")
            axs[axs_index[0]][axs_index[1]].set_title('Axis [0, 0]')
            return axs
