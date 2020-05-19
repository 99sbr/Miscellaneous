import re

import contractions
import pandas as pd
import pyLDAvis.gensim
import spacy
from gensim import models, corpora
from nltk.corpus import stopwords

nlp = spacy.load('en_core_web_lg', disable=['parser', 'ner'])
stop_words = stopwords.words('english')


def clean_up(text):
    removal = ['ADV', 'PRON', 'CCONJ', 'PUNCT', 'PART', 'DET', 'ADP', 'SPACE']
    text_out = []
    doc = nlp(text)
    for token in doc:
        if token.is_stop == False and token.is_alpha and len(token) > 2 and token.pos_ not in removal:
            lemma = token.lemma_
            text_out.append(lemma)
    return text_out


def topic_modelling(data):
    datalist = data.text.apply(lambda x: clean_up(x))
    dictionary = corpora.Dictionary(datalist)
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in datalist]

    num_topics = 8
    Lda = models.LdaMulticore
    ldamodel = Lda(doc_term_matrix,
                   num_topics=num_topics,
                   id2word=dictionary,
                   passes=20,
                   iterations=100,
                   chunksize=10000,
                   eval_every=10)
    topic_data = pyLDAvis.gensim.prepare(
        ldamodel, doc_term_matrix, dictionary, mds='tsne')
    all_topics = {}
    lambd = 0.5  # Adjust this accordingly
    for i in range(1, 8):  # Adjust this to reflect number of topics chosen for final LDA model
        topic = topic_data.topic_info[topic_data.topic_info.Category ==
                                      'Topic' + str(i)]
        topic['relevance'] = topic['loglift'] * \
            (1 - lambd) + topic['logprob'] * lambd
        all_topics['Topic ' + str(i)] = topic.sort_values(
            by='relevance', ascending=False).Term[:5].values

    lda_topics = pd.DataFrame(all_topics).T
    return lda_topics


def rightTypes(ngram):
    # function to filter for ADJ/NN bigrams
    """

    :param ngram:
    :return:
    """
    if '-pron-' in ngram or '' in ngram or ' ' in ngram or 't' in ngram:
        return False
    for word in ngram:
        if word in stop_words:
            return False
    acceptable_types = ('JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS')
    second_type = ('NN', 'NNS', 'NNP', 'NNPS')
    tags = nltk.pos_tag(ngram)
    if tags[0][1] in acceptable_types and tags[1][1] in second_type:
        return True
    else:
        return False


def rightTypesTri(ngram):
    """

    :param ngram:
    :return:
    """
    if '-pron-' in ngram or '' in ngram or ' ' in ngram or '  ' in ngram or 't' in ngram:
        return False
    for word in ngram:
        if word in stop_words:
            return False
    first_type = ('JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS')
    third_type = ('JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS')
    tags = nltk.pos_tag(ngram)
    if tags[0][1] in first_type and tags[2][1] in third_type:
        return True
    else:
        return False


def data_cleaning(data):
    """

    :param data: df['clean_text']
    :return: unlist_clean_text
    """

    # function to remove non-ascii characters
    def _removeNonAscii(s): return "".join(i for i in s if ord(i) < 128)

    data.clean_text = data.clean_text.map(lambda x: _removeNonAscii(x))

    # remove url
    data['clean_text'] = data['clean_text'].apply(
        lambda x: re.sub(r'http\S+', '', x))

    # resolve contractions
    data['clean_text'] = data['clean_text'].apply(
        lambda x: [contractions.fix(word) for word in x.split()])
    data['clean_text'] = data['clean_text'].apply(lambda x: ' '.join(x))

    # replace special chars
    data['clean_text'] = data['clean_text'].str.replace("[^a-zA-Z0-9]", " ")

    data.clean_text = data.clean_text.str.split()
    # turn all text' tokens into one single list
    unlist_clean_text = [item for items in data.clean_text for item in items]
    return unlist_clean_text


def load_data(data_path):
    """

    :param data_path:
    :return:
    """
    data = pd.read_csv(data_path, memory_map=True)
    data = data[['text']]
    data['clean_text'] = data.text
    return data


def freq_trigram_finder(trigramFinder):
    trigram_freq = trigramFinder.ngram_fd.items()
    trigramFreqTable = pd.DataFrame(list(trigram_freq), columns=['trigram', 'freq']).sort_values(by='freq',
                                                                                                 ascending=False)

    trigramFreqTable.reset_index(drop=True, inplace=True)
    filtered_tri = trigramFreqTable[trigramFreqTable.trigram.map(
        lambda x: rightTypesTri(x))]
    freq_tri = filtered_tri[:20].trigram.values
    return freq_tri


def pmi_trigram_finder(trigramFinder, trigrams):
    trigramFinder.apply_freq_filter(20)
    trigramPMITable = pd.DataFrame(list(trigramFinder.score_ngrams(trigrams.pmi)),
                                   columns=['trigram', 'PMI']).sort_values(by='PMI', ascending=False)
    pmi_tri = trigramPMITable[:20].trigram.values
    return pmi_tri


def ttest_trigram_finder(trigramFinder, trigrams):
    trigramTtable = pd.DataFrame(list(trigramFinder.score_ngrams(trigrams.student_t)),
                                 columns=['trigram', 't']).sort_values(by='t', ascending=False)
    filteredT_tri = trigramTtable[trigramTtable.trigram.map(
        lambda x: rightTypesTri(x))]
    t_tri = filteredT_tri[:20].trigram.values
    return t_tri


def chi_trigram_finder(trigramFinder, trigrams):
    trigramChiTable = pd.DataFrame(list(trigramFinder.score_ngrams(trigrams.chi_sq)),
                                   columns=['trigram', 'chi-sq']).sort_values(by='chi-sq', ascending=False)
    chi_tri = trigramChiTable[:20].trigram.values
    return chi_tri


def likelihood_trigram_finder(trigramFinder, trigrams):
    trigramLikTable = pd.DataFrame(list(trigramFinder.score_ngrams(trigrams.likelihood_ratio)),
                                   columns=['trigram', 'likelihood ratio']).sort_values(by='likelihood ratio',
                                                                                        ascending=False)
    filteredLik_tri = trigramLikTable[trigramLikTable.trigram.map(
        lambda x: rightTypesTri(x))]
    lik_tri = filteredLik_tri[:20].trigram.values
    return lik_tri


if __name__ == "__main__":
    import nltk

    print('\n **Loading and Preprocessing Data** \n ')
    try:
        data_path = "data - data.csv"
        data = load_data(data_path)
    except Exception as e:
        print(e)
        exit(0)
    unlist_text = data_cleaning(data)
    print('\n ** Vocabulary Size:{} ** \n '.format(len(unlist_text)))
    bigrams = nltk.collocations.BigramAssocMeasures()
    trigrams = nltk.collocations.TrigramAssocMeasures()
    bigramFinder = nltk.collocations.BigramCollocationFinder.from_words(
        unlist_text)
    trigramFinder = nltk.collocations.TrigramCollocationFinder.from_words(
        unlist_text)

    print('\n ** Phrase Extraction in Progress ** \n ')
    freq_tri = freq_trigram_finder(trigramFinder)
    pmi_tri = pmi_trigram_finder(trigramFinder, trigrams)
    t_tri = ttest_trigram_finder(trigramFinder, trigrams)
    chi_tri = chi_trigram_finder(trigramFinder, trigrams)
    lik_tri = likelihood_trigram_finder(trigramFinder, trigrams)

    print('\n ** Compiling Results ** \n')
    trigramsCompare = pd.DataFrame(
        [freq_tri, pmi_tri, t_tri, chi_tri, lik_tri]).T
    trigramsCompare.columns = ['Frequency With Filter', 'PMI', 'T-test With Filter', 'Chi-Sq Test',
                               'Likelihood Ratio Test With Filter']

    COL_ORDER = ['PMI', 'Chi-Sq Test', 'Likelihood Ratio Test With Filter', 'T-test With Filter',
                 'Frequency With Filter']
    from ordered_set import OrderedSet

    trigram_set = OrderedSet()
    for col in COL_ORDER:
        x = [trigram_set.add(x) for x in trigramsCompare[col].values]

    top_results = [' '.join(x) for x in list(trigram_set)]
    print('\n ** Displaying Top {} Topics Results from Corpus** \n'.format(50))
    # Displaying Top-K results
    print(top_results[:50])

    print('\n ** LDA: Topic Modelling in Progress ** \n')
    lda_topics = topic_modelling(data)

    print("\n ** Some Other Relevant Topics in Discussion ** \n:")
    print(lda_topics)
