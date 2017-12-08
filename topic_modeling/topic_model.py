from gensim import corpora, models, similarities
import nltk
import glob
import sys

# Example code for orientation: https://rstudio-pubs-static.s3.amazonaws.com/79360_850b2a69980c4488b1db95987a24867a.html
# Input:

def load_texts(file_list):
    raw_texts = []
    for text_file in file_list:

        with open(text_file) as infile:
            raw_texts.append(infile.read().strip())

    return raw_texts


def tokenize(raw_texts, lower_case = True):

    tokens_texts = []
    for raw_text in raw_texts:
        tokens  = [word for sent in nltk.sent_tokenize(raw_text) for word in nltk.word_tokenize(sent)]

        if lower_case == True:
            tokens_lower = [token.lower() for token in tokens]
            tokens_texts.append(tokens_lower)
        else:
            tokens_texts.append(tokens)
    return tokens_texts



def create_ids_and_counts(tokens_texts):

    # Assignems unique id to each word
    dictionary = corpora.Dictionary(tokens_texts)
    corpus = [dictionary.doc2bow(text) for text in tokens_texts]

    return dictionary, corpus


def counts_to_file(dictionary, corpus, file_list, outdir):

    for sub_corpus, sub_file in zip(corpus, file_list):

        count_file_name = 'word_counts-'+sub_file.split('/')[-1]

        with open(outdir+'/'+count_file_name, 'w') as outfile:

            for t_id_count_tuple in sub_corpus:

                t_id = t_id_count_tuple[0]
                count = t_id_count_tuple[1]

                word = dictionary[t_id]
                outfile.write(word+','+str(count)+'\n')



def create_topic_model(corpus, dictionary, topic_file_name):

    lda = models.LdaModel(corpus, num_topics=2,id2word=dictionary, update_every=5,chunksize=10000,passes=100)

    with open(topic_file_name, 'w') as outfile:
        for topic in lda.show_topics():
            outfile.write(topic)


def main(indir, outdir):

    file_list = glob.glob(indir+'/*.txt')

    raw_texts = load_texts(file_list)

    texts_tokens = tokenize(raw_texts, lower_case = True)

    dictionary, corpus = create_ids_and_counts(texts_tokens)

    counts_to_file(dictionary, corpus, file_list, outdir)

    create_topic_model(corpus, dictionary, outdir+'/model.txt')



#print(lda.show_topics())

if __name__ == '__main__':


    indir = sys.argv[1]
    outdir = sys.argv[0]

    main(indir, outdir)

    #file_list = ['test.txt']
    # mimi test corpus for testing
    #test_text = "When the binary modes of composition are thus construed, no auxiliary technique of grouping is needed. The syntactical simplicity thus gained proves useful in certain abstract studies (e.g. ß 10; also later chapters). In applications, however, such simplicity is less important than facility of reading; and excess of parentheses is a hindrance, for we have to count them off in pairs to know which ones are mates. It is hence convenient in practice to omit the outside parentheses as hitherto, and furthermore to suppress most of the remaining parentheses in favor of a more graphic notation of dots."
    #test_texts = [test_text]



    #toy_corpus = '/Users/piasommerauer/Data/toy-data/huckfinn.txt'
    #with open(toy_corpus) as infile:
    #    test_text = infile.read()



    #lda = models.LdaModel(corpus, num_topics=2,id2word=dictionary, update_every=5,chunksize=10000,passes=100)
