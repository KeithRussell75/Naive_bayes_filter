#Russell Keith
#Using python 3.9

# Import libraries
import numpy as np
import pandas as pd
import argparse
import string

#downloader for autograder
import nltk
nltk.download("popular")
#libaries to delete stop words
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
#for lemmatizer
from nltk.stem import WordNetLemmatizer
#for stemming
from nltk.stem import PorterStemmer

class NaiveBayesFilter:
    def __init__(self, test_set_path):
        self.vocabulary = pd.DataFrame(columns=['word', 'wi_spam', 'wi_ham'])
        self.training_set= None
        self.test_set = None
        self.p_spam = pd.DataFrame(columns=['word'])
        self.p_ham = pd.DataFrame(columns=['word'])
        self.StartHam = None
        self.StartSpam = None
        self.test_set_path = test_set_path

    def read_csv(self):
        self.training_set = pd.read_csv('train.csv', sep=',', header=0, names=['v1', 'v2'], encoding = 'utf-8')
        self.test_set = pd.read_csv(self.test_set_path, sep=',', header=0, names=['v1', 'v2'], encoding = 'utf-8')


    def data_cleaning(self):
        # Normalization
        # Replace addresses (hhtp, email), numbers (plain, phone), money symbols
        unacceptableAddresses = ['http','www','.com']
        
        acceptableChars = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]
        #parameter for stop words
        stopWords = set(stopwords.words('english'))
        #for seperation
        Hi = 0
        Si = 0
        #for vocab dataframe
        setVocab = set()

        for i,j in self.training_set.iterrows():
            
            currentString = self.training_set.at[i,"v2"]
            currentString = currentString.translate(str.maketrans('', '', string.punctuation)).lower()

            #check if word contains the unacceptable addresses and replace
            for word in currentString.split(" "):
                if(any(words in word for words in unacceptableAddresses)):
                    currentString = currentString.replace(word,"")
                    
                else:
                    #edit for only acceptable characters only
                    currentString = currentString.translate(str.maketrans('', '', string.punctuation))
                    ogWord = word
                    for letter in word:
                        if (not (letter in acceptableChars)):
                            word = word.replace(letter,"")

                    currentString = currentString.replace(ogWord, word)

            # Remove the stop-words #does not remove names
            wordTokens = word_tokenize(currentString)

            # csList creats a list of for the new string 
            # makes everything lowercase
            csList = [word for word in wordTokens if not word.lower() in stopWords]
            newString = ''
            for x in csList:
                newString += ' ' + x
            currentString = newString

            # Lemmatization - Graduate Students
            lemma = WordNetLemmatizer()
            for word in currentString.split():
                lemmaWord = lemma.lemmatize(word)
                currentString = currentString.replace(word, lemmaWord)

            # Stemming - Gradutate Students
            stemmer = PorterStemmer()
            for word in currentString.split():
                stemWord = stemmer.stem(word)
                currentString = currentString.replace(word, stemWord)

            # Tokenization and vectorization
            stringTokenized = word_tokenize(currentString)

            # Vectorization

            # Remove duplicates - Can you think of any data structure that can help you remove duplicates?
            # sets do not allow word repeats
            setVocab.update(stringTokenized)

            # Convert to dataframe 
            self.training_set.at[i, "v2"] = stringTokenized

            # Separate the spam and ham dataframes

            if (j["v1"] == "ham"):
                for word in stringTokenized:
                    self.p_ham.at[Hi, "word"] = word
                    Hi = Hi + 1
            else:
                for word in stringTokenized:
                    self.p_spam.at[Si, "word"] = word
                    Si = Si + 1

        #turn vocabulary into a dataframe col , word, WiSpam, WiHam
        setVocab = list(setVocab)
        index = 0
        for val in setVocab:
            self.vocabulary.at[index , 'word'] = val
            index = index + 1
        pass

    def fit_bayes(self):
        #calculate the number of word repeats in spam
        headers = ['fqe']
        SwordCount = self.p_spam.pivot_table(index=['word'], aggfunc='size')
        HwordCount = self.p_ham.pivot_table(index=['word'], aggfunc='size')
        dfSC = SwordCount.to_frame()
        dfHC = HwordCount.to_frame()

        dfSC.columns = headers
        dfHC.columns = headers

        #delete words that were only mentioned a little *adjustment*
        for i,j in self.vocabulary.iterrows():
            if j["word"] in dfSC.index.values and j["word"] in dfHC.index.values:
                if dfSC.at[j["word"], "fqe"] <= 15 and dfHC.at[j["word"], "fqe"] <= 15:
                    dfSC = dfSC.drop(labels = [j["word"]], axis = 0)
                    dfHC = dfHC.drop(labels = [j["word"]], axis = 0)
                    vindex = self.vocabulary.index[self.vocabulary['word']== j["word"]]
                    self.vocabulary = self.vocabulary.drop(index = vindex, axis=0)

            elif j["word"] in dfHC.index.values:
                if dfHC.at[j["word"], "fqe"] <= 3:
                    dfHC = dfHC.drop(labels = [j["word"]], axis = 0)
                    vindex = self.vocabulary.index[self.vocabulary['word']== j["word"]]
                    self.vocabulary = self.vocabulary.drop(index = vindex, axis=0)

            elif j["word"] in dfSC.index.values:
                if dfSC.at[j["word"], "fqe"] <= 3:
                    dfSC = dfSC.drop(labels = [j["word"]], axis = 0)
                    vindex = self.vocabulary.index[self.vocabulary['word']== j["word"]]
                    self.vocabulary = self.vocabulary.drop(index = vindex, axis=0)

        #delete words that made it past data cleaning
        dfSC.drop(index=dfSC.index[:3],inplace=True)
        self.vocabulary = self.vocabulary.sort_values(by=['word'])
        self.vocabulary.drop(index=self.vocabulary.index[:2],inplace=True)
        
        # Calculate P(Spam) and P(Ham)
        self.StartHam = self.test_set["v1"].value_counts()["ham"]/len(self.test_set)
        self.StartSpam = self.test_set["v1"].value_counts()["spam"]/len(self.test_set)
        
        # Calculate Nspam, Nham and Nvocabulary
        Nham = 0
        Nspam = 0
        for i,j in dfSC.iterrows():
            Nspam = j["fqe"] + Nspam
        for i,j in dfHC.iterrows():
            Nham = j["fqe"] + Nham
        Nvocabulary = len(self.vocabulary)

        # Laplace smoothing parameter
        alpha = .5

        # Calculate P(wi|Spam) and P(wi|Ham)
        for i,j in self.vocabulary.iterrows():
            if j["word"] in dfSC.index.values and j["word"] in dfHC.index.values:
                self.vocabulary.at[i, "wi_spam"]= (dfSC.at[j["word"], "fqe"] + alpha)/(Nspam + (alpha * Nvocabulary))
                self.vocabulary.at[i, "wi_ham"] = (dfHC.at[j["word"], "fqe"]  + alpha)/(Nspam + (alpha * Nvocabulary))
            else:
                if j["word"] in dfSC.index.values:
                    self.vocabulary.at[i, "wi_spam"]= (dfSC.at[j["word"], "fqe"] + alpha)/(Nspam + (alpha * Nvocabulary))
                    self.vocabulary.at[i, "wi_ham"] = 0
                elif j["word"] in dfHC.index.values:
                    self.vocabulary.at[i, "wi_ham"] = (dfHC.at[j["word"], "fqe"]  + alpha)/(Nspam + (alpha * Nvocabulary))
                    self.vocabulary.at[i, "wi_spam"] = 0
                else:
                    self.vocabulary.at[i, "wi_ham"] = 0
                    self.vocabulary.at[i, "wi_spam"] = 0


    def train(self):
        self.read_csv()
        self.data_cleaning()
        self.fit_bayes()
    
    def sms_classify(self, message):
        '''
        classifies a single message as spam or ham
        Takes in as input a new sms (w1, w2, ..., wn),
        performs the same data cleaning steps as in the training set,
        calculates P(Spam|w1, w2, ..., wn) and P(Ham|w1, w2, ..., wn),
        compares them and outcomes whether the message is spam or not.
        '''
        #
        message = message.translate(str.maketrans('', '', string.punctuation)).lower()


        #message = message.lower()
        stopWords = set(stopwords.words('english'))
        wordTokens = word_tokenize(message)

        # csList creats a list of for the new string 
        # makes everything lowercase
        csList = [word for word in wordTokens if not word.lower() in stopWords]
        newString = ''
        for x in csList:
            newString += ' ' + x
        message = newString

        # Lemmatization - Graduate Students
        lemma = WordNetLemmatizer()
        for word in message.split():
            lemmaWord = lemma.lemmatize(word)
            message = message.replace(word, lemmaWord)

        # Stemming - Gradutate Students
        stemmer = PorterStemmer()
        for word in message.split():
            stemWord = stemmer.stem(word)
            message = message.replace(word, stemWord)


        p_messageS = self.StartSpam
        p_messageH = self.StartHam

        message = message.split()

        #comparison for spam and ham messages
        for word in message:
            Vindex = self.vocabulary.index[self.vocabulary['word'] == word].to_list()
            if len(Vindex) != 0 :
                p_messageS = self.vocabulary.at[Vindex[0], "wi_spam"] *  p_messageS
                p_messageH = self.vocabulary.at[Vindex[0], "wi_ham"] *  p_messageH

        if p_messageH > p_messageS:
            return 'ham'
        elif p_messageH < p_messageS:
            return 'spam'
        else:
            return 'needs human classification'

        pass

    def classify_test(self):
        '''
        Calculate the accuracy of the algorithm on the test set and returns 
        the accuracy as a percentage.
        '''
        self.train()
        #variables for evaluation
        accuracy = 0
        TNumS = 0
        TNumH = 0
        PNumS = 0
        PNumH = 0
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        for i,j in self.test_set.iterrows():

            testClass = self.test_set.at[i, "v1"]
            if testClass == "ham":
                TNumH = TNumH +1
            else:
                TNumS = TNumS +1

            Mclass = self.sms_classify(self.test_set.at[i, "v2"])
            if Mclass == 'ham':
                PNumH = PNumH +1
            elif Mclass == 'spam':
                PNumS = PNumS +1

            if testClass == Mclass and testClass == "spam":
                TP = TP +1
            elif testClass == Mclass and testClass == "ham":
                TN = TN +1
            elif testClass == "spam" and Mclass == "ham":
                FN = FN +1
            elif testClass == "ham" and Mclass == "spam":
                FP = FP +1

        #accuacy based on true positive/negative and false positive/negative
        #print(TNumS, PNumS) 
        #print(TNumH, PNumH)
        #print(TP, TN) 
        #print(FP, FN)
        #accuracy = (((TNumS + PNumS)/TNumS)-1) * 100
        accuracy = (TP +TN)/(TP + TN + FP + FN) * 100
        return accuracy


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Naive Bayes Classifier')
    parser.add_argument('--test_dataset', type=str, default = "test.csv", help='path to test dataset')
    args = parser.parse_args()
    classifier = NaiveBayesFilter(args.test_dataset)
    acc = classifier.classify_test()
    print("Accuracy: ", acc)
