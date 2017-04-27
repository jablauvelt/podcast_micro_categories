import collections

class Vocabulary1(object):
    
    def __init__(self, word_string, w2vmodel):
        self.word_dict={}
        for word in word_string.split(' '):
            if word not in self.word_dict.keys():
                if word in w2vmodel:
                    self.word_dict[word]=w2vmodel[word]

                
    