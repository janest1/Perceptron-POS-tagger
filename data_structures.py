from sparse_vector import Vector


class Sentence(object):
    def __init__(self, snt):
        ''' Modify if necessary.
        '''
        self.snt = snt
        self.feats = []
        return

    def features(self,):
        ''' Implement your features here.
        '''

        # for i in range(len(self.snt)):
        #     self.words[self.snt[i]] = {}
        #
        #     curr_word_vec = np.zeros(len(self.vocab))
        #     curr_word_vec[self.vocab.indexof(self.snt[i][0])] = 1
        #
        #     prev_word_vec = np.zeros(len(self.vocab))
        #     prev_tag_vec = np.zeros(len(self.tags))
        #     if i == 0:
        #         # prev word and tag = start symbol
        #         prev_word_vec[0] = 1
        #         prev_tag_vec[0] = 1
        #     else:
        #         prev = self.snt[i-1]
        #         prev_word_vec[self.vocab.indexof(prev[0])] = 1
        #         prev_tag_vec[self.tags.indexof(prev[1])] = 1
        #
        #     self.words[self.snt[i]]['curr_word_vec'] = curr_word_vec
        #     self.words[self.snt[i]]['prev_word_vec'] = prev_word_vec
        #     self.words[self.snt[i]]['prev_tag_vec'] = prev_tag_vec

