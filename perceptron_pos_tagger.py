from sparse_vector import Vector
from collections import defaultdict
import random


class Perceptron_POS_Tagger(object):
    def __init__(self, tags):
        ''' Modify if necessary. 
        '''

        self.tags = tags
        self.weights = Vector({})

    def featurize(self, curr_word, curr_tag, prev_word, prev_tag):
        """"""
        vector = Vector({})
        bi_word = 'w-1_{} w0_{}'.format(prev_word, curr_word)
        vector.v[bi_word] = 1
        bi_tag = 't-1_{} t0_{}'.format(prev_tag, curr_tag)
        vector.v[bi_tag] = 1
        emission = 'w0_{} t0_{}'.format(curr_word, curr_tag)
        vector.v[emission] = 1

        return vector

    def compute_accuracy(self, gold, auto):
        print('compute accuracy')

        correct = 0.0
        total = 0.0

        for g_snt, a_snt in zip(gold, auto):
            correct += sum([g_tup[1] == a_tup[1] for g_tup, a_tup in zip(g_snt, a_snt)])
            total += len(g_snt)

        return correct / total

    def get_sentence_features(self, tagged_sentence):
        sentence_vector = self.featurize(tagged_sentence[0][0], tagged_sentence[0][1], '$START', '<S>')

        for i in range(1, len(tagged_sentence)):
            sentence_vector += self.featurize(tagged_sentence[i][0], tagged_sentence[i][1], tagged_sentence[i - 1][0],
                                              tagged_sentence[i - 1][1])

        return sentence_vector

    def tag(self, test_sent):
        ''' Implement the Viterbi decoding algorithm here.
        '''

        trellis = defaultdict(lambda: defaultdict(int))
        backpointer = {}

        # initialization step
        for tag in self.tags:
            backpointer[tag] = {}
            initial_vector = self.featurize(test_sent[0], tag, '$START', '<S>')
            trellis[tag][0] = self.weights.dot(initial_vector)
            backpointer[tag][0] = '<S>'

        # recursive step
        for t in range(1, len(test_sent)):
            best_tag = 'NN'
            for tag in self.tags:
                max_score = 0

                for tag_prime in self.tags:
                    tmp = self.featurize(test_sent[t], tag, test_sent[t-1], tag_prime)

                    # compute score for each tag using feature representations
                    current_score = trellis[tag_prime][t-1] + self.weights.dot(tmp)
                    if current_score > max_score:
                        max_score = current_score
                        best_tag = tag_prime

                trellis[tag][t] = max_score
                backpointer[tag][t] = best_tag

        # # termination steps
        # max_score = 0
        # best_tag = 'NNS'
        #
        # # get best score of transition from each state to end state
        # for tag in self.tags:
        #     final_vector = self.featurize('$END', '</S>', test_sent[-1], tag)
        #     current_score = trellis[tag][len(test_sent)-1] + self.weights.dot(final_vector)
        #
        #     if current_score > max_score:
        #         max_score = current_score
        #         best_tag = tag
        best_final_score = 0
        best_final_tag = 'NN'
        for tag in self.tags:
            current_final_score = trellis[tag][len(test_sent)-1]
            if current_final_score > best_final_score:
                best_final_score = current_final_score
                best_final_tag = tag

        backpointer['</S>'] = {}
        backpointer['</S>'][len(test_sent)-1] = best_final_tag

        # traverse backpointer from end state to start state to get predicted tag sequence
        current_tag = best_final_tag
        t = len(test_sent) - 1
        path = [[test_sent[t], current_tag]]

        if len(test_sent) == 1:
            return path

        while current_tag != '<S>':
            t -= 1
            path.insert(0, [test_sent[t], current_tag])
            current_tag = backpointer[current_tag][t]

        return path

    def train(self, train_data, dev_data):
        ''' Implement the Perceptron training algorithm here.
        '''

        # results_file = open('25000train_800dev_online.txt', 'w')
        # results_file.write('25000 train 1000 dev online\n')

        for i in range(5):
            print('--------------------------------')
            print('online_iteration ', i)
            train_sentence_count = 0
            # online_train = random.sample(train_data, 25000)
            # online_dev = random.sample(dev_data, 800)
            online_train = train_data[:1000]
            online_dev = dev_data[:500]

            for sent in online_train:
                predicted = self.tag([tup[0] for tup in sent])

                # featurize gold and predicted to get representations for full sequence
                predicted_feats = self.get_sentence_features(predicted)
                gold_feats = self.get_sentence_features(sent)

                # adjust weights according to difference between correct and predicted sequence
                if predicted_feats != gold_feats:
                    self.weights += gold_feats - predicted_feats
                else:
                    print('correct prediction')

            tagged_dev = []
            dev_count = 0
            print('tagging dev set')
            for dev_sent in online_dev:
                dev_tagged = self.tag([tup[0] for tup in dev_sent])
                tagged_dev.append(dev_tagged)

            print()
            acc = self.compute_accuracy(online_dev, tagged_dev)
            print(acc)
            # results_file.write(str(train_sentence_count) + '\t' + str(acc) + '\n')
