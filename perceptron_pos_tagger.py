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
        prev_emission = 'w-1_{} t-1_{}'.format(prev_word, prev_tag)
        vector.v[prev_emission] = 1

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

        best_final_score = 0
        best_final_tag = 'NN'
        for tag in self.tags:
            current_final_score = trellis[tag][len(test_sent) - 1]
            if current_final_score > best_final_score:
                best_final_score = current_final_score
                best_final_tag = tag

        # traverse backpointer from end state to start state to get predicted tag sequence
        current_tag = best_final_tag
        t = len(test_sent) - 1
        path = [[test_sent[t], current_tag]]

        if len(test_sent) == 1:
            return path

        while t > 0:
            current_tag = backpointer[current_tag][t]
            t -= 1
            path.insert(0, [test_sent[t], current_tag])

        return path

    def train(self, train_data, dev_data):
        ''' Implement the Perceptron training algorithm here.
        '''

        # results_file = open('25000train_1000dev_averaged.txt', 'w')
        # results_file.write('2500 train 1000 dev averaged\n')

        for i in range(5):
            print('--------------------------------')
            print('minibatch_iteration ', i)
            averaged_train = random.sample(train_data, 25000)

            for sent in averaged_train:
                if i == 0:
                    # first iteration has all zero weights, so a default tag of NN is chosen for each
                    # step in the sequence. Get first round of averaged perceptron weight updates using
                    # this assumption instead of running Viterbi on the first iteration
                    predicted = [[tup[0], 'NN'] for tup in sent]

                else:
                    predicted = self.tag([tup[0] for tup in sent])

                # featurize gold and predicted to get representations for full sequence
                predicted_feats = self.get_sentence_features(predicted)
                gold_feats = self.get_sentence_features(sent)

                # adjust weights according to difference between correct and predicted sequence
                if predicted_feats != gold_feats:
                    self.weights += (gold_feats - predicted_feats)
                else:
                    print('correct prediction')

            self.weights = (1/len(averaged_train)) * self.weights

            tagged_dev = []
            print('tagging dev set....')
            for dev_sent in dev_data:
                plain_dev_sent = [tup[0] for tup in dev_sent]
                dev_tagged = self.tag(plain_dev_sent)
                tagged_dev.append(dev_tagged)

            print()
            acc = self.compute_accuracy(dev_data, tagged_dev)
            print(acc)
            #results_file.write(str(i) + '\t' + str(acc) + '\n')
