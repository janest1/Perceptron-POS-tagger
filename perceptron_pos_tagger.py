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
        for t in range(1, len(test_sent)-1):
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

        # termination steps
        max_score = 0
        best_tag = 'NN'

        # get best score of transition from each state to end state
        for tag in self.tags:
            final_vector = self.featurize('$END', '</S>', test_sent[-1], tag)
            current_score = trellis[tag][len(test_sent)-1] + self.weights.dot(final_vector)

            if current_score > max_score:
                max_score = current_score
                best_tag = tag

        backpointer['</S>'] = {}
        backpointer['</S>'][len(test_sent)-1] = best_tag

        # traverse backpointer from end state to start state to get predicted tag sequence
        current_tag = best_tag
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

        results_file = open('1000train_500dev_averaged_smaller_update.txt', 'w')
        results_file.write('1000 train 500 dev averaged (update weights after 100 every iteration)\n')

        for i in range(8):
            print('--------------------------------')
            print('minibatch_iteration ', i)
            #train_sentence_count = 0
            minibatch = random.sample(train_data, 10)
            mini_dev = random.sample(dev_data, 3)
            minibatch_update = Vector({})

            for sent in minibatch:
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
                    minibatch_update += gold_feats - predicted_feats
                else:
                    print('correct prediction')

                # if train_sentence_count % 500 == 0:
                #     print('mini training iteration', i)
                #     print('training sentence', train_sentence_count)
                #     print('p:', predicted)
                #     print('g:', sent)
                #     print('******')

                #train_sentence_count += 1

            print('updating weights....')
            #self.weights += (1/len(minibatch)) * minibatch_update
            self.weights += minibatch_update
            self.weights = (1/len(minibatch)) * self.weights

            tagged_dev = []
            #dev_count = 0
            print('tagging dev set....')
            for dev_sent in mini_dev:
                plain_dev_sent = [tup[0] for tup in dev_sent]
                dev_tagged = self.tag(plain_dev_sent)
                tagged_dev.append(dev_tagged)

                # if dev_count % 200 == 0:
                #     print('~~tagging dev after mini iteration ', i)
                #     print('~~dev sentence', dev_count)
                #     print('~~~~~~~~~~~~~~~~~~~~~~~~~~')
                # dev_count += 1

            print()
            acc = self.compute_accuracy(mini_dev, tagged_dev)
            print(acc)
            results_file.write(str(i) + '\t' + str(acc) + '\n')
