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

    def tag(self, test_sent):
        ''' Implement the Viterbi decoding algorithm here.
        '''

        trellis = defaultdict(lambda: defaultdict(int))
        backpointer = {}

        # initialization step
        for tag in self.tags:
            backpointer[tag] = {}
            initial_vector = self.featurize(test_sent[0], tag, '$START', '<S>')
            trellis[tag][0] = initial_vector.dot(self.weights)
            backpointer[tag][0] = '<S>'

        # recursive step
        for t in range(1, len(test_sent)-1):
            best_tag = 'NN'
            for tag in self.tags:
                max_score = 0

                for tag_prime in self.tags:
                    tmp = self.featurize(test_sent[t], tag, test_sent[t-1], tag_prime)

                    # compute score for each tag using feature representations
                    current_score = trellis[tag_prime][t-1] + tmp.dot(self.weights)
                    if current_score > max_score:
                        max_score = current_score
                        best_tag = tag_prime

                trellis[tag][t] = max_score
                backpointer[tag][t] = best_tag

        # termination steps
        max_score = 0
        best_tag = 'NNS'

        # get best score of transition from each state to end state
        for tag in self.tags:
            final_vector = self.featurize('$END', '</S>', test_sent[-1], tag)
            current_score = trellis[tag][len(test_sent)-1] + final_vector.dot(self.weights)

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

        results_file = open('10000train_1000dev_averaged.txt', 'w')
        # plain_dev = [[tup[0] for tup in sent] for sent in dev_data]

        for i in range(5):
            print('--------------------------------')
            print('minibatch_iteration ', i)
            x = 0
            minibatch = []
            # online_train = []
            for k in range(10000):
                minibatch.append(random.choice(train_data))
                #online_train.append(random.choice(train_data))
            mini_dev = []
            #online_dev = []
            for m in range(1000):
                mini_dev.append(random.choice(dev_data))
                #online_dev.append(random.choice(dev_data))

            plain_mini_dev = [[tup[0] for tup in sent] for sent in mini_dev]

            minibatch_update = Vector({})
            for sent in minibatch:
            #for sent in online_train:
                plain_sent = [tup[0] for tup in sent]
                predicted = self.tag(plain_sent)

                # featurize gold and predicted to get representations for full sequence
                predicted_feats = self.featurize(predicted[0][0], predicted[0][1], '$START', '<S>')
                gold_feats = self.featurize(sent[0][0], sent[0][1], '$START', '<S>')
                for j in range(1, len(predicted)):
                    predicted_feats += self.featurize(predicted[j][0], predicted[j][1], predicted[j-1][0], predicted[j-1][1])
                    gold_feats += self.featurize(sent[j][0], sent[j][1], sent[j-1][0], sent[j-1][1])

                # adjust weights according to difference between correct and predicted sequence
                if predicted_feats != gold_feats:
                    #self.weights += gold_feats - predicted_feats
                    minibatch_update += gold_feats - predicted_feats
                else:
                    print('correct prediction')

                if x % 100 == 0:
                    print('sentence', x)
                    print('p:', predicted)
                    print('g:', sent)
                    print('******')

                x += 1

            #self.weights += minibatch_update.element_wise_divide(len(minibatch))
            self.weights.__rmul__(minibatch_update)

            tagged_dev = []
            dev_count = 0
            for dev_sent in plain_mini_dev:
            #for dev_sent in online_dev:
                dev_tagged = self.tag(dev_sent)
                tagged_dev.append(dev_tagged)

                if dev_count % 50 == 0:
                    print('~~tagging dev. Mini iteration ', i)
                    print('~~len(plain_mini_dev):{}, len(tagged_dev):{}'.format(len(plain_mini_dev), len(tagged_dev)))
                    print('~~dev sentence', dev_count)
                    print(dev_tagged)
                    print('~~########################')
                dev_count += 1

            print()
            acc = self.compute_accuracy(mini_dev, tagged_dev)
            print(acc)
            results_file.write(str(x) + '\t' + str(acc) + '\n')
