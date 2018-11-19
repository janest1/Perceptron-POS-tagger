import sys
from perceptron_pos_tagger import Perceptron_POS_Tagger
from data_structures import Sentence


def get_tags(sentences):
    tags = []

    for sent in sentences:
        for tup in sent:
            if tup[1] not in tags:
                tags.append(tup[1])

    return tags


def read_in_gold_data(filename):
    with open(filename) as f:
        lines = f.readlines()
        lines = [[tup.split('_') for tup in line.split()] for line in lines]
        # sents = [Sentence(line) for line in lines]

    return lines


def read_in_plain_data(filename):
    with open(filename) as f:
        lines = f.readlines()
        lines = [line.split() for line in lines]
        # sents = [Sentence(line) for line in lines]

    return lines


def output_auto_data(auto_data):
    ''' According to the data structure you used for "auto_data",
        write code here to output your auto tagged data into a file,
        using the same format as the provided gold data (i.e. word_pos word_pos ...)
    '''

    with open('dev_tagged.txt', 'w') as outfile:
        for line in auto_data:
            outfile.write(str(line))
            outfile.write('\n')


if __name__ == '__main__':

    # Run python train_test_tagger.py train/ptb_02-21.tagged dev/ptb_22.tagged dev/ptb_22.snt test/ptb_23.snt to train & test your tagger
    # train_file = sys.argv[1]
    # gold_dev_file = sys.argv[2]
    # plain_dev_file = sys.argv[3]
    # test_file = sys.argv[4]

    #remove this
    train_file = 'train/ptb_02-21.tagged'
    gold_dev_file = 'dev/ptb_22.tagged'
    plain_dev_file = 'dev/ptb_22.snt'
    test_file = 'test/ptb_23.snt'

    # Read in data
    train_data = read_in_gold_data(train_file)
    gold_dev_data = read_in_gold_data(gold_dev_file)
    plain_dev_data = read_in_plain_data(plain_dev_file)
    test_data = read_in_plain_data(test_file)

    # Train your tagger
    tags = get_tags(train_data)
    my_tagger = Perceptron_POS_Tagger(tags)
    my_tagger.train(train_data, gold_dev_data)

    # Apply your tagger on dev & test data
    auto_test_data = my_tagger.tag(test_data)

    # Output your auto tagged data
    output_auto_data(auto_test_data)
