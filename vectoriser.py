import numpy
from sklearn import svm
import random

# file contains 1000 samples of 28 * 28 bytes each.
BYTES_PER_SAMPLE = 28 * 28


def get_test_training_samples(digit, percentage_training=0.5, directory='./data/'):
    assert 0 <= digit <= 9

    filename = directory + "data{digit}".format(digit=digit)
    samples = []
    with open(filename) as file:
        while True:
            sample = numpy.array(
                [ord(byte) for byte in file.read(BYTES_PER_SAMPLE)]
            )
            if len(sample) < BYTES_PER_SAMPLE:
                break
            samples.append((sample, digit)) # (sample, label)


    num_samples = len(samples)
    training_samples = samples[0:int(num_samples * percentage_training)]
    test_samples = samples[int(num_samples * percentage_training):]

    return training_samples, test_samples

def print_vector(vector):
    greyscale = [
        " ",
        " ",
        ".,-",
        "_ivc=!/|\\~",
        "gjez2]/(YL)t[+T7Vf",
        "mdK4ZGbNDXY5P*Q",
        "W8KMA",
        "#%$"
    ]
    new_vector = vector.reshape((28, 28)) / 32 # sample down

    greyscale_character = lambda x: random.choice(greyscale[x])

    lines = []
    for row in new_vector:
        lines.append(''.join(map(greyscale_character, row)))
    return '\n'.join(lines)

def get_trained_classifier(directory):
    training_set = []
    for digit in range(1, 10): # 1-9
        training_samples, _ = get_test_training_samples(digit, percentage_training=1.0, directory=directory)
        training_set.extend(training_samples)

    training_vectors, training_labels = zip(*training_set) # heh, hack
    classifier = svm.SVC(kernel='poly', decision_function_shape='ovo')
    print("training...")
    classifier.fit(training_vectors, training_labels)
    return classifier

def test_classifier():
    training_set = []
    test_set = []

    for digit in range(10): # 0-9
        training_samples, test_samples = get_test_training_samples(digit)
        training_set.extend(training_samples)
        test_set.extend(test_samples)

    training_vectors, training_labels = zip(*training_set) # heh, hack
    classifier = svm.SVC(kernel='poly', decision_function_shape='ovo')
    print("training...")
    classifier.fit(training_vectors, training_labels)
    test_vectors, test_labels = zip(*test_set)
    print("testing...")
    score = classifier.score(test_vectors, test_labels)
    import ipdb; ipdb.set_trace()
    return score
