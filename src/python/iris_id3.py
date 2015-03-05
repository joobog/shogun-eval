#!/usr/bin/python

from  modshogun import *

file=open("/home/betke/git/shogun-data/uci/iris/iris.data")

category = {'Iris-setosa' : 1.0, 'Iris-versicolor' : 2.0, 'Iris-virginica' : 3.0}

features = []
labels = []

for line in file:
    words = line.rstrip().split(',')
    words[0] = float(words[0])
    words[1] = float(words[1])
    words[2] = float(words[2])
    words[3] = float(words[3])
    words[4] = category[words[4]]
    features.append(words[0:4])
    labels.append(words[4])

file.close()



from numpy import random, delete 
from numpy import array, concatenate


features = array(features)
labels = array(labels)

num_test_vectors = 10;

test_indices = random.randint(features.shape[0], size = num_test_vectors)
test_features = features[test_indices]
test_labels = labels[test_indices]

features = delete(features, test_indices, 0)
labels = delete(labels, test_indices, 0)

test_feats = RealFeatures(test_features.T)
test_labels = MulticlassLabels(test_labels)


def ID3_routine(features, labels):
    print('ID3')
    train_feats = RealFeatures(features.T)
    train_labels = MulticlassLabels(labels)
    id3 = ID3ClassifierTree()
    id3.set_labels(train_labels)
    id3.train(train_feats)
    #id3.print_serializable()
    #f=open("./id3.machine", "w")
    #id3.save_serializable(f)
    #f.close()
    output = id3.apply_multiclass(test_feats)
    print output.get_labels()
    print test_labels.get_labels()

ID3_routine(features, labels)

def ID3CrossValidation(features, labels):
    accuracy = MulticlassAccuracy()

    train_feats = RealFeatures(features.T)
    train_labels = MulticlassLabels(labels)

    id3 = ID3ClassifierTree()
    id3.set_labels(train_labels)
    id3.train(train_feats)

    split = CrossValidationSplitting(train_labels, 10)
    cross_val = CrossValidation(id3, train_feats, train_labels, split, accuracy, False)
    cross_val.set_num_runs(10)
    result = cross_val.evaluate()

    print('Mean Accuracy : ' + str(CrossValidationResult.obtain_from_generic(result).mean))

#ID3CrossValidation(features, labels)

from numpy import array, dtype, int32, float32

def CHAID_routine(dependent_var_type, feature_types, num_bins, features, labels):
    print('CHAID')
    train_feats = RealFeatures(features.T)
    train_labels = MulticlassLabels(labels)
    tree = CHAIDTree(dependent_var_type, feature_types, num_bins)
    tree.set_labels(train_labels)
    tree.train(train_feats)
    output = tree.apply_multiclass(test_feats)
    print output.get_labels()
    print test_labels.get_labels()

# nominal : 0
# ordinal : 1
# continous : 2
ft = array([2, 2, 2, 2], dtype=int32)
chaid = CHAID_routine(0, ft, 3, features, labels)

def CART_routine(features, labels):
    print('CART')
    ft = array([False, False, False, False], dtype=bool)
    train_feats = RealFeatures(features.T)
    train_labels = MulticlassLabels(labels)
    tree = CARTree(ft, PT_REGRESSION, 10, True)
    tree.set_labels(train_labels)
    tree.train(train_feats)
    output = tree.apply_multiclass(test_feats)
    print output.get_labels()
    print test_labels.get_labels()

chaid = CART_routine(features, labels)

#accuracy = MulticlassAccuracy()
#print 'Accuracy : ' + str(accuracy.evaluate(output, test_labels))

#error_rate = []

#num_repetitions = 3

#for i in range(10, 150, 10):
    #indices = random.randint(features.shape[0], size = i)
    #print(indices)
    #train_features = features[indices]
    #train_labels = labels[indices]

    #average_error = 0
    #for i in xrange(num_repetitions):
        #output = ID3_routine(train_features, train_labels)
        #average_error = average_error + (1 - accuracy.evaluate(output, test_labels))

    #error_rate.append(average_error/num_repetitions)

#import matplotlib.pyplot as pyplot

#from scipy.interpolate import interp1d
#from numpy import linspace, arange

#fig, axix = pyplot.subplots(1, 1)
#x = arange(10, 150, 10)
#f = interp1d(x, error_rate)

#xnew = linspace(10, 140, 10)
#pyplot.plot(x, error_rate, 'o', xnew, f(xnew), '-')
#pyplot.xlim([10, 150])
#pyplot.xlabel("training data size")
#pyplot.ylabel("classification error")
#pyplot.title("decision tree performance")
#pyplot.show()
#pyplot.savefig("errors")
