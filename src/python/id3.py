#!/usr/bin/python




# training data
train_income=['Low','Medium','Low','High','Low','High','Medium','Medium','High','Low','Medium',
        'Medium','High','Low','Medium']

train_age = ['Old','Young','Old','Young','Old','Young','Young','Old','Old','Old','Young','Old',
        'Old','Old','Young']

train_education = ['University','College','University','University','University','College','College',
        'High School','University','High School','College','High School','University','High School','College']

train_marital = ['Married','Single','Married','Single','Married','Single','Married','Single','Single',
        'Married','Married','Single','Single','Married','Married']

train_usage = ['Low','Medium','Low','High','Low','Medium','Medium','Low','High','Low','Medium','Low',
        'High','Low','Medium']

# print data
print 'Training Data Table : \n'
print 'Income \t\t Age \t\t Education \t\t Marital Status \t Usage'
for i in xrange(len(train_income)):
        print train_income[i]+' \t\t '+train_age[i]+' \t\t '+train_education[i]+' \t\t '+train_marital[i]+' \t\t '+train_usage[i]

from modshogun import ID3ClassifierTree, RealFeatures, MulticlassLabels
from numpy import array, concatenate

# encoding dictionary
income = {'Low' : 1.0, 'Medium' : 2.0, 'High' : 3.0}
age = {'Young' : 1.0, 'Old' : 2.0}
education = {'High School' : 1.0, 'College' : 2.0, 'University' : 3.0}
marital_status = {'Married' : 1.0, 'Single' : 2.0}
usage = {'Low' : 1.0, 'Medium' : 2.0, 'High' : 3.0}


# encode training data
for i in xrange(len(train_income)):
    train_income[i] = income[train_income[i]]
    train_age[i] = age[train_age[i]]
    train_education[i] = education[train_education[i]]
    train_marital[i] = marital_status[train_marital[i]]
    train_usage[i] = usage[train_usage[i]]

# form Shogun feature matrix
train_data = array([train_income, train_age, train_education, train_marital])
print train_data
RealFeatures(train_data);
train_feats = RealFeatures(train_data);

# form Shogun multiclass labels
MulticlassLabels(array(train_usage));
labels = MulticlassLabels(array(train_usage));







# create ID3ClassifierTree object
id3 = ID3ClassifierTree()

# set labels
id3.set_labels(labels)

# learn the tree from training features
is_successful = id3.train(train_feats)

# test data
test_income = ['Medium','Medium','Low','High','High']
test_age = ['Old','Young','Old','Young','Old']
test_education = ['University','College','High School','University','College']
test_marital = ['Married','Single','Married','Single','Married']
test_usage = ['Low','Medium','Low','High','High']

# tabulate test data
print 'Test Data Table : \n'
print 'Income \t\t Age \t\t Education \t\t Marital Status \t Usage'
for i in xrange(len(test_income)):
        print test_income[i]+' \t\t '+test_age[i]+' \t\t '+test_education[i]+' \t\t '+test_marital[i]+' \t\t ?'



# encode test data
#for i in xrange(len(test_income)):
    #test_income[i] = income[test_income[i]]
    #test_age[i] = age[test_age[i]]
    #test_education[i] = education[test_education[i]]
    #test_marital[i] = marital_status[test_marital[i]]

    ## bind to shogun features    
    #test_data = array([test_income, test_age, test_education, test_marital])
    #test_feats = RealFeatures(test_data)

    ## apply decision tree classification
    #test_labels = id3.apply_multiclass(test_feats)



