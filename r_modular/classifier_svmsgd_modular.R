# In this example a two-class linear support vector machine classifier is trained
# on a toy data set and the trained classifier is used to predict labels of test
# examples. As training algorithm the Stochastic Gradient Descent (SGD) solver is
# used with the SVM regularization parameter C=0.9. The number of iterations, i.e.
# passes though all training examples, is set to num_iter=5 .
# 
# For more details on the SGD solver see
#  L. Bottou, O. Bousquet. The tradeoff of large scale learning. In NIPS 20. MIT
#  Press. 2008.

library(shogun)

fm_train_real <- t(as.matrix(read.table('../data/fm_train_real.dat')))
fm_test_real <- t(as.matrix(read.table('../data/fm_test_real.dat')))
label_train_twoclass <- as.double(read.table('../data/label_train_twoclass.dat')$V1)

# sgd
print('SVMSGD')

realfeat <- RealFeatures()
dummy <- realfeat$set_feature_matrix(fm_train_real)
feats_train <- SparseRealFeatures()
dump <- feats_train$obtain_from_simple(realfeat)
realfeat <- RealFeatures()
dummy <- realfeat$set_feature_matrix(fm_test_real)
feats_test <- SparseRealFeatures()
dump <- feats_test$obtain_from_simple(realfeat)

C <- 2.3
num_threads <- as.integer(1)
labels <- BinaryLabels()
labels$set_labels(label_train_twoclass)

svm <- SVMSGD(C, feats_train, labels)
#dump <- svm$io$set_loglevel(0)
#dump <- svm$set_epochs(num_iter)
dump <- svm$train()

dump <- svm$set_features(feats_test)
lab <- svm$apply()
out <- lab$get_labels()
