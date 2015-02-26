# In this example a two-class linear support vector machine classifier (SVM) is
# trained on a toy data set and the trained classifier is used to predict labels
# of test examples. As training algorithm the SVMLIN solver is used with the SVM
# regularization parameter C=0.9 and the bias in the classification rule switched
# on and the precision parameter epsilon=1e-5. The example also shows how to
# retrieve parameters (vector w and bias b)) of the trained linear classifier.
# 
# For more details on the SVMLIN solver see
#  V. Sindhwani, S.S. Keerthi. Newton Methods for Fast Solution of Semi-supervised
#  Linear SVMs. Large Scale Kernel Machines MIT Press (Book Chapter), 2007

library(shogun)

fm_train_real <- t(as.matrix(read.table('../data/fm_train_real.dat')))
fm_test_real <- t(as.matrix(read.table('../data/fm_test_real.dat')))
label_train_twoclass <- as.double(read.table('../data/label_train_twoclass.dat')$V1)

# svm lin
print('SVMLin')

realfeat <- RealFeatures()
dump <- realfeat$set_feature_matrix(fm_train_real)
feats_train <- SparseRealFeatures()
dump <- feats_train$obtain_from_simple(realfeat)
realfeat <- RealFeatures()
dump <- realfeat$set_feature_matrix(fm_test_real)
feats_test <- SparseRealFeatures()
dump <- feats_test$obtain_from_simple(realfeat)

C <- 1.42
epsilon <- 1e-5
num_threads <- as.integer(1)
labels <- BinaryLabels()
labels$set_labels(label_train_twoclass)

svm <- SVMLin(C, feats_train, labels)
dump <- svm$set_epsilon(epsilon)
dump <- svm$parallel$set_num_threads(num_threads)
dump <- svm$set_bias_enabled(TRUE)
dump <- svm$train()

dump <- svm$set_features(feats_test)
dump <- svm$get_bias()
dump <- svm$get_w()
lab <- svm$apply()
out <- lab$get_labels()
