# In this example a two-class linear support vector machine classifier is trained
# on a toy data set and the trained classifier is then used to predict labels of
# test examples. As training algorithm the LIBLINEAR solver is used with the SVM
# regularization parameter C=0.9 and the bias in the classification rule switched
# on and the precision parameters epsilon=1e-5.
# 
# For more details on LIBLINEAR see
#     http://www.csie.ntu.edu.tw/~cjlin/liblinear/

library(shogun)

fm_train_real <- t(as.matrix(read.table('../data/fm_train_real.dat')))
fm_test_real <- t(as.matrix(read.table('../data/fm_test_real.dat')))
label_train_twoclass <- as.double(read.table('../data/label_train_twoclass.dat')$V1)

# liblinear
print('LibLinear')

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

svm <- LibLinear(C, feats_train, labels)
dump <- svm$set_epsilon(epsilon)
dump <- svm$parallel$set_num_threads(num_threads)
dump <- svm$set_bias_enabled(TRUE)
dump <- svm$train()

dump <- svm$set_features(feats_test)
lab <- svm$apply()
out <- lab$get_labels()
