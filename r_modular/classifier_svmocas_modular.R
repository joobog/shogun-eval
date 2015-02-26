# In this example a two-class linear support vector machine classifier is trained
# on a toy data set and the trained classifier is used to predict labels of test
# examples. As training algorithm the OCAS solver is used with the SVM
# regularization parameter C=0.9 and the bias term in the classification rule
# switched off and the precision parameter epsilon=1e-5 (duality gap).
# 
# For more details on the OCAS solver see
#  V. Franc, S. Sonnenburg. Optimized Cutting Plane Algorithm for Large-Scale Risk
#  Minimization.The Journal of Machine Learning Research, vol. 10,
#  pp. 2157--2192. October 2009.
# 

library(shogun)

fm_train_real <- t(as.matrix(read.table('../data/fm_train_real.dat')))
fm_test_real <- t(as.matrix(read.table('../data/fm_test_real.dat')))
label_train_twoclass <- as.double(read.table('../data/label_train_twoclass.dat')$V1)

# svm ocas
print('SVMOcas')

realfeat <- RealFeatures()
dump <- realfeat$set_feature_matrix(fm_train_real)

feats_train <- SparseRealFeatures()
dump <- feats_train$obtain_from_simple(realfeat)
realfeat <- RealFeatures()
dump <- realfeat$set_feature_matrix(fm_test_real)
dump <- feats_test <- SparseRealFeatures()
feats_test$obtain_from_simple(realfeat)

C <- 1.42
epsilon <- 1e-5
num_threads <- as.integer(1)
labels <- BinaryLabels()
labels$set_labels(label_train_twoclass)

svm <- SVMOcas(C, feats_train, labels)
dump <- svm$set_epsilon(epsilon)
dump <- svm$parallel$set_num_threads(num_threads)
dump <- svm$set_bias_enabled(FALSE)
dump <- svm$train()

dump <- svm$set_features(feats_test)
lab <- svm$apply()
out <- lab$get_labels()
