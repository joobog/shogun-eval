#!/home/betke/install/R/bin/Rscript
# This example shows usage of a k-nearest neighbor (KNN) classification rule on
# a toy data set. The number of the nearest neighbors is set to k=3 and the distances
# are measured by the Euclidean metric. Finally, the KNN rule is applied to predict
# labels of test examples.

library(shogun)

#fm_train_real <- t(as.matrix(read.table('~/git/shogun-data/toy/fm_train_real.dat')))
#fm_test_real <- t(as.matrix(read.table('~/git/shogun-data/toy/fm_test_real.dat')))
#label_train_multiclass <- as.double(read.table('~/git/shogun-data/toy/label_train_multiclass.dat')$V1)

## knn
#print('KNN')

#feats_train <- RealFeatures()
#dump <- feats_train$set_feature_matrix(fm_train_real)
#feats_test <- RealFeatures()
#dump <- feats_test$set_feature_matrix(fm_test_real)
#distance <- EuclideanDistance()

#k <- as.integer(3)
#num_threads <- as.integer(1)
#labels <- MulticlassLabels()
#dump <- labels$set_labels(label_train_multiclass)

#knn <- KNN(k, distance, labels)
#dump <- knn$parallel$set_num_threads(num_threads)
#dump <- knn$train(feats_train)
#lab <- knn$apply(feats_test)
#out <- lab$get_labels()
