# The constant kernel gives a trivial kernel matrix with all entries set to the same value
# defined by the argument 'c'.
# 

library(shogun)

fm_train_real <- as.matrix(read.table('../data/fm_train_real.dat'))
fm_test_real <- as.matrix(read.table('../data/fm_test_real.dat'))

# const
print('Const')

feats_train <- RealFeatures()
dummy <- feats_train$set_feature_matrix(fm_train_real)
feats_test <- RealFeatures()
dummy <- feats_test$set_feature_matrix(fm_test_real)
c <- 23.

kernel <- ConstKernel(feats_train, feats_train, c)

km_train <- kernel$get_kernel_matrix()
dump <- kernel$init(feats_train, feats_test)
km_test <- kernel$get_kernel_matrix()
