# In this example a kernel matrix is computed for a given real-valued data set.
# The kernel used is the Chi2 kernel which operates on real-valued vectors. It
# computes the chi-squared distance between sets of histograms. It is a very
# useful distance in image recognition (used to detect objects). The preprocessor
# LogPlusOne adds one to a dense real-valued vector and takes the logarithm of
# each component of it. It is most useful in situations where the inputs are
# counts: When one compares differences of small counts any difference may matter
# a lot, while small differences in large counts don't. This is what this log
# transformation controls for.

library(shogun)

fm_train_real <- t(as.matrix(read.table('../data/fm_train_real.dat')))
fm_test_real <- t(as.matrix(read.table('../data/fm_test_real.dat')))

#LogPlusOne
print('LogPlusOne')

feats_train <- RealFeatures()
dump <- feats_train$set_feature_matrix(fm_train_real)
feats_test <- RealFeatures()
dump <- feats_test$set_feature_matrix(fm_test_real)

preproc <- LogPlusOne()
dump <- preproc$init(feats_train)
dump <- feats_train$add_preproc(preproc)
dump <- feats_train$apply_preproc()
dump <- feats_test$add_preproc(preproc)
dump <- feats_test$apply_preproc()

width <- 1.4
size_cache <- as.integer(10)

kernel <- Chi2Kernel(feats_train, feats_train, width, size_cache)

km_train <- kernel$get_kernel_matrix()
dump <- kernel$init(feats_train, feats_test)
km_test <- kernel$get_kernel_matrix()
