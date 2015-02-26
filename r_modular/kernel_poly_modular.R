# This example initializes the polynomial kernel with real data.
# If variable 'inhomogene' is 'True' +1 is added to the scalar product
# before taking it to the power of 'degree'. If 'use_normalization' is
# set to 'true' then kernel matrix will be normalized by the square roots
# of the diagonal entries.

library(shogun)

fm_train_real <- t(as.matrix(read.table('../data/fm_train_real.dat')))
fm_test_real <- t(as.matrix(read.table('../data/fm_test_real.dat')))

# poly
print('Poly')

feats_train <- RealFeatures()
dummy <- feats_train$set_feature_matrix(fm_train_real)
feats_test <- RealFeatures()
dummy <- feats_test$set_feature_matrix(fm_test_real)
degree <- as.integer(4)
inhomogene <- FALSE

kernel <- PolyKernel(
	feats_train, feats_train, degree, inhomogene)

km_train <- kernel$get_kernel_matrix()
dump <- kernel$init(feats_train, feats_test)
km_test <- kernel$get_kernel_matrix()
