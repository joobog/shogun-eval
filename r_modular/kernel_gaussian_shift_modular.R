# An experimental kernel inspired by the WeightedDegreePositionStringKernel and the Gaussian kernel.
# The idea is to shift the dimensions of the input vectors against eachother. 'shift_step' is the step
# size of the shifts and  max_shift is the maximal shift.

library(shogun)

fm_train_real <- t(as.matrix(read.table('../data/fm_train_real.dat')))
fm_test_real <- t(as.matrix(read.table('../data/fm_test_real.dat')))

# gaussian_shift
print('GaussianShift')

feats_train <- RealFeatures()
dummy <- feats_train$set_feature_matrix(fm_train_real)
feats_test <- RealFeatures()
dummy <- feats_test$set_feature_matrix(fm_test_real)
width <- 1.8
max_shift <- as.integer(2)
shift_step <- as.integer(1)

kernel <- GaussianShiftKernel(
	feats_train, feats_train, width, max_shift, shift_step)

km_train <- kernel$get_kernel_matrix()
dump <- kernel$init(feats_train, feats_test)
km_test <- kernel$get_kernel_matrix()
