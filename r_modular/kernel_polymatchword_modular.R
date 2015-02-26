# The PolyMatchWordString kernel is defined on strings of equal length.
# The kernel sums over the matches of two stings of the same length and
# takes the sum to the power of 'degree'. The strings in this example
# consist of the characters 'ACGT' corresponding to the DNA-alphabet. Each
# column of the matrices of type char corresponds to one training/test example.

library(shogun)

fm_train_word <- as.matrix(read.table('../data/fm_train_word.dat'))
fm_test_word <- as.matrix(read.table('../data/fm_test_word.dat'))

## poly_match_word
#print('PolyMatchWord')
#
#feats_train <- WordFeatures(traindata_word)
#feats_test <- WordFeatures(testdata_word)
#degree <- 2
#inhomogene <- TRUE
#
#kernel <- PolyMatchWordKernel(feats_train, feats_train, degree, inhomogene)
#
#km_train <- kernel$get_kernel_matrix()
#kernel$init(kernel, feats_train, feats_test)
#km_test <- kernel$get_kernel_matrix()
