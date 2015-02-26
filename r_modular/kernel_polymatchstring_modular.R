# This is an example for the initialization of the PolyMatchString kernel on string data.
# The PolyMatchString kernel sums over the matches of two stings of the same length and
# takes the sum to the power of 'degree'. The strings consist of the characters 'ACGT' corresponding
# to the DNA-alphabet. Each column of the matrices of type char corresponds to
# one training/test example.

library(shogun)

fm_train_dna <- as.matrix(read.table('../data/fm_train_dna.dat'))
fm_test_dna <- as.matrix(read.table('../data/fm_test_dna.dat'))

# poly_match_string
print('PolyMatchString')

feats_train <- StringCharFeatures("DNA")
dump <- feats_train$set_features(fm_train_dna)
feats_test <- StringCharFeatures("DNA")
dump <- feats_test$set_features(fm_test_dna)
degree <- as.integer(3)
inhomogene <- FALSE

kernel <- PolyMatchStringKernel(feats_train, feats_train, degree, inhomogene)

km_train <- kernel$get_kernel_matrix()
dump <- kernel$init(feats_train, feats_test)
km_test <- kernel$get_kernel_matrix()
