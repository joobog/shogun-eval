# This is an example for the initialization of a linear kernel on string data. The
# strings are all of the same length and consist of the characters 'ACGT' corresponding
# to the DNA-alphabet. Each column of the matrices of type char corresponds to
# one training/test example.

library(shogun)

fm_train_dna <- as.matrix(read.table('../data/fm_train_dna.dat'))
fm_test_dna <- as.matrix(read.table('../data/fm_test_dna.dat'))

# linear_string
print('LinearString')

feats_train <- StringCharFeatures("DNA")
dump <- feats_train$set_features(fm_train_dna)
feats_test <- StringCharFeatures("DNA")
dump <- feats_test$set_features(fm_test_dna)

kernel <- LinearStringKernel(feats_train, feats_train)

km_train <- kernel$get_kernel_matrix()
dump <- kernel$init(feats_train, feats_test)
km_test <- kernel$get_kernel_matrix()
