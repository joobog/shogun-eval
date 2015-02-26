# The Weighted Degree String kernel.
# 
# The WD kernel of order d compares two sequences X and
# Y of length L by summing all contributions of k-mer matches of
# lengths k in 1...d , weighted by coefficients beta_k. It
# is defined as
# 
#     k(X, Y)=\sum_{k=1}^d\beta_k\sum_{l=1}^{L-k+1}I(u_{k,l}(X)=u_{k,l}(Y)).
# 
# Here, $u_{k,l}(X)$ is the string of length k starting at position
# l of the sequence X and I(.) is the indicator function
# which evaluates to 1 when its argument is true and to 0
# otherwise.
# 

library(shogun)

fm_train_dna <- as.matrix(read.table('../data/fm_train_dna.dat'))
fm_test_dna <- as.matrix(read.table('../data/fm_test_dna.dat'))

# weighted_degree_string
print('WeightedDegreeString')

feats_train <- StringCharFeatures("DNA")
dump <- feats_train$set_features(fm_train_dna)
feats_test <- StringCharFeatures("DNA")
dump <- feats_test$set_features(fm_test_dna)
degree <- as.integer(20)

kernel <- WeightedDegreeStringKernel(feats_train, feats_train, degree)

#weights <- arange(1,degree+1,dtype <- double)[::-1]/ \
#	sum(arange(1,degree+1,dtype <- double))
#kernel$set_wd_weights(weights)

km_train <- kernel$get_kernel_matrix()
dump <- kernel$init(feats_train, feats_test)
km_test <- kernel$get_kernel_matrix()
