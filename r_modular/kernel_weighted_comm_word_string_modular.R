# The WeightedCommWordString kernel may be used to compute the weighted
# spectrum kernel (i.e. a spectrum kernel for 1 to K-mers, where each k-mer
# length is weighted by some coefficient \f$\beta_k\f$) from strings that have
# been mapped into unsigned 16bit integers.
# 
# These 16bit integers correspond to k-mers. To applicable in this kernel they
# need to be sorted (e.g. via the SortWordString pre-processor).
# 
# It basically uses the algorithm in the unix "comm" command (hence the name)
# to compute:
# 
# k({\bf x},({\bf x'})= \sum_{k=1}^K\beta_k\Phi_k({\bf x})\cdot \Phi_k({\bf x'})
# 
# where \f$\Phi_k\f$ maps a sequence \f${\bf x}\f$ that consists of letters in
# \f$\Sigma\f$ to a feature vector of size \f$|\Sigma|^k\f$. In this feature
# vector each entry denotes how often the k-mer appears in that \f${\bf x}\f$.
# 
# Note that this representation is especially tuned to small alphabets
# (like the 2-bit alphabet DNA), for which it enables spectrum kernels
# of order 8.
# 
# For this kernel the linadd speedups are quite efficiently implemented using
# direct maps.
# 

library(shogun)

fm_train_dna <- as.matrix(read.table('../data/fm_train_dna.dat'))
fm_test_dna <- as.matrix(read.table('../data/fm_test_dna.dat'))

# weighted_comm_word_string
print('WeightedCommWordString')

order <- as.integer(3)
start <- as.integer(order-1)
gap <- as.integer(0)
reverse <- TRUE

charfeat <- StringCharFeatures("DNA")
dump <- charfeat$set_features(fm_train_dna)
feats_train <- StringWordFeatures(charfeat$get_alphabet())
dump <- feats_train$obtain_from_char(charfeat, start, order, gap, reverse)
preproc <- SortWordString()
dump <- preproc$init(feats_train)
dump <- feats_train$add_preproc(preproc)
dump <- feats_train$apply_preproc()

charfeat <- StringCharFeatures("DNA")
dump <- charfeat$set_features(fm_test_dna)
feats_test <- StringWordFeatures(charfeat$get_alphabet())
dump <- feats_test$obtain_from_char(charfeat, start, order, gap, reverse)
dump <- feats_test$add_preproc(preproc)
dump <- feats_test$apply_preproc()

use_sign <- FALSE

kernel <- WeightedCommWordStringKernel(feats_train, feats_train, use_sign)
km_train <- kernel$get_kernel_matrix()

kernel <- WeightedCommWordStringKernel(feats_train, feats_test, use_sign)
km_test <- kernel$get_kernel_matrix()
