# In this example a hidden markov model with 3 states and 6 transitions is trained
# on a string data set. After calling the constructor of the HMM class specifying
# the number of states and transitions the model is trained. Via the Baum-Welch
# algorithm the optimal transition and emission probabilities are estimated. The
# best path, i.e. the path with highest probability given the model can then be
# calculated using get_best_path_state.

library(shogun)

fm_train_cube <- as.matrix(read.table('../data/fm_train_cube.dat', colClasses=c('character')))

# HMM
print('HMM')

N <- as.integer(3)
M <- as.integer(6)
pseudo <- 1e-1
order <- as.integer(1)
start <- as.integer(order-1)
gap <- as.integer(0)
reverse <- FALSE
num_examples <- as.integer(2)

charfeat <- StringCharFeatures("CUBE")
dump <- charfeat$set_features(fm_train_cube)
feats <- StringWordFeatures(charfeat$get_alphabet())
dump <- feats$obtain_from_char(charfeat, start, order, gap, reverse)
preproc <- SortWordString()
dump <- preproc$init(feats)
dump <- feats$add_preproc(preproc)
dump <- feats$apply_preproc()

hmm <- HMM(feats, N, M, pseudo)
dump <- hmm$train()
dump <- hmm$baum_welch_viterbi_train("BW_NORMAL")

num_examples <- feats$get_num_vectors()
num_param <- hmm$get_num_model_parameters()

derivs <- matrix(0, num_param, num_examples)
for (i in 0:(num_examples-1))
{
	for (j in 0:(num_param-1))
	{
		derivs[j,i] <- hmm$get_log_derivative(j, i)
	}
}

best_path <- 0
best_path_state <- 0

for (i in 0:(num_examples-1))
{
	best_path = best_path + hmm$best_path(i)
	for (j in 0:(N-1))
	{
		best_path_state = best_path_state + hmm$get_best_path_state(i, j)
	}
}

dump <- hmm$get_log_likelihood()
dump <- hmm$get_log_likelihood_sample()
