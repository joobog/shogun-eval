# In this example a two-class support vector machine classifier is trained on a
# DNA splice-site detection data set and the trained classifier is used to predict
# labels on test set. As training algorithm SVM^light is used with SVM
# regularization parameter C=1.2 and the Weighted Degree kernel of degree 20 and
# the precision parameter epsilon=1e-5.
# 
# For more details on the SVM^light see
#  T. Joachims. Making large-scale SVM learning practical. In Advances in Kernel
#  Methods -- Support Vector Learning, pages 169-184. MIT Press, Cambridge, MA USA, 1999.
# 
# For more details on the Weighted Degree kernel see
#  G. Raetsch, S.Sonnenburg, and B. Schoelkopf. RASE: recognition of alternatively
#  spliced exons in C. elegans. Bioinformatics, 21:369-377, June 2005.

library(shogun)

fm_train_dna <- as.matrix(read.table('../data/fm_train_dna.dat'))
fm_test_dna <- as.matrix(read.table('../data/fm_test_dna.dat'))
label_train_dna <- as.double(read.table('../data/label_train_dna.dat')$V1)

# svm light
dosvmlight <- function()
{
	print('SVMLight')

	feats_train <- StringCharFeatures("DNA")
	dump <- feats_train$set_feature_matrix(feats_train, fm_train_dna)
	feats_test <- StringCharFeatures("DNA")
	dump <- feats_test$set_feature_matrix(feats_test, fm_test_dna)
	degree <- as.integer(20)

	kernel <- WeightedDegreeStringKernel(feats_train, feats_train, degree)

	C <- 1.017
	epsilon <- 1e-5
	num_threads <- as.integer(3)
	labels <- Labels(label_train_dna)

	svm <- SVMLight(C, kernel, labels)
	dump <- svm$set_epsilon(svm, epsilon)
	dump <- svm$parallel$set_num_threads(svm$parallel, num_threads)
	dump <- svm$train(svm)

	dump <- kernel$init(kernel, feats_train, feats_test)
	lab <- svm$apply(svm)
	out <- lab$get_labels(lab)
}
try(dosvmlight())
