# A user defined custom kernel is assigned in this example, for which only the lower triangle
# may be given (set_triangle_kernel_matrix_from_triangle) or
# a full matrix (set_full_kernel_matrix_from_full), or a full matrix which is then internally stored as a
# triangle (set_triangle_kernel_matrix_from_full). Labels for the examples are given, a svm is trained and
# the svm is used to classify the examples.
# 

library(shogun)

## custom
#print('Custom')
#
#dim <- 7
#data <- rand(dim, dim)
#feats <- RealFeatures(data)
#symdata <- data+data'
#lowertriangle <- array([symdata[(x,y)] for x in xrange(symdata.shape[1])
#	for y in xrange(symdata.shape[0]) if y< <- x])
#
#kernel <- CustomKernel(feats, feats)
#
#kernel$set_triangle_kernel_matrix_from_triangle(lowertriangle)
#km_triangletriangle <- kernel$get_kernel_matrix()
#
#kernel$set_triangle_kernel_matrix_from_full(symdata)
#km_fulltriangle <- kernel$get_kernel_matrix()
#
#kernel$set_full_kernel_matrix_from_full(data)
#km_fullfull <- kernel$get_kernel_matrix()
