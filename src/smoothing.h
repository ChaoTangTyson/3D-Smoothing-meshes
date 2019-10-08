#ifndef MYTOOLS2
#define MYTOOLS2

#define NOMINMAX

#include <vector>
#include <iostream>
#include <Eigen/Dense>
#include <igl/vertex_triangle_adjacency.h>
#include <random>
#include <Eigen/LU>
#include <igl/boundary_loop.h>
#include <igl/boundary_facets.h>
#include <igl/setdiff.h> 
#include <igl/slice.h>
#include <igl/slice_into.h>
#include <igl/cotmatrix.h>
#include <igl/massmatrix.h>
#include <igl/invert_diag.h>

///////////////////////////////////
///////The smoothing part//////////
///////////////////////////////////
Eigen::MatrixXd Ex_smoothing(
	Eigen::MatrixXd const & V,
	Eigen::MatrixXi const & F,
	double const & lambda, int const & iter);

Eigen::MatrixXd Im_smoothing(
	Eigen::MatrixXd const & V,
	Eigen::MatrixXi const & F,
	double const & lambda, int const & iter);
Eigen::MatrixXd Noise(
	Eigen::MatrixXd const & V,
	float noiseLevel);
#endif


