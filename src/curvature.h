#ifndef MYTOOLS
#define MYTOOLS

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
///////The curvature part//////////
///////////////////////////////////
Eigen::SparseMatrix<double> Uniform_Laplacian(
	Eigen::MatrixXd const & V,
	Eigen::MatrixXi const & F);

Eigen::SparseMatrix<double> Barycentric_Area(
	Eigen::MatrixXd const & V,
	Eigen::MatrixXi const & F);

Eigen::SparseMatrix<double> GetCot(
	Eigen::MatrixXd const & V,
	Eigen::MatrixXi const & F);
Eigen::SparseMatrix<double> GetCot2(
	Eigen::MatrixXd const & V,
	Eigen::MatrixXi const & F);

Eigen::SparseMatrix<double> Laplace_Beltrami(
	Eigen::MatrixXd const & V,
	Eigen::MatrixXi const & F);

Eigen::VectorXd compute_meanH(
	Eigen::MatrixXd const & V,
	Eigen::MatrixXi const & F,
	Eigen::SparseMatrix<double> L);

Eigen::VectorXd compute_K(
	Eigen::MatrixXd const & V,
	Eigen::MatrixXi const & F);

Eigen::MatrixXd Reconstruction(
	Eigen::MatrixXd const & V,
	Eigen::MatrixXi const & F,
	int const & k
);

void calculate_vertex_normal(
	Eigen::MatrixXd const & V, 
	Eigen::MatrixXi const & F, 
	Eigen::MatrixXd const & FN,
	Eigen::MatrixXd & out_VN);

#endif


