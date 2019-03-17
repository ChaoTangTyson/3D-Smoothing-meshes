
#include <igl/adjacency_list.h>
#include <igl/vertex_triangle_adjacency.h>
#include <igl/parula.h>
#include <igl/invert_diag.h>
#include <math.h>
#include <stdio.h>
#include <random>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include<Eigen/SparseCholesky>

#include "curvature.h"
#include "smoothing.h"

using namespace  std;
///////////////////////////////////
///////The smoothing part//////////
///////////////////////////////////

///////////////////////////////////
/////////Explicit Laplacian ///////
///////////////////////////////////
Eigen::MatrixXd Ex_smoothing(
	Eigen::MatrixXd const & V,
	Eigen::MatrixXi const & F,
	double const & lambda,int const & iter)
{
	Eigen::MatrixXd New_V(V.rows(),V.cols());
	Eigen::MatrixXd Old_V = V;
	
	// cout << "Pass the first phase test" << endl;
	// Get the Laplacian operator
	Eigen::SparseMatrix<double> Lapla = Laplace_Beltrami(V, F);

	for (int i = 0; i < iter; i++)
	{
		New_V = Old_V + lambda * Lapla * Old_V;
		Old_V = New_V;
	}
	/*
	cout << "Pass the second phase test" << endl;
	cout << "Old V1  " << V.row(0) << endl;
	cout << "New V1  " << New_V.row(0) << endl;
	cout << "Setp " << (lambda * Lapla * Old_V).rows() << endl;
	*/
	return  New_V;
}

Eigen::MatrixXd Im_smoothing(
	Eigen::MatrixXd const & V,
	Eigen::MatrixXi const & F,
	double const & lambda, int const & iter)
{
	Eigen::MatrixXd New_V(V.rows(), V.cols());


	return New_V;
}




