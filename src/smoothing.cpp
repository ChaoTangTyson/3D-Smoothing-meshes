
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
	Eigen::MatrixXd New_V = V;
	Eigen::MatrixXd Old_V = V;

	Eigen::SparseMatrix<double> I(V.rows(), V.rows());
	I.setIdentity();
	
	// cout << "Pass the first phase test" << endl;
	// Update oustide the loop and Get the Laplacian operator
	Eigen::SparseMatrix<double> Lapla = Laplace_Beltrami(V, F);

	for (int i = 0; i < iter; i++)
	{
		New_V = (I + lambda * Lapla) * Old_V;
		Old_V = New_V;
	}
	/*
	cout << "Pass the second phase test" << endl;
	cout << "New V1  " << New_V.row(0) << endl;
	cout << "Setp " << (lambda * Lapla * Old_V).rows() << endl;
	*/
	return  New_V;
}
///////////////////////////////////
/////////Implicit Laplacian ///////
///////////////////////////////////
Eigen::MatrixXd Im_smoothing(
	Eigen::MatrixXd const & V,
	Eigen::MatrixXi const & F,
	double const & lambda, int const & iter)
{
	Eigen::MatrixXd New_V = V;

	// cout << "Pass the first phase test" << endl;
	// Update oustide the loop and Get the Laplacian operator
	Eigen::SparseMatrix<double> M = Barycentric_Area(V, F);
	Eigen::SparseMatrix<double> C;
	igl::cotmatrix(V, F, C);

	Eigen::SparseMatrix<double> A = M - lambda * C;
	Eigen::SimplicialCholesky<Eigen::SparseMatrix<double >> cholesky(A);

	for (int i = 0; i < iter; i++)
	{
		New_V = cholesky.solve(M * New_V);
	}
	/*
	cout << "Pass the second phase test" << endl;
	cout << "New V1  " << New_V.row(0) << endl;
	cout << "Setp " << (lambda * Lapla * Old_V).rows() << endl;
	*/
	return  New_V;
}
///////////////////////////////////
/////////Add noise  ///////
///////////////////////////////////

Eigen::MatrixXd Noise(
	Eigen::MatrixXd const & V,
	float noiseLevel)
{
	Eigen::Vector3d m = V.colwise().minCoeff();
	Eigen::Vector3d M = V.colwise().maxCoeff();
	double xscale = (M[0] - m[0]) / 100;
	double yscale = (M[1] - m[1]) / 100;
	double zscale = (M[2] - m[2]) / 100;

	default_random_engine generator;
	normal_distribution<double> distribution(0.0, noiseLevel);

	Eigen::MatrixXd Noise_V;
	Noise_V.resizeLike(V);
	Noise_V.setZero();

	for (size_t i = 0; i < V.rows(); i++)
	{
		Noise_V.row(i)[0] = xscale * distribution(generator) + V.row(i)[0];
		Noise_V.row(i)[1] = yscale * distribution(generator) + V.row(i)[1];
		Noise_V.row(i)[2] = zscale * distribution(generator) + V.row(i)[2];
	}
	return Noise_V;
}





