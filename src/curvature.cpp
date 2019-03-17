
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

///////////////////////////////////
///////The curvature part//////////
///////////////////////////////////

void get_boundary_vex(
	Eigen::MatrixXd const & V, 
	Eigen::MatrixXi const & F,
	Eigen::MatrixXd & out_bvex)
{
	// You job is to use igl::boundary_loop to find boundary vertices
	// 
	// V : input vertices, N-by-3
	// F : input faces
	//
	// out_bvex : output vertices, K-by-3
	// 
	//  Hints:
	//   Eigen::VectorXi b_bex_index
	//     igl::boundary_loop( F_in , b_bex_index )
	// 
	Eigen::VectorXi b_bex_index;
	igl::boundary_loop(F, b_bex_index);
	// cout << "the size of b_b_index  " <<b_bex_index.size() << endl;
	out_bvex.resize(b_bex_index.size(), 3);

	for (size_t i=0; i < b_bex_index.size(); ++i)
	{
		out_bvex.row(i) = V.row(b_bex_index[i]);
		// cout << "b_bex_index[i]  "<< b_bex_index[i] << endl;
		// cout << "V.row(b_bex_index[i])  " << V.row(b_bex_index[i]) << endl;
	}
	//  cout << "the output of out_bvex   " << out_bvex.size() << endl;
}

void get_boundary_edges( 
	Eigen::MatrixXi const & F,
	Eigen::MatrixXi & out_b_edge)
{
	// You job is to use igl::boundary_facets to find boundary edges
	//  
	// F : input faces
	// 
	// out_bedge : output edges, K-by-2 (two vertices index)
	// 
	//  Hints:
	//   Eigen::MatrixXi b_edge
	//     igl::boundary_facets( F_in , b_edge )
	//  
	igl::boundary_facets(F, out_b_edge);
}

Eigen::SparseMatrix<double> Uniform_Laplacian(
	Eigen::MatrixXd const & V,
	Eigen::MatrixXi const & F) 
{
	// Input vertex V and F, this function is my method to generate the Uniform_Laplacian
	// The size of the sparse matrix
	int N = V.rows();
	// create a sparse matrix as the output
	Eigen::SparseMatrix<double> L_uniform(N, N);
	// Get the adjacent_vertex index for each vertex using igl built-in function
	std::vector<std::vector<int> > adj_V;
	igl::adjacency_list(F,adj_V);

	for (size_t i = 0; i < V.rows(); i++)
	{
		double Neighbour_Number = adj_V[i].size();
		// Add value to the Laplacian matrix,the neighbour vertex contribute 1/neighbout_numer.
		for (size_t j = 0; j < Neighbour_Number; j++)
		{
			L_uniform.insert(i, adj_V[i][j]) = 1.0 / Neighbour_Number;
		}
		// The diagonal value (which means the vertex i it self) contribute 1.
		L_uniform.insert(i, i) = -1.0;
	}
	return L_uniform;
}

Eigen::SparseMatrix<double> Barycentric_Area(
	Eigen::MatrixXd const & V,
	Eigen::MatrixXi const & F)
{
	int N = V.rows();
	Eigen::SparseMatrix<double> Area(N, N);
	
	// Caluclate the area for each Face so as to improve the efficiency by
	// Repearted calculating Face area
	
	Eigen::VectorXd Face_area(F.rows());
	for (size_t i = 0; i < F.rows(); i++)
	{
		Eigen::Vector3d v_1 = V.row((F.row(i)[0]));
		Eigen::Vector3d v_2 = V.row((F.row(i)[1]));
		Eigen::Vector3d v_3 = V.row((F.row(i)[2]));

		double Costheta = (v_2 - v_1).dot(v_3 - v_1) / ((v_2 - v_1).norm()*(v_3 - v_1).norm());
		double Sintheta = sqrt(1 - Costheta * Costheta);
		double current_area = 0.5 * (v_2 - v_1).norm() * (v_3 - v_1).norm() * Sintheta;

		Face_area[i] = current_area;
	}
	// std::cout << Face_area << std::endl;
	
	// Get the face index for each vertex using igl built-in function
	std::vector<std::vector<int> > VF;
	std::vector<std::vector<int> > VFi;
	igl::vertex_triangle_adjacency(V.rows(), F, VF, VFi);

	for (size_t i = 0; i < V.rows(); i++)
	{
		double Area_sum = 0;

		for (size_t j = 0; j < VF[i].size(); j++)
		{
			Area_sum = Area_sum + Face_area[VF[i][j]];
		}

		Area.insert(i, i) = Area_sum / 3.0;
	}
	// std::cout << "outside the loop" << std::endl;
	return Area;
}

Eigen::SparseMatrix<double> Laplace_Beltrami(
	Eigen::MatrixXd const & V,
	Eigen::MatrixXi const & F)
{
	// Input vertex V and F, this function is my method to generate the Non-Uniform Laplacian
	// The size of the sparse matrix
	int N = V.rows();
	// create a sparse matrix as the output
	Eigen::SparseMatrix<double> Lap_bel(N, N);

	// Hints
	// Compute Laplace-Beltrami operator
	//	Eigen::SparseMatrix<double> L, Area, AreaInv;
	//	igl::cotmatrix(V, F, L);
	//	igl::massmatrix(V, F, igl::MASSMATRIX_TYPE_VORONOI, Area );
	//	
	// 以下为验证集(validation code)
	Eigen::SparseMatrix<double>  Area_1, Area_2, C;
	igl::cotmatrix(V, F, C);
	igl::massmatrix(V, F, igl::MASSMATRIX_TYPE_BARYCENTRIC, Area_1);
	Area_2 = Barycentric_Area(V, F);
	//std::cout << "Area by my function for first element " << Area_1.coeff(0, 0) << std::endl;
	//std::cout << "Area by built-in-funciton for first element " << Area_2.coeff(0, 0) << std::endl;
	//以上为验证

	Eigen::SparseMatrix<double> AreaInv(N, N);
	
	igl::invert_diag(Barycentric_Area(V, F), AreaInv);

	Lap_bel = AreaInv * C;
	
	return Lap_bel;
}

Eigen::VectorXd compute_meanH(
	Eigen::MatrixXd const & V,
	Eigen::MatrixXi const & F,
	Eigen::SparseMatrix<double> Laplacian
	)
{
	// You job is to use igl::cotmatrix, igl::massmatrix, igl::invert_diag 
	// to compute mean curvature H at each vertex
	Eigen::MatrixXd dsL = Laplacian.toDense();
	Eigen::MatrixXd LaplacianV = dsL * V;
    std::cout << "successful get LaplacianV " << std::endl;

	//------------------------------------------
	// replace this 
	Eigen::VectorXd H;
	H.resize(V.rows());
	H.setZero();

	H = 0.5*(LaplacianV).rowwise().norm();
	return H;
}

Eigen::VectorXd compute_K(
	Eigen::MatrixXd const & V,
	Eigen::MatrixXi const & F)
{
	Eigen::VectorXd K(V.rows());
	//  Get the Area matrix
	Eigen::SparseMatrix<double> AreaM = Barycentric_Area(V, F);
	// Get the adjacent_vertex index for each vertex using igl built-in function
	std::vector<std::vector<int> > adj_V;
	// Set the sorted flag to be true 
	// sorted flag that indicates if the list should be sorted counter - clockwise
	igl::adjacency_list(F, adj_V,true);

	for (size_t i = 0; i < V.rows(); i++)
	{
		double current_Area = AreaM.coeff(i, i);
		double thetaSum = 0;
		
		for (int j = 0; j < adj_V[i].size(); j++)
		{
			Eigen::VectorXd CurrentV = V.row(i);
			Eigen::VectorXd V1 = V.row(adj_V[i][j]);
			Eigen::VectorXd V2 = V.row(adj_V[i][(j+1)% adj_V[i].size()]);
			// std::cout << "V1 " << adj_V[i][j]  << " with " << V.row(adj_V[i][j]) << std::endl;
			// std::cout << "V2 " << adj_V[i][(j + 1) % adj_V[i].size()] << " with " << V.row(adj_V[i][(j + 1) % adj_V[i].size()]) << std::endl;
			
			double cos_thetaj = (V1 - CurrentV).dot(V2 - CurrentV) / ((V1 - CurrentV).norm() * (V2 - CurrentV).norm() );
			double thetaj = acos(cos_thetaj);
			thetaSum = thetaSum + thetaj;
			// std::cout << "thetaSum  " << thetaSum << std::endl;
		}
		double M_PI = 3.1415926535897935;
		K[i] = (2 * M_PI - thetaSum) / current_Area;
		
	}

	return K;
}

void calculate_vertex_normal(
	Eigen::MatrixXd const & V,
	Eigen::MatrixXi const & F,
	Eigen::MatrixXd const & FN,
	Eigen::MatrixXd & out_VN)
{
	//
	// input:
	//   V: vertices
	//   F: face 
	//   FN: face normals
	// output:
	//   out_VN
	// 
	std::vector<std::vector<int> > VF;
	std::vector<std::vector<int> > VFi;
	igl::vertex_triangle_adjacency(V.rows(), F, VF, VFi);


	Eigen::MatrixXd VN(V.rows(), 3);

	for (int i = 0; i < V.rows(); i++)
	{
		Eigen::RowVector3d nv(0, 0, 0);

		for (int j = 0; j < VF[i].size(); j++)
		{
			nv = nv + FN.row(VF[i][j]);
		}

		nv = nv / VF[i].size();

		VN.row(i) = nv;
	}

	out_VN = VN;
}

///////////////////////////////////
///////The smoothing part//////////
///////////////////////////////////
