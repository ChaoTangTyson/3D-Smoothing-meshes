#include <cmath>
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
#include <Eigen/Core>
#include <Eigen/SparseCore>

#include "curvature.h"
#include "Spectra/SymEigsSolver.h"
#include "Spectra/MatOp/SparseSymMatProd.h"

///////////////////////////////////
///////The curvature part//////////
///////////////////////////////////
double const M_PI = 3.1415926535897935;

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

Eigen::SparseMatrix<double> GetCot(
	Eigen::MatrixXd const & V,
	Eigen::MatrixXi const & F)
{
	
	int N = V.rows();
	Eigen::SparseMatrix<double> Cot(N, N);
	// Get adjacent face for each vertex
	std::vector<std::vector<int> > VF;
	std::vector<std::vector<int> > VFi;
	igl::vertex_triangle_adjacency(V.rows(), F, VF, VFi);
	std::cout << "VF0 size  " << VF[0].size() << std::endl;
	// Get adjacent vertices for each vertex
	std::vector<std::vector<int> > adj_V;
	igl::adjacency_list(F, adj_V, true);
	std::cout << "V0 size adj " << adj_V[0].size() << std::endl;

	for (int i = 0; i < N;i++) // only the first vertex
	{
		Eigen::VectorXd Vi = V.row(i);
		double sumcotVi = 0;
		for (int j = 0; j < adj_V[i].size(); j++)
		{
			int indexJ = adj_V[i][j];
			Eigen::VectorXd Vj = V.row(indexJ);
			double sumcotVj = 0;
			// Loop for each face to check if it has edges ViVj
			for (int f = 0; f < VF[i].size(); f++)
			{
				Eigen::VectorXi thisF = F.row(VF[i][f]);

				if (thisF[0] == i && thisF[1] == indexJ)
				{
					Eigen::VectorXd Vk = V.row(thisF[2]);
					double Costheta = (Vi - Vk).dot(Vj - Vk) / ((Vi - Vk).norm()*(Vj - Vk).norm());
					double cottheta = 1/tan(acos(Costheta));
					sumcotVj = sumcotVj + cottheta;
				}
				else if (thisF[0] == i && thisF[2] == indexJ)
				{
					Eigen::VectorXd Vk = V.row(thisF[1]);
					double Costheta = (Vi - Vk).dot(Vj - Vk) / ((Vi - Vk).norm()*(Vj - Vk).norm());
					double cottheta = 1 / tan(acos(Costheta));
					sumcotVj = sumcotVj + cottheta;
				}
				else if (thisF[1] == i && thisF[0] == indexJ)
				{
					Eigen::VectorXd Vk = V.row(thisF[2]);
					double Costheta = (Vi - Vk).dot(Vj - Vk) / ((Vi - Vk).norm()*(Vj - Vk).norm());
					double cottheta = 1 / tan(acos(Costheta));
					sumcotVj = sumcotVj + cottheta;
				}
				else if (thisF[2] == i && thisF[0] == indexJ)
				{
					Eigen::VectorXd Vk = V.row(thisF[1]);
					double Costheta = (Vi - Vk).dot(Vj - Vk) / ((Vi - Vk).norm()*(Vj - Vk).norm());
					double cottheta = 1 / tan(acos(Costheta));
					sumcotVj = sumcotVj + cottheta;
				}
				else if (thisF[2] == i && thisF[1] == indexJ)
				{
					Eigen::VectorXd Vk = V.row(thisF[0]);
					double Costheta = (Vi - Vk).dot(Vj - Vk) / ((Vi - Vk).norm()*(Vj - Vk).norm());
					double cottheta = 1 / tan(acos(Costheta));
					sumcotVj = sumcotVj + cottheta;
				}
				else if (thisF[1] == i && thisF[2] == indexJ)
				{
					Eigen::VectorXd Vk = V.row(thisF[0]);
					double Costheta = (Vi - Vk).dot(Vj - Vk) / ((Vi - Vk).norm()*(Vj - Vk).norm());
					double cottheta = 1 / tan(acos(Costheta));
					sumcotVj = sumcotVj + cottheta;
				}

			}
			Cot.insert(i, indexJ) = 0.5 * sumcotVj;
			sumcotVi = sumcotVi + 0.5 * sumcotVj;
		}
		Cot.insert(i, i) = - sumcotVi;
	}

	return Cot;
}

Eigen::SparseMatrix<double> GetCot2(
	Eigen::MatrixXd const & V,
	Eigen::MatrixXi const & F)
{
	int N = V.rows();
	Eigen::SparseMatrix<double> Cot(N, N);

	// Get adjacent vertices for each vertex
	std::vector<std::vector<int> > adj_V;
	igl::adjacency_list(F, adj_V, true);

	for (int i = 0; i < N; i++)
	{
		Eigen::VectorXd Vi = V.row(i);
		int neighbour = adj_V[i].size();
		double sumVi = 0;
		for (int j = 0; j < adj_V[i].size(); j++)
		{
			Eigen::Vector3d Vleft = V.row(adj_V[i][(j + neighbour - 1) % neighbour]);
			Eigen::Vector3d Vright = V.row(adj_V[i][(j + 1)% neighbour]);
			Eigen::Vector3d Vj = V.row(adj_V[i][j]);

			double angle1 = acos((Vi - Vleft).dot(Vj - Vleft) / ((Vi - Vleft).norm() * (Vj - Vleft).norm()));
			double angle2 = acos((Vi - Vright).dot(Vj - Vright) / ((Vi - Vright).norm() * (Vj - Vright).norm()));
			double cotan1 = 1/tan(angle1);
			double cotan2 = 1/tan(angle2);
			double sumVj = 0.5 * (cotan1 + cotan2);

			Cot.insert(i, adj_V[i][j]) = sumVj;
			sumVi += sumVj;
		}
		Cot.insert(i, i) = - sumVi;
	}
	return Cot;
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
	Eigen::SparseMatrix<double>  Area_1, Area_2, C,C_1,C_2;
	igl::cotmatrix(V, F, C);
	C_1 = GetCot(V, F);
	C_2 = GetCot2(V, F);
	igl::massmatrix(V, F, igl::MASSMATRIX_TYPE_BARYCENTRIC, Area_1);
	Area_2 = Barycentric_Area(V, F);
	//std::cout << "Area by my function for first element " << Area_1.coeff(0, 0) << std::endl;
	//std::cout << "Area by built-in-funciton for first element " << Area_2.coeff(0, 0) << std::endl;
	std::cout << "Cot by my function1 for first element " << C_1.coeff(100, 100) << std::endl;
	std::cout << "Cot by my function2 for first element " << C_2.coeff(100, 100) << std::endl;
	std::cout << "Cot by built-in-funciton for first element " << C.coeff(100, 100) << std::endl;
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
		double PI = 3.1415926535897935;
		K[i] = (2 * PI - thetaSum) / current_Area;
		
	}

	return K;
}

Eigen::MatrixXd Reconstruction(
	Eigen::MatrixXd const & V,
	Eigen::MatrixXi const & F,
	int const & k
)
{
	Eigen::MatrixXd New_V(V.rows(),V.cols());
	New_V.setZero() ;

	Eigen::SparseMatrix<double> M = Barycentric_Area(V, F);
	// Eigen::SparseMatrix<double> M(V.rows(), V.rows());
	// igl::massmatrix(V, F, igl::MASSMATRIX_TYPE_BARYCENTRIC, M);
	//  Eigen::SparseMatrix<double> M_inv(V.rows(), V.rows());
	// igl::invert_diag(M, M_inv);
	Eigen::SparseMatrix<double> Minv_half = M.cwiseSqrt().cwiseInverse();

	Eigen::SparseMatrix<double> C(V.rows(), V.rows());
	igl::cotmatrix(V, F, C);
	

	Eigen::SparseMatrix<double> Lapla_telta = Minv_half * -1.0 * C * Minv_half;

	// Spectra::SparseSymMatProd<double> op(C);
	Spectra::SparseSymMatProd<double> op(Lapla_telta);
	Spectra::SymEigsSolver< double, Spectra::SMALLEST_ALGE, Spectra::SparseSymMatProd<double> > eigs(&op, k, 2*k + 15);

	// Initialize and compute
	eigs.init();
	int nconv = eigs.compute();

	// Check if the Built-in-funcion works
	if (eigs.info() == Spectra::SUCCESSFUL)
	{
		std::cout << "Eigen Vectors calculation SUCCESSFUL" << std::endl;
	}
	else
	{
		std::cout << "Eigen Vectors NOT FOUND" << std::endl;
		std::cout << "Return the Original Vertex coordinates" << std::endl;
		return New_V;
	}
	// Retrieve results
	Eigen::MatrixXcd EigenVec_complex = eigs.eigenvectors();

	Eigen::MatrixXd EigenVec = Minv_half * EigenVec_complex.real();
	// std::cout << " Check the symetric of C " << EigenVec.col(0).transpose() * EigenVec.col(0) << std::endl;

	for (int i = 0; i < EigenVec.cols(); i++)
	{
		New_V +=  EigenVec.col(i) * (V.transpose() * M * EigenVec.col(i)).transpose();
	}
	return New_V;
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
