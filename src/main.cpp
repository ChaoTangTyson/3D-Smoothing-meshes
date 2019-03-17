#include <igl/readOFF.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <igl/vertex_triangle_adjacency.h>
#include <imgui/imgui.h>
#include <igl/readPLY.h>
#include <igl/cotmatrix.h>
#include <igl/massmatrix.h>
#include <igl/invert_diag.h>
#include <igl/file_exists.h>
#include <igl/setdiff.h> 
#include <igl/slice.h>
#include <igl/slice_into.h>
#include <igl/boundary_loop.h>
#include <igl/boundary_facets.h>
#include <igl/unique.h>

#include <Eigen/Dense>
#include <Eigen/SVD>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <Eigen/LU>

// #include <Spectra/SymEigsSolver.h>

#include <iostream>
#include "curvature.h"
#include "smoothing.h"


using namespace std;


class MyContext
{
public:

	MyContext() :nv_len(0), point_size(10), line_width(6), mode(0),lambda(0.01), iteration(1)
	{

	}
	~MyContext() {}

	Eigen::MatrixXd m_V;
	Eigen::MatrixXi m_F;	 
	Eigen::MatrixXd m_VN;
	Eigen::MatrixXd m_C;
	Eigen::MatrixXd input_pts;

	Eigen::SparseMatrix<double> lmat;

	int m_num_vex;
	float nv_len;
	float point_size;
	float line_width;
	
	int k;// The number of EigenVector used to reconstruct the meshes
 	int mode;
	int iteration;
	float lambda ;
	float percentage_lambda = 0.1;

	void concate(Eigen::MatrixXd const & VA,
		Eigen::MatrixXi const & FA, 
		Eigen::MatrixXd const & VB, 
		Eigen::MatrixXi const & FB,
		Eigen::MatrixXd & out_V,
		Eigen::MatrixXi & out_F	)
	{

		out_V.resize(VA.rows() + VB.rows(), VA.cols());
		out_V << VA, VB;
		out_F.resize(FA.rows() + FB.rows(), FA.cols());
		out_F << FA, (FB.array() + VA.rows());
		
	}

	void reset_display(igl::opengl::glfw::Viewer& viewer)
	{
		viewer.data().clear(); 
		// hide default wireframe
		viewer.data().show_lines = 0;
		viewer.data().show_overlay_depth = 1; 

		//======================================================================

		viewer.data().line_width = line_width;
		viewer.data().point_size = point_size;

		if (mode == 0 )
		{
			// add mesh
			viewer.data().set_mesh(m_V, m_F);
			viewer.core.align_camera_center(m_V, m_F);
		}
		else if (mode == 1)
		{
			// add mesh
			viewer.data().set_mesh(m_V, m_F);
			viewer.data().set_colors(m_C);
			viewer.core.align_camera_center(m_V, m_F); 
		}

	}

private:

};

MyContext g_myctx;

bool key_down(igl::opengl::glfw::Viewer& viewer, unsigned char key, int modifier)
{

	std::cout << "Key: " << key << " " << (unsigned int)key << std::endl;
	if (key=='q' || key=='Q')
	{
		exit(0);
	}
	return false;
}

void get_example_mesh(std::string const meshname , Eigen::MatrixXd & V, Eigen::MatrixXi & F, Eigen::MatrixXd & VN)
{


	std::vector<const char *> cands{ 
		"../../example_meshes/", 
		"../../../example_meshes/",
		"../../../../example_meshes/",
		"../../../../../example_meshes/" };

	bool found = false;
	for (const auto & val : cands)
	{
		if ( igl::file_exists(val+ meshname) )
		{	
			std::cout << "loading example mesh from:" << val+ meshname << "\n";

			if (igl::readOFF(val+ meshname, V,F)) {
				igl::per_vertex_normals(V, F, VN);
				found = 1;
				break;
			}
			else {
				std::cout << "file loading failed " << cands[0] + meshname << "\n"; 
			}
		}
	}

	if (!found) {
		std::cout << "cannot locate "<<cands[0]+ meshname <<"  Press any key to exit" << endl ;
		char c;
		cin >> c;
		exit(1);
	}

}


int main(int argc, char *argv[])
{
	//------------------------------------------
	// load data  
	Eigen::MatrixXd V;
	Eigen::MatrixXd VN;
	Eigen::MatrixXi F;  

	get_example_mesh("camelhead.off", V, F, VN);
	std::cout << "eigen version.:" << EIGEN_WORLD_VERSION << "," << EIGEN_MAJOR_VERSION << EIGEN_MINOR_VERSION << "\n";
	
	//------------------------------------------
	// for visualization
	g_myctx.m_V = V;
	g_myctx.m_F = F;
	g_myctx.m_VN = VN;

	//------------------------------------------
	// Init the viewer
	igl::opengl::glfw::Viewer viewer;

	// Attach a menu plugin
	igl::opengl::glfw::imgui::ImGuiMenu menu;
	viewer.plugins.push_back(&menu);

	// menu variable Shared between two menus
	double doubleVariable = 0.1f; 
	// The index number of the current mesh
	int current_mesh = 0;
	// Add content to the default menu window via defining a Lambda expression with captures by reference([&])
	menu.callback_draw_viewer_menu = [&]()
	{
		// Draw parent menu content
		menu.draw_viewer_menu();

		// Add new group
		if (ImGui::CollapsingHeader("New Group", ImGuiTreeNodeFlags_DefaultOpen))
		{
			// Expose variable directly ...
			ImGui::InputDouble("double", &doubleVariable, 0, 0, "%.4f");

			// ... or using a custom callback
			static bool boolVariable = true;
			if (ImGui::Checkbox("bool", &boolVariable))
			{
				// do something
				std::cout << "boolVariable: " << std::boolalpha << boolVariable << std::endl;
			}

			// Expose an enumeration type
			enum Orientation { Up = 0, Down, Left, Right };
			static Orientation dir = Up;
			ImGui::Combo("Direction", (int *)(&dir), "Up\0Down\0Left\0Right\0\0");

			// We can also use a std::vector<std::string> defined dynamically
			static int num_choices = 3;
			static std::vector<std::string> choices;
			static int idx_choice = 0;
			if (ImGui::InputInt("Num letters", &num_choices))
			{
				num_choices = std::max(1, std::min(26, num_choices));
			}
			if (num_choices != (int)choices.size())
			{
				choices.resize(num_choices);
				for (int i = 0; i < num_choices; ++i)
					choices[i] = std::string(1, 'A' + i);
				if (idx_choice >= num_choices)
					idx_choice = num_choices - 1;
			}
			ImGui::Combo("Letter", &idx_choice, choices);

		}
	};

	// Add additional windows via defining a Lambda expression with captures by reference([&])
	
	menu.callback_draw_custom_window = [&]()
	{
		// Define next window position + size
		ImGui::SetNextWindowPos(ImVec2(180.f * menu.menu_scaling(), 10), ImGuiSetCond_FirstUseEver);
		ImGui::SetNextWindowSize(ImVec2(250, 400), ImGuiSetCond_FirstUseEver);
		ImGui::Begin( "MyProperties", nullptr, ImGuiWindowFlags_NoSavedSettings );
		
		// select mesh
		const char* mesh_list[] = { "camelhead","bunny" , "camel" ,"cow","cube" };
		
		if (ImGui::ListBox("mesh_list\n(single select)", &current_mesh, mesh_list, IM_ARRAYSIZE(mesh_list), 6))
		{
			if (current_mesh ==  0){get_example_mesh("camelhead.off", g_myctx.m_V, g_myctx.m_F, g_myctx.m_VN);}
			else if (current_mesh == 1) { get_example_mesh("bunny.off", g_myctx.m_V, g_myctx.m_F, g_myctx.m_VN); }
			else if (current_mesh == 2) { get_example_mesh("camel.off", g_myctx.m_V, g_myctx.m_F, g_myctx.m_VN); }
			else if (current_mesh == 3) { get_example_mesh("cow.off", g_myctx.m_V, g_myctx.m_F, g_myctx.m_VN); }
			else if (current_mesh == 4) { get_example_mesh("cube.off", g_myctx.m_V, g_myctx.m_F, g_myctx.m_VN); }

			g_myctx.mode = 0;
			g_myctx.reset_display(viewer);
		}

		// point size
		// [event handle] if value changed
		/*
	    if (ImGui::InputFloat("point_size", &g_myctx.point_size))
		{
			std::cout << "point_size changed\n";
			viewer.data().point_size = g_myctx.point_size;
		}
		
		// line width
		// [event handle] if value changed
		if(ImGui::InputFloat("line_width", &g_myctx.line_width))
		{
			std::cout << "line_width changed\n";
			viewer.data().line_width = g_myctx.line_width;
		}
		*/
		/////////////////////////////////////////////////////////////////
		/////////////////////Uniform Discretization//////////////////////
		/////////////////////////////////////////////////////////////////
		if (ImGui::CollapsingHeader("Uniform Discretization", ImGuiTreeNodeFlags_DefaultOpen))
		{
			if (ImGui::Button("Mean curvature", ImVec2(-1, 0)))
			{
				// Calculate the uniform Mean curvature
				Eigen::SparseMatrix<double> L_uni;
				L_uni = Uniform_Laplacian(g_myctx.m_V, g_myctx.m_F);
				cout << "Sparse L_uni done" << endl;
				Eigen::VectorXd H_uni;
				H_uni = compute_meanH(g_myctx.m_V, g_myctx.m_F, L_uni);

				H_uni = 5 * H_uni.array() / (H_uni.maxCoeff() - H_uni.minCoeff());
				//replace by color scheme
				igl::parula(H_uni, false, g_myctx.m_C);

				g_myctx.mode = 1;
				g_myctx.reset_display(viewer);
			}
			if (ImGui::Button("Gaussian curvature", ImVec2(-1, 0)))
			{
				// Calculate the uniform Gaussian curvature
				// waiting to change
				Eigen::SparseMatrix<double> L_uni;
				L_uni = Uniform_Laplacian(g_myctx.m_V, g_myctx.m_F);
				cout << "Sparse L_uni done" << endl;

				Eigen::VectorXd K = compute_K(g_myctx.m_V, g_myctx.m_F);
				K = 1000 * K.array() / (K.maxCoeff() - K.minCoeff());

				//replace by color scheme
				igl::parula(K, false, g_myctx.m_C);

				g_myctx.mode = 1;
				g_myctx.reset_display(viewer);
			}
		}
		/////////////////////////////////////////////////////////////////
		/////////////////////Non-Uniform L-B part/////////////////////////
		/////////////////////////////////////////////////////////////////
		if (ImGui::CollapsingHeader("Non-uniform Discretization", ImGuiTreeNodeFlags_DefaultOpen))
		{
			if (ImGui::Button("Non-uniform Mean curvature", ImVec2(-1, 0)))
			{
				// Calculate the uniform Mean curvature
				Eigen::SparseMatrix<double> L_B;
				L_B = Laplace_Beltrami(g_myctx.m_V, g_myctx.m_F);
				cout << "Sparse Laplace_Beltrami matrix done" << endl;
				Eigen::VectorXd H_nonU;
				H_nonU = compute_meanH(g_myctx.m_V, g_myctx.m_F, L_B);

				H_nonU = 10 * H_nonU.array() / (H_nonU.maxCoeff() - H_nonU.minCoeff());
				//replace by color scheme
				igl::parula(H_nonU, false, g_myctx.m_C);

				g_myctx.mode = 1;
				g_myctx.reset_display(viewer);
			}
		}
		/////////////////////////////////////////////////////////////////
		/////////////////////Reconstruction part/////////////////////////
		/////////////////////////////////////////////////////////////////
		// ImGui::Text("Reconstruction with K EigenVectors");
		if (ImGui::CollapsingHeader("Reconstruction with K EigenVectors", ImGuiTreeNodeFlags_DefaultOpen))
		{ 
			if (ImGui::InputInt("EigenVector used", &g_myctx.k))
			{
				cout << "There are " << g_myctx.k << "  EigenVectors used to reconstruct meshes"<<endl;
			}
			if (ImGui::Button("Reconstruction", ImVec2(-1, 0)))
			{
				// Waiting to change
				g_myctx.mode = 1;
				g_myctx.reset_display(viewer);
			}

		}
		/////////////////////////////////////////////////////////////////
		/////////////////////Mesh smoothing part////////////////
		/////////////////////////////////////////////////////////////////
		if (ImGui::CollapsingHeader("Laplacian Mesh Smoothing", ImGuiTreeNodeFlags_DefaultOpen))
		{
			if (ImGui::InputFloat("Time step lambda", &g_myctx.percentage_lambda ))
			{
				// Get the size of the current model
				Eigen::Vector3d m = g_myctx.m_V.colwise().minCoeff();
				Eigen::Vector3d M = g_myctx.m_V.colwise().maxCoeff();

				double ModelSize = (M - m).norm();
				cout << "The model size :  " << ModelSize << endl;
				g_myctx.lambda = g_myctx.percentage_lambda / 100000 * ModelSize;
				cout << "Time step set :  " << g_myctx.lambda << endl;
			}
			if (ImGui::InputInt("Iteration", &g_myctx.iteration))
			{
				cout << "Iteration set :  " << g_myctx.iteration << endl;
			}
			if (ImGui::Button("Explicit Smoothing", ImVec2(-1, 0)))
			{
				Eigen::MatrixXd New_V = Ex_smoothing(g_myctx.m_V, g_myctx.m_F, g_myctx.lambda, g_myctx.iteration);
				g_myctx.m_V = New_V;

				Eigen::SparseMatrix<double> L_B;
				L_B = Laplace_Beltrami(g_myctx.m_V, g_myctx.m_F);
				// cout << "Sparse Laplace_Beltrami matrix done" << endl;

				Eigen::VectorXd H_nonU;
				H_nonU = compute_meanH(g_myctx.m_V, g_myctx.m_F, L_B);

				H_nonU = 10 * H_nonU.array() / (H_nonU.maxCoeff() - H_nonU.minCoeff());
				//replace by color scheme
				igl::parula(H_nonU, false, g_myctx.m_C);

				g_myctx.mode = 1;
				g_myctx.reset_display(viewer);
			}
			if (ImGui::Button("Implicit Smoothing", ImVec2(-1, 0)))
			{
				// Waiting to change
				g_myctx.mode = 1;
				g_myctx.reset_display(viewer);
			}

		}

		//mode - List box
		/*
		const char* listbox_items[] = { "Vex/Edge" , "H" ,"k"};
		if (ImGui::ListBox("listbox\n(single select)", & g_myctx.mode, listbox_items, IM_ARRAYSIZE(listbox_items), 4))
		{
			g_myctx.reset_display(viewer);
		}
		*/
		ImGui::End();
	};


	// registered a event handler
	viewer.callback_key_down = &key_down;

	g_myctx.reset_display(viewer);

	// Call GUI
	viewer.launch();

}
