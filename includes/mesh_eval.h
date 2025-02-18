//
// Created by Yidi Ma on 14.02.25.
//

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include <iomanip>
#include <cmath>
#include <fstream>
#include <igl/readOFF.h>
#include <igl/cotmatrix.h>
#include <igl/principal_curvature.h>
#include <vector>


Eigen::MatrixXd V; // Vertices (Nx3)
Eigen::MatrixXi F; // Faces (Mx3)


void loadOffFile(const std::string pathToOffFile) {
    if (!igl::readOFF(pathToOffFile, V, F)) {
        std::cerr << "Error loading OFF file: " << pathToOffFile << std::endl;
        exit(-1);
    }
    std::cout << "Loaded mesh with " << V.rows() << " vertices and " << F.rows() << " faces.\n";

}


void laplacianSmoothness() {
    Eigen::SparseMatrix<double> L;
    igl::cotmatrix(V, F, L); // Compute Laplacian
    Eigen::MatrixXd smoothness = L * V; // Laplacian applied to vertex positions

    std::cout << "Laplacian smoothness norm: " << smoothness.norm() << std::endl;
}


void edgeLengthDistr() {
    std::vector<double> edgeLengths;

    for (int i = 0; i < F.rows(); i++) {
        for (int j = 0; j < 3; j++) { // Iterate over triangle edges
            int v1 = F(i, j);
            int v2 = F(i, (j+1) % 3);

            double length = (V.row(v1) - V.row(v2)).norm();
            edgeLengths.push_back(length);
        }
    }

    double sum = 0, sumSq = 0;
    for (double len : edgeLengths) {
        sum += len;
        sumSq += len * len;
    }
    double mean = sum / edgeLengths.size();
    double variance = (sumSq / edgeLengths.size()) - (mean * mean);

    std::cout << "Edge Length - Mean: " << mean << ", Variance: " << variance << std::endl;

}


void computeCurvature() {
    Eigen::MatrixXd PD1, PD2; // Principal directions
    Eigen::VectorXd PV1, PV2; // Principal curvatures

    igl::principal_curvature(V, F, PD1, PD2, PV1, PV2);

    Eigen::VectorXd meanCurvature = 0.5 * (PV1 + PV2);
    Eigen::VectorXd gaussianCurvature = PV1.cwiseProduct(PV2);

    std::cout << "Mean curvature range: [" << meanCurvature.minCoeff() << ", " << meanCurvature.maxCoeff() << "]\n";
    std::cout << "Gaussian curvature range: [" << gaussianCurvature.minCoeff() << ", " << gaussianCurvature.maxCoeff() << "]\n";

}


void mesh_evaluation(const std::string pathToOffFile) {
    // 1) load OFF file
    loadOffFile(pathToOffFile);

    // 2) Laplacian Smoothness
    laplacianSmoothness();

    // 3) edge length distribution
    edgeLengthDistr();

    // 4) mean and gaussian curvature
    // computeCurvature();

}