//
// Created by Yidi Ma on 01.02.25.
//

#ifndef MESH_GENERATION_H
#define MESH_GENERATION_H

#endif //MESH_GENERATION_H

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <Eigen/Core>
#include <vector>
#include <iomanip>
#include <cmath>
#include <fstream>



struct Vertex {
    float x, y, z;
    Eigen::Vector4f position;
};

struct Triangle {
    int v1, v2, v3;
};


/**
 * @brief Convert depth map pixels to 3D points
 * @param depthMap    Input depth map
 * @param K           Camera intrinsic matrix
 * @return vector of 3D vertices
 */
inline std::vector<Vertex> depthMapToPointCloud(
    const cv::Mat& depthMap, const cv::Mat& K,
    const float threshold_lower_bound,
    const float threshold_upper_bound)
{

    std::vector<Vertex> vertices(depthMap.rows * depthMap.cols);  // Preallocate with grid size

    float fx = K.at<double>(0, 0);
    float fy = K.at<double>(1, 1);
    float cx = K.at<double>(0, 2);
    float cy = K.at<double>(1, 2);

    for(int y = 0; y < depthMap.rows; y++) {
        for(int x = 0; x < depthMap.cols; x++) {

            int idx = y * depthMap.cols + x;  // Linear index
            float depth = depthMap.at<float>(y, x);

            // Initialize with invalid values
            vertices[idx] = {0.0f, 0.0f, -1.0f};  // Use z = -1 to mark invalid

            if(depth > threshold_lower_bound &&
                depth < threshold_upper_bound) {
                Vertex v;
                v.z = depth;
                v.x = (x - cx) * depth / fx;
                v.y = (y - cy) * depth / fy;
                v.position = Eigen::Vector4f(v.x, v.y, v.z, 1.0f);

                if (std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z)) {
                    vertices[idx] = v;
                }
            }
        }
    }
    return vertices;
}


inline bool isVertexValid(const Vertex& vertex) {
    return !std::isnan(vertex.x) &&
           !std::isnan(vertex.y) &&
           !std::isnan(vertex.z);
}

inline bool isEdgeValid(const Vertex& v1, const Vertex& v2, float edgeThreshold) {
    float dist = (v1.position - v2.position).norm();
    return dist < edgeThreshold;
}

inline bool isTriangleValid(const Vertex& v0, const Vertex& v1, const Vertex& v2, float edgeThreshold) {
    if (isVertexValid(v0) && isVertexValid(v1) && isVertexValid(v2)) {
        // Check edge validity
        bool edgesValid = isEdgeValid(v0, v1, edgeThreshold) &&
                          isEdgeValid(v0, v2, edgeThreshold) &&
                          isEdgeValid(v1, v2, edgeThreshold);

        return edgesValid;  // Ensure non-zero area
    }
    return false;
}

/**
 * @brief Create triangles from organized point cloud
 * @param vertices    Original depth map for checking valid depths
 * @param width       Width of depth map
 * @param height      Height of depth map
 * @return vector of triangles
 */
inline std::vector<Triangle> generateMeshTriangles(const std::vector<Vertex> vertices, int width, int height) {
    std::vector<Triangle> triangles;
    std::vector<std::vector<int>> vertexMap(height, std::vector<int>(width, -1));
    int currentVertexIndex = 0;

    // First pass: create vertex index mapping
    for(int y = 0; y < height; y++) {
        for(int x = 0; x < width; x++) {
            float vertex_idx = y * width + x;
            if(vertices[vertex_idx].z > 0.0f) {
                vertexMap[y][x] = currentVertexIndex;
            }
            currentVertexIndex++;
        }
    }

    // Second pass: create triangles using valid vertex indices
    float edgeThreshold = 0.02; // 2cm
    int invalidVertex = 0;
    int invalidEdge = 0;

    for(int y = 0; y < height-1; y++) {
        for(int x = 0; x < width-1; x++) {
            // Get vertex indices for the quad corners
            int idx00 = vertexMap[y][x];
            int idx01 = vertexMap[y][x+1];
            int idx10 = vertexMap[y+1][x];
            int idx11 = vertexMap[y+1][x+1];

            // Skip if any vertex is invalid -> index of -1 means vertex has negative depth and was filtered out
            if(idx00 < 0 || idx01 < 0 || idx10 < 0 || idx11 < 0) {
                invalidVertex++;
                continue;
            }

            // Get 3D points for the quad corners
            const Vertex& p00 = vertices[idx00];
            const Vertex& p01 = vertices[idx01];
            const Vertex& p10 = vertices[idx10];
            const Vertex& p11 = vertices[idx11];

            /*
            // Check edge validity using Euclidean distance
            auto distance = [](const Vertex& a, const Vertex& b) {
                return std::sqrt((a.x - b.x) * (a.x - b.x) +
                                 (a.y - b.y) * (a.y - b.y) +
                                 (a.z - b.z) * (a.z - b.z));
            };

            // Skip if depth discontinuity is too large
            if (distance(p00, p01) >= edgeThreshold ||
                distance(p00, p10) >= edgeThreshold ||
                distance(p01, p11) >= edgeThreshold ||
                distance(p10, p11) >= edgeThreshold) {
                edgeThresholdViolations++;
                continue;
            }
            */


            if (isTriangleValid(p00, p01, p10, edgeThreshold)) {
                triangles.push_back({idx00, idx10, idx01});
            } else {
                invalidEdge++;
            }

            if (isTriangleValid(p01, p10, p11, edgeThreshold)) {
                triangles.push_back({idx01, idx10, idx11});
            } else {
                invalidEdge++;
            }


            /*
            // Create two triangles
            Triangle t1{idx00, idx10, idx01};
            Triangle t2{idx01, idx10, idx11};

            triangles.push_back(t1);
            triangles.push_back(t2);
            */
        }
    }

    std::cout << "[MESH] Rejected triangles due to invalid vertices: " << invalidVertex << std::endl;
    std::cout << "[MESH] Rejected triangles due to invalid edges: " << invalidEdge << std::endl;
    std::cout << "[MESH] Generated valid triangles: " << triangles.size() << std::endl;

    return triangles;
}


/**
 * @brief Save mesh to PLY file
 * @param filename    Output filename
 * @param vertices    Vector of vertices
 * @param triangles   Vector of triangles
 */
inline void savePLYMesh(const std::string& filename,
                        const std::vector<Vertex>& vertices,
                        const std::vector<Triangle>& triangles) {
    // Verify indices
    int maxIndex = vertices.size() - 1;
    for(const auto& t : triangles) {
        if(t.v1 < 0 || t.v2 < 0 || t.v3 < 0 ||
           t.v1 > maxIndex || t.v2 > maxIndex || t.v3 > maxIndex) {
            std::cerr << "Invalid triangle indices: " << t.v1 << ", " << t.v2 << ", " << t.v3
                      << " (max valid index: " << maxIndex << ")" << std::endl;
            return;
           }
    }

    std::ofstream file(filename);
    if(!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }

    // Write PLY header
    file << "ply\n";
    file << "format ascii 1.0\n";
    file << "element vertex " << vertices.size() << "\n";
    file << "property float x\n";
    file << "property float y\n";
    file << "property float z\n";
    file << "element face " << triangles.size() << "\n";
    file << "property list uchar int vertex_indices\n";
    file << "end_header\n";

    // Write vertices
    for(const auto& v : vertices) {
        file << v.x << " " << v.y << " " << v.z << "\n";
    }

    // Write triangles
    for(const auto& t : triangles) {
        file << "3 " << t.v1 << " " << t.v2 << " " << t.v3 << "\n";
    }

    file.close();
}

/**
 * @brief Save mesh to OFF file
 * @param filename    Output filename
 * @param vertices    Vector of vertices
 * @param triangles   Vector of triangles
 */
inline void saveOFFMesh(const std::string& filename,
                        const std::vector<Vertex>& vertices,
                        const std::vector<Triangle>& triangles) {
    // Verify indices
    int maxIndex = vertices.size() - 1;
    for (const auto& t : triangles) {
        if (t.v1 < 0 || t.v2 < 0 || t.v3 < 0 ||
            t.v1 > maxIndex || t.v2 > maxIndex || t.v3 > maxIndex) {
            std::cerr << "Invalid triangle indices: " << t.v1 << ", " << t.v2 << ", " << t.v3
                      << " (max valid index: " << maxIndex << ")" << std::endl;
            return;
            }
    }

    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }

    // Write OFF header
    file << "OFF\n";
    file << vertices.size() << " " << triangles.size() << " 0\n";

    // Write vertices
    for (const auto& v : vertices) {
        file << v.x << " " << v.y << " " << v.z << "\n";
    }

    // Write triangles
    for (const auto& t : triangles) {
        file << "3 " << t.v1 << " " << t.v2 << " " << t.v3 << "\n";
    }

    file.close();
    std::cout << "Saved mesh to " << filename << std::endl;
}



/**
 * @brief Save point cloud to PLY file without mesh triangles
 * @param filename    Output filename
 * @param vertices    Vector of vertices
 */
inline void savePLYPointCloud(const std::string& filename, const std::vector<Vertex>& vertices) {
    // Filter out invalid vertices (those with z = -1)
    std::vector<Vertex> valid_vertices;
    valid_vertices.reserve(vertices.size());
    for (const auto& v : vertices) {
        if (v.z > -1.0f && std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z)) {
            valid_vertices.push_back(v);
        }
    }

    std::ofstream file(filename, std::ios::binary);
    if(!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }

    // Write PLY header
    file << "ply\n";
    file << "format ascii 1.0\n";  // Using ASCII format for better compatibility
    file << "element vertex " << valid_vertices.size() << "\n";
    file << "property float x\n";
    file << "property float y\n";
    file << "property float z\n";
    file << "end_header\n";

    // Write vertices
    for(const auto& v : valid_vertices) {
        file << std::fixed << std::setprecision(6)  // 6 decimal places for precision
             << v.x << " " << v.y << " " << v.z << "\n";
    }

    file.close();
    std::cout << "Saved " << valid_vertices.size() << " valid points to " << filename << std::endl;
}


/**
 * @brief Main function to generate mesh from depth map
 * @param depthMap    Input depth map
 * @param K           Camera intrinsic matrix
 * @param dataset     dataset name
 * @param upperDistanceThreshold
 * @param lowerDistanceThreshold
 */
inline void generateMeshFromDepth(
    const cv::Mat& depthMap, const cv::Mat& K, const std::string& dataset,
    const float upperDistanceThreshold, const float lowerDistanceThreshold) {
    // Add some debug info
    cv::Mat depth_valid;
    cv::threshold(depthMap, depth_valid, 0.1, 1, cv::THRESH_BINARY);

    // Sample some depth values
    for(int y = 0; y < depthMap.rows; y += depthMap.rows/4) {
        for(int x = 0; x < depthMap.cols; x += depthMap.cols/4) {
            std::cout << "[DEPTH] Depth at (" << x << "," << y << "): "
                      << depthMap.at<float>(y,x) << std::endl;
        }
    }

    // Convert depth map to point cloud and save point cloud for visualization
    std::vector<Vertex> vertices = depthMapToPointCloud(depthMap, K, lowerDistanceThreshold, upperDistanceThreshold);
    std::string pcOtputPath = "/workspace/results/" + dataset + "_pointclouds.ply";
    savePLYPointCloud(pcOtputPath, vertices);

    // Generate triangles
    std::vector<Triangle> triangles = generateMeshTriangles(vertices, depthMap.cols, depthMap.rows);

    // Save mesh to PLY file
    std::string meshOutputPath = "/workspace/results/" + dataset + "_mesh.off";
    // savePLYMesh(outputPath, vertices, triangles);
    saveOFFMesh(meshOutputPath, vertices, triangles);

    std::cout << "[MESH] Mesh generated with " << vertices.size() << " vertices and "
              << triangles.size() << " triangles" << std::endl;
}



