#ifndef GP_REGISTRATION_H
#define GP_REGISTRATION_H

#include "gp_compressor.h"
#include <Eigen/Dense>

class gp_registration : public gp_compressor
{
public:
    using gp_compressor::point;
    using gp_compressor::pointcloud;
private:
    double step;
    Eigen::Vector3d mean1;
    Eigen::Vector3d mean2;
    double accumulated_weight;
    Eigen::Matrix3d covariance;
    void add_derivatives(const Eigen::MatrixXd& X, const Eigen::MatrixXd& dX);
    void get_transformation(Eigen::Matrix3d& R, Eigen::Vector3d& t);
    void compute_transformation();
    void get_local_points(Eigen::MatrixXf& points, int* occupied_indices, const std::vector<int>& index_search, int i);
public:
    void add_cloud(pointcloud::ConstPtr other_cloud);
    gp_registration(pointcloud::ConstPtr cloud, float res = 0.1f, int sz = 10);
};

#endif // GP_REGISTRATION_H
