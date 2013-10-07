#ifndef GP_REGISTRATION_H
#define GP_REGISTRATION_H

#include "gp_compressor.h"
#include "asynch_visualizer.h"
#include <Eigen/Dense>

class gp_registration : public gp_compressor
{
public:
    using gp_compressor::point;
    using gp_compressor::pointcloud;
protected:
    double ls;
    Eigen::RowVectorXd P;
    Eigen::VectorXd delta;
    double added_derivatives;
    double step;
    Eigen::Matrix3d R_cloud;
    Eigen::Vector3d t_cloud;
    int step_nbr;
    /*Eigen::Vector3d mean1;
    Eigen::Vector3d mean2;
    double accumulated_weight;
    Eigen::Matrix3d covariance;*/
    //pcl::PointCloud<pcl::PointXYZ>::Ptr ncenters;
    //pcl::PointCloud<pcl::Normal>::Ptr normals;
    asynch_visualizer* vis;
    void add_derivatives(const Eigen::MatrixXd& X, const Eigen::MatrixXd& dX);
    //void get_transformation(Eigen::Matrix3d& R, Eigen::Vector3d& t);
    void compute_transformation();
    void get_local_points(Eigen::MatrixXd& points, int* occupied_indices, const std::vector<int>& index_search, int i);
    void get_transform_jacobian(Eigen::MatrixXd& J, const Eigen::Vector3d& x);
    void gradient_step(Eigen::Matrix3d& R, Eigen::Vector3d& t);
public:
    void transform_pointcloud(pcl::PointCloud<pcl::PointXYZ>::Ptr c, const Eigen::Matrix3d& R, const Eigen::Vector3d& t);
    void transform_pointcloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr c, const Eigen::Matrix3d& R, const Eigen::Vector3d& t);
    void add_cloud(pointcloud::ConstPtr other_cloud);
    bool registration_done();
    void registration_step();
    void get_cloud_transformation(Eigen::Matrix3d& R, Eigen::Vector3d& t);
    gp_registration(pointcloud::ConstPtr cloud, double res = 0.1f, int sz = 10,
                    asynch_visualizer* vis = NULL);
};

#endif // GP_REGISTRATION_H
