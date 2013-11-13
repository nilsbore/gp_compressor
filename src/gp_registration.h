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
    double ls; // mean likelihood
    double cls; // mean color likelihood
    Eigen::RowVectorXd P; // mean derivative with respect to point
    Eigen::VectorXd delta; // step direction
    bool delta_diff_small;
    double added_derivatives; // number of derivatives in mean
    double step; // step size
    int step_nbr; // step iteration
    int max_steps; // maximum number of iteration
    asynch_visualizer* vis; // for visualizing the registration process and map
    void add_derivatives(const Eigen::MatrixXd& X, const Eigen::MatrixXd& dX);
    void compute_transformation();
    void get_local_points(Eigen::MatrixXd& points, int* occupied_indices, const std::vector<int>& index_search, int i);
    void get_transform_jacobian(Eigen::MatrixXd& J, const Eigen::Vector3d& x);
    void gradient_step(Eigen::Matrix3d& R, Eigen::Vector3d& t);
public:
    static void transform_pointcloud(pcl::PointCloud<pcl::PointXYZ>::Ptr c, const Eigen::Matrix3d& R, const Eigen::Vector3d& t);
    static void transform_pointcloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr c, const Eigen::Matrix3d& R, const Eigen::Vector3d& t);
    void add_cloud(pointcloud::ConstPtr other_cloud);
    bool registration_done();
    void registration_step();
    void get_cloud_transformation(Eigen::Matrix3d& R, Eigen::Vector3d& t);
    double get_likelihood();
    double get_color_likelihood();
    gp_registration(pointcloud::ConstPtr cloud, double res = 0.1f, int sz = 10,
                    asynch_visualizer* vis = NULL);
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

#endif // GP_REGISTRATION_H
