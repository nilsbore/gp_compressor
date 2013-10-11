#ifndef GP_COMPRESSOR_H
#define GP_COMPRESSOR_H

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <string>
#include <vector>
#include <list>
#include "sparse_gp.h"
#include "gp_octree.h"
#include "rbf_kernel.h"
#include "gaussian_noise.h"
#include "probit_noise.h"

class gp_compressor
{
public:
    typedef pcl::PointXYZRGB point;
    typedef pcl::PointCloud<point> pointcloud;
    typedef pcl::octree::OctreePointCloudSearch<point, gp_leaf>::OctreeT::LeafNodeIterator leaf_iterator;
protected:
    int iteration;
    pointcloud::Ptr cloud; // the input pointcloud with RGB color information
    gp_octree octree;
    double res;
    int sz;

    // the rotations of the patches
    std::vector<Eigen::Quaterniond, Eigen::aligned_allocator<Eigen::Quaterniond> > rotations;
    // the 3D means of the patches
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > means;
    // the means in RGB color space
    //std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > RGB_means;

    std::vector<std::list<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > > S; // more or less for debugging
    std::vector<std::list<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > > to_be_added;
    // same for RGB, first the R vectors then G and B
    //Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> RGB;
    // the masks showing where in the patches there are observations
    Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> W;
    std::vector<sparse_gp<rbf_kernel, gaussian_noise> > gps;

    void compute_rotation(Eigen::Matrix3d& R, const Eigen::MatrixXd& points);
    void project_points(Eigen::Vector3d& center, const Eigen::Matrix3d& R, Eigen::MatrixXd& points,
                        const Eigen::Matrix<short, Eigen::Dynamic, Eigen::Dynamic>& colors,
                        const std::vector<int>& index_search, int* occupied_indices, int i);
    void project_cloud();
    void compress_depths();
    void compress_colors();
    void train_processes();
public:
    gp_compressor(pointcloud::ConstPtr ncloud, double res = 0.1f, int sz = 10);
    void save_compressed(const std::string& name);
    pointcloud::Ptr load_compressed();
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

#endif // GP_COMPRESSOR_H
