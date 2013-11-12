#ifndef GP_COMPRESSOR_H
#define GP_COMPRESSOR_H

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <string>
#include <vector>
#include <list>
#include "sparse_gp.h"
#include "sparse_gp_field.h"
#include "gp_octree.h"
#include "rbf_kernel.h"
#include "gaussian_noise.h"
#include "gaussian_noise_3d.h"
#include "probit_noise.h"

class gp_compressor
{
public:
    typedef pcl::PointXYZRGB point;
    typedef pcl::PointCloud<point> pointcloud;
    typedef pcl::octree::OctreePointCloudSearch<point, gp_leaf>::OctreeT::LeafNodeIterator leaf_iterator;
    typedef std::pair<Eigen::Vector3d, Eigen::Vector3d> point_pair;
protected:
    int iteration; // TEMP, just for plotting parts added in latest iteration
    pointcloud::Ptr cloud; // the input pointcloud with RGB color information
    gp_octree octree; // the octree for fast search in the cloud
    double res; // size of octree voxels
    int sz; // number of points for visualization along one side of patch
    // these two could be needed for ray-tracing from the camera
    // if that is to be done in this class instead of gp_mapping
    Eigen::Matrix3d R_cloud; // accumulated rotation of registered cloud, put in registration?
    Eigen::Vector3d t_cloud; // accumulated translation of registered cloud

    // the rotations of the patches
    std::vector<Eigen::Quaterniond, Eigen::aligned_allocator<Eigen::Quaterniond> > rotations;
    // the 3D means of the patches
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > means;
    // the means in RGB color space
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > RGB_means;

    // in the patch coordinate system, to be added this iteration
    std::vector<std::list<point_pair, Eigen::aligned_allocator<point_pair> > > S;
    // color values to be added
    //std::vector<std::list<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > > RGB; // TEST
    // in the global coordinate system, waiting to be added eventually
    //std::vector<std::list<std::Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > > to_be_added;
    std::vector<std::list<point_pair, Eigen::aligned_allocator<point_pair> > > to_be_added;

    // same for RGB, first the R vectors then G and B
    //Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> RGB;
    // the masks showing where in the patches there are observations
    Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> W;
    Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> free;
    std::vector<sparse_gp<rbf_kernel, gaussian_noise> > gps;
    std::vector<sparse_gp_field<rbf_kernel, gaussian_noise_3d> > RGB_gps;

    void compute_rotation(Eigen::Matrix3d& R, const Eigen::MatrixXd& points);
    void project_points(Eigen::Vector3d& center, const Eigen::Matrix3d& R, Eigen::MatrixXd& points,
                        const Eigen::MatrixXd& colors, const std::vector<int>& index_search, int* occupied_indices, int i);
    void project_cloud();
    void train_processes();
    void flatten_colors(Eigen::Matrix<short, 3, 1>& rtn, const Eigen::Vector3d& x);
public:
    gp_compressor(pointcloud::ConstPtr ncloud, double res = 0.1f, int sz = 10);
    void save_compressed(const std::string& name);
    pointcloud::Ptr load_compressed();
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

#endif // GP_COMPRESSOR_H
