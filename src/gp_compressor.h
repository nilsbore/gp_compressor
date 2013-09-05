#ifndef GP_COMPRESSOR_H
#define GP_COMPRESSOR_H

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <string>
#include <vector>
#include <list>
#include "sparse_gp.h"

class gp_compressor
{
public:
    typedef pcl::PointXYZRGB point;
    typedef pcl::PointCloud<point> pointcloud;
protected:
    pointcloud::ConstPtr cloud; // the input pointcloud with RGB color information
    float res;
    int sz;

    // the rotations of the patches
    std::vector<Eigen::Quaternionf, Eigen::aligned_allocator<Eigen::Quaternionf> > rotations;
    // the 3D means of the patches
    std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f> > means;
    // the means in RGB color space
    //std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f> > RGB_means;

    std::vector<std::list<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f> > > S;
    // same for RGB, first the R vectors then G and B
    //Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> RGB;
    // the masks showing where in the patches there are observations
    Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> W;
    std::vector<sparse_gp> gps;

    void compute_rotation(Eigen::Matrix3f& R, const Eigen::MatrixXf& points);
    void project_points(Eigen::Vector3f& center, const Eigen::Matrix3f& R, Eigen::MatrixXf& points,
                        const Eigen::Matrix<short, Eigen::Dynamic, Eigen::Dynamic>& colors,
                        const std::vector<int>& index_search, int* occupied_indices, int i);
    void project_cloud();
    void compress_depths();
    void compress_colors();
    void train_processes();
public:
    gp_compressor(pointcloud::ConstPtr cloud, float res = 0.1f, int sz = 10);
    void save_compressed(const std::string& name);
    pointcloud::Ptr load_compressed();
};

#endif // GP_COMPRESSOR_H
