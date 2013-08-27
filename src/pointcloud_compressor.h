#ifndef POINTCLOUD_COMPRESSOR_H
#define POINTCLOUD_COMPRESSOR_H

#include "dictionary_representation.h"
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <string>
#include <vector>
#include <list>

class pointcloud_compressor : public dictionary_representation
{
public:
    typedef pcl::PointXYZRGB point;
    typedef pcl::PointCloud<point> pointcloud;
private:
    pointcloud::ConstPtr cloud; // the input pointcloud with RGB color information
    // DEBUG, calculating the rms error
    std::vector<std::list<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f> > > patch_points;

    float proj_error; // the error at which the orthogonal matching pursuit stops in ksvd
    float RGB_proj_error; // the error at which the orthogonal matching pursuit stops in ksvd
    float stop_diff; // the difference between iteration errors at which ksvd terminates
    float RGB_stop_diff; // the difference between iteration errors at which ksvd terminates

    void compute_rotation(Eigen::Matrix3f& R, const Eigen::MatrixXf& points);
    void project_points(Eigen::Vector3f& center, const Eigen::Matrix3f& R, Eigen::MatrixXf& points,
                        const Eigen::Matrix<short, Eigen::Dynamic, Eigen::Dynamic>& colors,
                        const std::vector<int>& index_search, int* occupied_indices, int i);
    void project_cloud();
    void compress_depths();
    void compress_colors();
    float compute_rms_error();
public:
    pointcloud_compressor(pointcloud::ConstPtr cloud, float res = 0.1f, int sz = 10, int dict_size = 100,
                          int words_max = 10, float proj_error = 1e-3f, float stop_diff = 1e-5f, int RGB_dict_size = 200,
                          int RGB_words_max = 20, float RGB_proj_error = 1e3f, float RGB_stop_diff = 1e2f);
    void save_compressed(const std::string& name);
};

#endif // POINTCLOUD_COMPRESSOR_H
