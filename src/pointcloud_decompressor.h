#ifndef POINTCLOUD_DECOMPRESSOR_H
#define POINTCLOUD_DECOMPRESSOR_H

#include "dictionary_representation.h"
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <string>
#include <vector>

class pointcloud_decompressor : public dictionary_representation
{
public:
    typedef pcl::PointXYZRGB point;
    typedef pcl::PointCloud<point> pointcloud;
private:
    bool display;
    void decompress_depths();
    void decompress_colors();
    pointcloud::Ptr reproject_cloud();
    void display_cloud(pointcloud::Ptr display_cloud,
                       pcl::PointCloud<pcl::PointXYZ>::Ptr display_centers,
                       pcl::PointCloud<pcl::Normal>::Ptr display_normals);
public:
    pointcloud_decompressor(bool display = false);
    pointcloud::Ptr load_compressed(const std::string& name);
};

#endif // POINTCLOUD_DECOMPRESSOR_H
