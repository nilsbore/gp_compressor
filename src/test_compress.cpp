#include <iostream>
#include <string>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include "pointcloud_compressor.h"

using namespace std;

int main(int argc, char** argv)
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
    std::string filename = "/home/nbore/Downloads/home_data_ascii/scene11_ascii.pcd";//"../data/office1.pcd";
    if (pcl::io::loadPCDFile<pcl::PointXYZRGB> (filename, *cloud) == -1)
    {
        std::cout << "Couldn't read file " << filename << std::endl;
        return 0;
    }
    pointcloud_compressor comp(cloud, 0.15f, 20, 200, 10, 5e-3f, 1e-5f, 600, 20, 5e4f, 1e3f);
    //pointcloud_compressor comp("../data/office1.pcd", 0.2f, 20, 200, 10, 1e-2f, 5e-5f, 300, 20, 1e4f, 1e1f);
    comp.save_compressed("test");

    return 0;
}
