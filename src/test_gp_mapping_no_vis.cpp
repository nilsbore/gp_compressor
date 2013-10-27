#include <iostream>
#include <string>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/utils.h>
#include <pcl/common/transforms.h>
#include <boost/thread/thread.hpp>
#include "gp_mapping.h"
#include <dirent.h>

using namespace std;

void read_files(std::vector<std::string>& files, const std::string& dirname)
{
    files.clear();
    DIR* dir = opendir(dirname.c_str());
    if (dir == NULL) {
        std::cout << "Can not read directory." << std::endl;
        exit(0);
    }
    struct dirent* ent;
    while ((ent = readdir(dir)) != NULL) {
        std::string entry(ent->d_name);
        if (entry.length() < 10) {
            continue;
        }
        files.push_back(dirname + "/" + entry);
    }
    closedir(dir);
    std::sort(files.begin(), files.end()); // sort times
}

int main(int argc, char** argv)
{
    std::vector<std::string> files;
    std::string dirname = "/home/nbore/Data/rgbd_dataset_freiburg1_room/pointclouds";
    read_files(files, dirname);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
    if (pcl::io::loadPCDFile<pcl::PointXYZRGB> (files[0], *cloud) == -1)
    {
        std::cout << "Couldn't read first file " << files[0] << std::endl;
        return 0;
    }
    gp_mapping comp(cloud, 0.20f, 15);
    int i = 0;
    for (const std::string& file : files) {
        if (i == 0) {
            ++i;
            continue;
        }
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr other_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
        if (pcl::io::loadPCDFile<pcl::PointXYZRGB> (file, *other_cloud) == -1)
        {
            std::cout << "Couldn't read file " << file << std::endl;
            return 0;
        }
        //comp.transform_pointcloud(other_cloud, R, t);
        comp.add_cloud(other_cloud);
        //comp.get_cloud_transformation(R, t);

        if (i % 40 == 0) {
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr display_cloud = comp.load_compressed();
            pcl::io::savePCDFileBinary("/home/nbore/Workspace/gp_compressor/test.pcd", *display_cloud);
        }
        ++i;
    }
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr display_cloud = comp.load_compressed();
    pcl::io::savePCDFileBinary("/home/nbore/Workspace/gp_compressor/test.pcd", *display_cloud);

    return 0;
}
