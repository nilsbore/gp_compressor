
#include <iostream>
#include <string>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/utils.h>
#include <pcl/common/transforms.h>
#include <boost/thread/thread.hpp>
#include "gp_mapping.h"
#include "asynch_visualizer.h"
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
    pcl::PointCloud<pcl::PointXYZ>::Ptr ncenters(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>());
    asynch_visualizer viewer(ncenters, normals);

    std::vector<std::string> files;
    std::string dirname = "/home/nbore/Data/rgbd_dataset_freiburg1_room/pointclouds";
    read_files(files, dirname);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
    if (pcl::io::loadPCDFile<pcl::PointXYZRGB> (files[0], *cloud) == -1)
    {
        std::cout << "Couldn't read first file " << files[0] << std::endl;
        return 0;
    }
    gp_mapping comp(cloud, 0.20f, 15, &viewer);
    viewer.display_cloud = comp.load_compressed();
    //Eigen::Matrix3d R;
    //Eigen::Vector3d t;
    pthread_t my_viewer_thread;
    pthread_create(&my_viewer_thread, NULL, viewer_thread, &viewer);
    int i = 0;
    for (const std::string& file : files) {
        if (i == 0) {
            ++i;
            continue;
        }
        /*if (i % 10 != 0) {
            ++i;
            continue;
        }*/
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr other_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
        if (pcl::io::loadPCDFile<pcl::PointXYZRGB> (file, *other_cloud) == -1)
        {
            std::cout << "Couldn't read file " << file << std::endl;
            return 0;
        }
        //comp.transform_pointcloud(other_cloud, R, t);
        comp.add_cloud(other_cloud);
        //comp.get_cloud_transformation(R, t);
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr display_cloud = comp.load_compressed();
        //comp.transform_pointcloud(other_cloud, R, t);
        viewer.lock();
        //viewer.display_cloud->swap(*display_cloud);
        viewer.display_cloud = display_cloud;
        //viewer.display_other = other_cloud;
        //viewer.other_has_transformed = true;
        viewer.map_has_transformed = true;
        viewer.unlock();
        ++i;
    }
    pthread_join(my_viewer_thread, NULL);

    return 0;
}
