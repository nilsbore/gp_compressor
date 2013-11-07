#include <iostream>
#include <string>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
//#include <pcl/visualization/range_image_visualizer.h>
#include <pcl/common/utils.h>
#include <pcl/common/transforms.h>
//#include <boost/thread/thread.hpp>
#include "gp_registration.h"
#include "asynch_visualizer.h"

using namespace std;

int main(int argc, char** argv)
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
    //std::string filename = "/home/nbore/Downloads/home_data_ascii/scene11_ascii.pcd";
    //std::string filename = "../data/office1.pcd";
    //std::string filename = "../data/room_scan1.pcd";
    std::string filename = "/home/nbore/Data/rgbd_dataset_freiburg1_room/pointclouds/1305031910.765238.pcd";
    if (pcl::io::loadPCDFile<pcl::PointXYZRGB> (filename, *cloud) == -1)
    {
        std::cout << "Couldn't read file " << filename << std::endl;
        return 0;
    }
    pcl::PointCloud<pcl::PointXYZ>::Ptr ncenters(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>());
    asynch_visualizer viewer(ncenters, normals);
    gp_registration comp(cloud, 0.40f, 30, &viewer);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr other_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
    //filename = "../data/room_scan2.pcd";
    filename = "/home/nbore/Data/rgbd_dataset_freiburg1_room/pointclouds/1305031911.097196.pcd";
    //filename = "/home/nbore/Data/rgbd_dataset_freiburg1_room/pointclouds/1305031914.133245.pcd";
    if (pcl::io::loadPCDFile<pcl::PointXYZRGB> (filename, *other_cloud) == -1)
    {
        std::cout << "Couldn't read file " << filename << std::endl;
        return 0;
    }
    viewer.display_cloud = comp.load_compressed();
    comp.add_cloud(other_cloud);
    //pthread_t my_viewer_thread;
    //pthread_create(&my_viewer_thread, NULL, viewer_thread, &viewer);
    viewer.create_thread();
    do {
        comp.registration_step();
    } while (!comp.registration_done());
    viewer.join_thread();
    //pthread_join(my_viewer_thread, NULL);

    return 0;
}
