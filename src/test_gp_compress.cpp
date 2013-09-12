#include <iostream>
#include <string>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <boost/thread/thread.hpp>
#include "gp_compressor.h"

using namespace std;

int main(int argc, char** argv)
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
    std::string filename = "/home/nbore/Downloads/home_data_ascii/scene11_ascii.pcd";
    //std::string filename = "../data/office1.pcd";
    if (pcl::io::loadPCDFile<pcl::PointXYZRGB> (filename, *cloud) == -1)
    {
        std::cout << "Couldn't read file " << filename << std::endl;
        return 0;
    }
    gp_compressor comp(cloud, 0.15f, 20);
    //pointcloud_compressor comp("../data/office1.pcd", 0.2f, 20, 200, 10, 1e-2f, 5e-5f, 300, 20, 1e4f, 1e1f);
    comp.save_compressed("test");

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr display_cloud = comp.load_compressed();

    boost::shared_ptr<pcl::visualization::PCLVisualizer>
            viewer(new pcl::visualization::PCLVisualizer ("3D Viewer"));
    viewer->setBackgroundColor(0, 0, 0);

    // Coloring and visualizing target cloud (rgb).
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(display_cloud);
    viewer->addPointCloud<pcl::PointXYZRGB>(display_cloud, rgb, "cloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
                                             1, "cloud");

    // Starting visualizer
    viewer->addCoordinateSystem(1.0);
    viewer->initCameraParameters();

    // Wait until visualizer window is closed.
    while (!viewer->wasStopped())
    {
        viewer->spinOnce(100);
        boost::this_thread::sleep(boost::posix_time::microseconds (100000));
    }

    return 0;
}
