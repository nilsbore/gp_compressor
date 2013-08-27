#include <iostream>
#include <string>
#include <pcl/visualization/pcl_visualizer.h>
#include <boost/thread/thread.hpp>
#include "pointcloud_decompressor.h"

using namespace std;

int main(int argc, char** argv)
{
    pointcloud_decompressor decomp;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud = decomp.load_compressed("test");

    boost::shared_ptr<pcl::visualization::PCLVisualizer>
            viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    viewer->setBackgroundColor(0, 0, 0);

    // Coloring and visualizing target cloud (rgb).
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
    viewer->addPointCloud<pcl::PointXYZRGB>(cloud, rgb, "cloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
                                             1, "cloud");

    // Starting visualizer
    viewer->addCoordinateSystem(1.0);
    viewer->initCameraParameters();

    // Wait until visualizer window is closed.
    while (!viewer->wasStopped ())
    {
        viewer->spinOnce(100);
        boost::this_thread::sleep (boost::posix_time::microseconds (100000));
    }

    return 0;
}
