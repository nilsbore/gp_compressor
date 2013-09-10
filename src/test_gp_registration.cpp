#include <iostream>
#include <string>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
//#include <pcl/visualization/range_image_visualizer.h>
#include <pcl/common/utils.h>
#include <pcl/common/transforms.h>
#include <boost/thread/thread.hpp>
#include "gp_registration.h"

using namespace std;

void display_cloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr display_cloud,
                   pcl::PointCloud<pcl::PointXYZ>::Ptr display_centers,
                   pcl::PointCloud<pcl::Normal>::Ptr display_normals)
{
    boost::shared_ptr<pcl::visualization::PCLVisualizer>
            viewer (new pcl::visualization::PCLVisualizer("3D Viewer"));
    viewer->setBackgroundColor(0, 0, 0);

    // Coloring and visualizing target cloud (red).
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(display_cloud);
    viewer->addPointCloud<pcl::PointXYZRGB>(display_cloud, rgb, "cloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
                                             1, "cloud");

    if (display_centers) {
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> point_cloud_color_handler(display_centers, 255, 0, 0);
        viewer->addPointCloud(display_centers, point_cloud_color_handler, "registered");
        viewer->addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(display_centers, display_normals, 50, 1.0, "normals");
    }

    // Starting visualizer
    viewer->addCoordinateSystem(1.0);
    viewer->initCameraParameters();

    // Wait until visualizer window is closed.
    while (!viewer->wasStopped())
    {
        viewer->spinOnce(100);
        boost::this_thread::sleep(boost::posix_time::microseconds (100000));
    }
}

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
    pcl::PointCloud<pcl::PointXYZ>::Ptr ncenters(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>());
    gp_registration comp(cloud, 0.20f, 10, ncenters, normals);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr other_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
    if (pcl::io::loadPCDFile<pcl::PointXYZRGB> (filename, *other_cloud) == -1)
    {
        std::cout << "Couldn't read file " << filename << std::endl;
        return 0;
    }

    // Set initial alignment estimate found using robot odometry.
    Eigen::AngleAxisf init_rotation(0.1, Eigen::Vector3f::UnitZ ());
    //Eigen::AngleAxisf init_rotation (0.0, Eigen::Vector3f::UnitZ ());
    Eigen::Translation3f init_translation(0.06, 0.00, 0.00);
    //Eigen::Translation3f init_translation (0, 0, 0);
    Eigen::Matrix4f init_guess = (init_translation * init_rotation).matrix();
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::transformPointCloud(*other_cloud, *transformed_cloud, init_guess);

    comp.add_cloud(transformed_cloud);
    //pointcloud_compressor comp("../data/office1.pcd", 0.2f, 20, 200, 10, 1e-2f, 5e-5f, 300, 20, 1e4f, 1e1f);
    //comp.save_compressed("test");

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr dcloud = comp.load_compressed();

    display_cloud(dcloud, ncenters, normals);

    return 0;
}
