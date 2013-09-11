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

class asynch_visualizer {
private:
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr display_cloud;
    pcl::PointCloud<pcl::PointXYZ>::Ptr display_centers;
    pcl::PointCloud<pcl::Normal>::Ptr display_normals;
public:
    bool has_transformed;
    void run_visualizer();
    asynch_visualizer(pcl::PointCloud<pcl::PointXYZRGB>::Ptr display_cloud,
                      pcl::PointCloud<pcl::PointXYZ>::Ptr display_centers,
                      pcl::PointCloud<pcl::Normal>::Ptr display_normals) :
        display_cloud(display_cloud), display_centers(display_centers), display_normals(display_normals)
    {
        has_transformed = true;
    }
};

void asynch_visualizer::run_visualizer()
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
        viewer->addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(display_centers, display_normals, 50, 1e-2f, "normals");
    }

    // Starting visualizer
    viewer->addCoordinateSystem(1.0);
    viewer->initCameraParameters();

    // Wait until visualizer window is closed.
    while (!viewer->wasStopped())
    {
        if (display_centers && has_transformed) {
            has_transformed = false;
            viewer->removePointCloud("registered");
            viewer->removePointCloud("normals");
            pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> point_cloud_color_handler(display_centers, 255, 0, 0);
            viewer->addPointCloud(display_centers, point_cloud_color_handler, "registered");
            viewer->addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(display_centers, display_normals, 50, 1e-2f, "normals");
        }
        viewer->spinOnce(100);
        boost::this_thread::sleep(boost::posix_time::microseconds (100000));
    }
}

void* viewer_thread(void* ptr)
{
  ((asynch_visualizer*)ptr)->run_visualizer();
}

int main(int argc, char** argv)
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
    //std::string filename = "/home/nbore/Downloads/home_data_ascii/scene11_ascii.pcd";
    std::string filename = "../data/office1.pcd";
    //std::string filename = "../data/room_scan1.pcd";
    if (pcl::io::loadPCDFile<pcl::PointXYZRGB> (filename, *cloud) == -1)
    {
        std::cout << "Couldn't read file " << filename << std::endl;
        return 0;
    }
    pcl::PointCloud<pcl::PointXYZ>::Ptr ncenters(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>());
    gp_registration comp(cloud, 0.70f, 30, ncenters, normals);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr other_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
    //filename = "../data/room_scan2.pcd";
    if (pcl::io::loadPCDFile<pcl::PointXYZRGB> (filename, *other_cloud) == -1)
    {
        std::cout << "Couldn't read file " << filename << std::endl;
        return 0;
    }

    // Set initial alignment estimate found using robot odometry.
    Eigen::AngleAxisf init_rotation(0.6, Eigen::Vector3f::UnitY());
    //Eigen::AngleAxisf init_rotation (0.0, Eigen::Vector3f::UnitZ ());
    Eigen::Translation3f init_translation(0.20, 0.10, 0.00);
    //Eigen::Translation3f init_translation (0, 0, 0);
    Eigen::Matrix4f init_guess = (init_translation * init_rotation).matrix();
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::transformPointCloud(*other_cloud, *transformed_cloud, init_guess);

    //pointcloud_compressor comp("../data/office1.pcd", 0.2f, 20, 200, 10, 1e-2f, 5e-5f, 300, 20, 1e4f, 1e1f);
    //comp.save_compressed("test");

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr dcloud = comp.load_compressed();
    comp.add_cloud(transformed_cloud);
    asynch_visualizer viewer(dcloud, ncenters, normals);
    pthread_t my_viewer_thread;
    pthread_create(&my_viewer_thread, NULL, viewer_thread, &viewer);
    do {
        comp.registration_step();
        viewer.has_transformed = true;
    } while (!comp.registration_done());
    pthread_join(my_viewer_thread, NULL);

    return 0;
}
