#include "asynch_visualizer.h"

#include <pcl/visualization/pcl_visualizer.h>

void asynch_visualizer::run_visualizer()
{
    boost::shared_ptr<pcl::visualization::PCLVisualizer>
            viewer (new pcl::visualization::PCLVisualizer("3D Viewer"));
    viewer->setBackgroundColor(0, 0, 0);

    // Coloring and visualizing target cloud (red).
    //pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(display_cloud);
    //viewer->addPointCloud<pcl::PointXYZRGB>(display_cloud, rgb, "cloud");
    //viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
         //                                    1, "cloud");

    //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> point_cloud_color_handler(display_centers, 255, 0, 0);
    //viewer->addPointCloud(display_centers, point_cloud_color_handler, "registered");
    //viewer->addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(display_centers, display_normals, 50, 1e-3f, "normals");

    // Starting visualizer
    viewer->addCoordinateSystem(1.0);
    viewer->initCameraParameters();

    // Wait until visualizer window is closed.
    while (!viewer->wasStopped())
    {
        lock();
        if (has_transformed) {
            viewer->removePointCloud("registered");
            viewer->removePointCloud("normals");
            pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> point_cloud_color_handler(display_centers, 255, 0, 0);
            viewer->addPointCloud(display_centers, point_cloud_color_handler, "registered");
            viewer->addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(display_centers, display_normals, 50, 1e-3f, "normals");
            has_transformed = false;
        }
        if (map_has_transformed) {
            viewer->removePointCloud("cloud");
            pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(display_cloud);
            viewer->addPointCloud<pcl::PointXYZRGB>(display_cloud, rgb, "cloud");
            viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
                                                     1, "cloud");
            map_has_transformed = true;
        }
        unlock();
        viewer->spinOnce(100);
        boost::this_thread::sleep(boost::posix_time::microseconds(100000));
    }
}

void asynch_visualizer::lock()
{
    pthread_mutex_lock(&mutex);
}

void asynch_visualizer::unlock()
{
    pthread_mutex_unlock(&mutex);
}

void* viewer_thread(void* ptr)
{
  ((asynch_visualizer*)ptr)->run_visualizer();
}
