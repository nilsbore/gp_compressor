#include "pointcloud_decompressor.h"

#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/io/pcd_io.h>
#include <stdint.h>
#include <boost/thread/thread.hpp>

using namespace Eigen;

pointcloud_decompressor::pointcloud_decompressor(bool display) : dictionary_representation(), display(display)
{

}

pointcloud_decompressor::pointcloud::Ptr pointcloud_decompressor::load_compressed(const std::string& name)
{
    read_from_file(name);
    decompress_depths();
    decompress_colors();
    return reproject_cloud();
}

void pointcloud_decompressor::decompress_depths()
{
    for (int i = 0; i < S.cols(); ++i) {
        S.col(i).setZero();
        for (int k = 0; k < number_words[i]; ++k) {
            S.col(i) += X(k, i)*D.col(I(k, i));
        }
    }
}

void pointcloud_decompressor::decompress_colors()
{
    for (int i = 0; i < RGB.cols(); ++i) {
        RGB.col(i).setZero();
        for (int k = 0; k < RGB_number_words[i]; ++k) {
            RGB.col(i) += RGB_X(k, i)*RGB_D.col(RGB_I(k, i));
        }
    }
}

pointcloud_decompressor::pointcloud::Ptr pointcloud_decompressor::reproject_cloud()
{
    int n = S.cols();
    pointcloud::Ptr ncloud(new pointcloud);
    pcl::PointCloud<pcl::PointXYZ>::Ptr ncenters(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    ncloud->width = n*sz*sz;
    ncenters->width = n;
    normals->width = n;
    ncloud->height = 1;
    ncenters->height = 1;
    normals->height = 1;
    ncloud->points.resize(ncloud->width * ncloud->height);
    ncenters->points.resize(ncenters->width * ncenters->height);
    normals->points.resize(normals->width * normals->height);
    Vector3f pt;
    int counter = 0;
    int ind;
    for (int i = 0; i < n; ++i) {
        for (int y = 0; y < sz; ++y) { // ROOM FOR SPEEDUP
            for (int x = 0; x < sz; ++x) {
                ind = x*sz + y;
                if (!W(ind, i)) {
                    continue;
                }
                pt(0) = S(ind, i);
                pt(1) = (float(x) + 0.5f)*res/float(sz) - res/2.0f;
                pt(2) = (float(y) + 0.5f)*res/float(sz) - res/2.0f;
                pt = rotations[i].toRotationMatrix()*pt + means[i];
                ncloud->at(counter).x = pt(0);
                ncloud->at(counter).y = pt(1);
                ncloud->at(counter).z = pt(2);
                if (short(RGB_means[i](0) + RGB(ind, i)) > 255) {
                    ncloud->at(counter).r = 255;
                }
                else if (short(RGB_means[i](0) + RGB(ind, i)) < 0) {
                    ncloud->at(counter).r = 0;
                }
                else {
                    ncloud->at(counter).r = short(RGB_means[i](0) + RGB(ind, i));
                }
                if (short(RGB_means[i](1) + RGB(ind, S.cols() + i)) > 255) {
                    ncloud->at(counter).g = 255;
                }
                else if (short(RGB_means[i](1) + RGB(ind, S.cols() + i)) < 0) {
                    ncloud->at(counter).g = 0;
                }
                else {
                    ncloud->at(counter).g = short(RGB_means[i](1) + RGB(ind, S.cols() + i));
                }
                if (short(RGB_means[i](2) + RGB(ind, 2*S.cols() + i)) > 255) {
                    ncloud->at(counter).b = 255;
                }
                else if (short(RGB_means[i](2) + RGB(ind, 2*S.cols() + i)) < 0) {
                    ncloud->at(counter).b = 0;
                }
                else {
                    ncloud->at(counter).b = short(RGB_means[i](2) + RGB(ind, 2*S.cols() + i));
                }
                ++counter;
            }
        }
        ncenters->at(i).x = means[i](0);
        ncenters->at(i).y = means[i](1);
        ncenters->at(i).z = means[i](2);
        normals->at(i).normal_x = rotations[i].toRotationMatrix()(0, 0);
        normals->at(i).normal_y = rotations[i].toRotationMatrix()(1, 0);
        normals->at(i).normal_z = rotations[i].toRotationMatrix()(2, 0);
    }
    ncloud->resize(counter);
    std::cout << "Size of transformed point cloud: " << ncloud->width*ncloud->height << std::endl;
    if (display) {
        display_cloud(ncloud, ncenters, normals);
    }
    return ncloud;
}

void pointcloud_decompressor::display_cloud(pointcloud::Ptr display_cloud,
                                          pcl::PointCloud<pcl::PointXYZ>::Ptr display_centers,
                                          pcl::PointCloud<pcl::Normal>::Ptr display_normals)
{
    boost::shared_ptr<pcl::visualization::PCLVisualizer>
            viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    viewer->setBackgroundColor (0, 0, 0);

    // Coloring and visualizing target cloud (red).
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(display_cloud);
    viewer->addPointCloud<point> (display_cloud, rgb, "cloud");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
                                                    1, "cloud");

    viewer->addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(display_centers, display_normals, 10, 0.05, "normals");

    // Starting visualizer
    viewer->addCoordinateSystem (1.0);
    viewer->initCameraParameters ();

    // Wait until visualizer window is closed.
    while (!viewer->wasStopped ())
    {
        viewer->spinOnce (100);
        boost::this_thread::sleep (boost::posix_time::microseconds (100000));
    }
}
