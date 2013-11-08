#include <iostream>
#include <string>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/utils.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include "gp_mapping.h"
#include "asynch_visualizer.h"
#include <dirent.h>

using namespace std;

void read_ground_truth(std::vector<double>& sec,
                       std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> >& pos,
                       std::vector<Eigen::Quaterniond, Eigen::aligned_allocator<Eigen::Quaterniond> >& rot,
                       const std::string& file)
{
    std::ifstream fin(file.c_str());

    std::string line;
    double timestamp;
    Eigen::Vector3d vec;
    Eigen::Quaterniond quat;
    int counter = 0;
    while (getline(fin, line)) {
        if (counter < 3) {
            ++counter;
            continue;
        }
        std::istringstream in(line);
        in >> timestamp;
        in >> vec[0]; in >> vec[1]; in >> vec[2];
        in >> quat.x(); in >> quat.y(); in >> quat.z(); in >> quat.w();
        sec.push_back(timestamp);
        pos.push_back(vec);
        rot.push_back(quat);
    }
    fin.close();
}

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

    // read all pointcloud filenames into a vector
    std::vector<std::string> files;
    std::string dirname = "/home/nbore/Data/rgbd_dataset_freiburg1_room/pointclouds";
    read_files(files, dirname);

    // read groundtruth position and timestamps into vectors
    std::vector<double> sec;
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > pos;
    std::vector<Eigen::Quaterniond, Eigen::aligned_allocator<Eigen::Quaterniond> > rot;
    std::string groundtruth = "/home/nbore/Data/rgbd_dataset_freiburg1_room/groundtruth1.txt";
    read_ground_truth(sec, pos, rot, groundtruth);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
    if (pcl::io::loadPCDFile<pcl::PointXYZRGB> (files[0], *cloud) == -1)
    {
        std::cout << "Couldn't read first file " << files[0] << std::endl;
        return 0;
    }
    Eigen::MatrixXd R_init = rot[0].toRotationMatrix();
    Eigen::Vector3d t_init = pos[0];
    //gp_registration::transform_pointcloud(cloud, R_init, t_init);
    gp_mapping comp(cloud, 0.15f, 20, &viewer);
    viewer.display_cloud = comp.load_compressed();
    viewer.create_thread();
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
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
        pcl::VoxelGrid<pcl::PointXYZRGB> sor;
        sor.setInputCloud(other_cloud);
        sor.setLeafSize(0.02f, 0.02f, 0.02f);
        sor.filter(*filtered_cloud);
        //gp_registration::transform_pointcloud(filtered_cloud, R_init, t_init);
        comp.add_cloud(filtered_cloud);
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr display_cloud = comp.load_compressed();
        viewer.lock();
        viewer.display_cloud = display_cloud;
        viewer.map_has_transformed = true;
        viewer.unlock();
        ++i;
    }
    viewer.join_thread();

    return 0;
}
