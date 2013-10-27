#include <iostream>
#include <string>
#include <fstream>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/utils.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
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
    int counter;
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

int find_closest_timestamp(const std::vector<double>& times, double timestamp)
{
    for (int i = 0; i < times.size(); ++i) {
        if (times[i] > timestamp) {
            if (i == 0) {
                return 0;
            }
            else {
                return i - 1;
            }
        }
    }
    return times.size() - 1;
}

double get_timestamp_from_filename(const std::string& file)
{
    std::string number = file.substr(file.size() - 21, 17);
    std::stringstream ss(number);
    double rtn;
    ss >> rtn;
    return rtn;
}

int main(int argc, char** argv)
{
    std::vector<std::string> files;
    std::string dirname = "/home/nbore/Data/rgbd_dataset_freiburg1_room/pointclouds";
    std::string groundtruth = "/home/nbore/Data/rgbd_dataset_freiburg1_room/groundtruth.txt";
    read_files(files, dirname);
    std::vector<double> sec;
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > pos;
    std::vector<Eigen::Quaterniond, Eigen::aligned_allocator<Eigen::Quaterniond> > rot;
    read_ground_truth(sec, pos, rot, groundtruth);
    for (int i = 0; i < 10; ++i) {
        std::cout << std::setprecision(14) << sec[i] << " ";
        std::cout << pos[i].transpose() << " ";
        std::cout << rot[i].x() << " " << rot[i].y() << " " << rot[i].z() << " " << rot[i].w() << std::endl;
    }

    Eigen::Matrix3d R;
    Eigen::Vector3d t;
    Eigen::Quaterniond quat;
    Eigen::Vector3d t_gt;
    Eigen::Quaterniond quat_gt;
    int step = 2;
    for (int i = 0; i + step < files.size(); ++i) {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr first_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
        if (pcl::io::loadPCDFile<pcl::PointXYZRGB> (files[i], *first_cloud) == -1)
        {
            std::cout << "Couldn't read file " << files[i] << std::endl;
            return 0;
        }
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr second_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
        if (pcl::io::loadPCDFile<pcl::PointXYZRGB> (files[i + step], *second_cloud) == -1)
        {
            std::cout << "Couldn't read file " << files[i + step] << std::endl;
            return 0;
        }
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
        pcl::VoxelGrid<pcl::PointXYZRGB> sor;
        sor.setInputCloud(second_cloud);
        sor.setLeafSize(0.01f, 0.01f, 0.01f);
        sor.filter(*filtered_cloud);

        gp_mapping comp(first_cloud, 0.20f, 15);
        comp.add_cloud(filtered_cloud);
        comp.get_cloud_transformation(R, t);
        quat = Eigen::Quaterniond(R);
        quat.normalize();

        double time_first = get_timestamp_from_filename(files[i]);
        double time_second = get_timestamp_from_filename(files[i + step]);
        int ind_first = find_closest_timestamp(sec, time_first);
        int ind_second = find_closest_timestamp(sec, time_second);

        quat_gt = rot[ind_first].inverse()*rot[ind_second];
        quat_gt.normalize();
        t_gt = pos[ind_second] - pos[ind_first];

        std::cout << quat.x() << " " << quat.y() << " " << quat.z() << " " << quat.w() << std::endl;
        std::cout << quat_gt.x() << " " << quat_gt.y() << " " << quat_gt.z() << " " << quat_gt.w() << std::endl;

        std::cout << t.transpose() << std::endl;
        std::cout << t_gt.transpose() << std::endl;

        exit(0);
    }

    return 0;
}
