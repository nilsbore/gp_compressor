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
#include <dirent.h>

#include "gp_registration.h"
#include "octave_convenience.h"

//using namespace std;

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
    std::string groundtruth = "/home/nbore/Data/rgbd_dataset_freiburg1_room/groundtruth1.txt";
    read_files(files, dirname);
    std::vector<double> sec;
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > pos;
    std::vector<Eigen::Quaterniond, Eigen::aligned_allocator<Eigen::Quaterniond> > rot;
    read_ground_truth(sec, pos, rot, groundtruth);
    /*for (int i = 0; i < 10; ++i) {
        std::cout << std::setprecision(14) << sec[i] << " ";
        std::cout << pos[i].transpose() << " ";
        std::cout << rot[i].x() << " " << rot[i].y() << " " << rot[i].z() << " " << rot[i].w() << std::endl;
    }*/

    Eigen::Matrix3d R;
    Eigen::Vector3d t;
    Eigen::Quaterniond quat;
    Eigen::Vector3d t_gt;
    Eigen::Quaterniond quat_gt;
    int step = 5;
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

        // get ground truth transformation
        double time_first = get_timestamp_from_filename(files[i]);
        double time_second = get_timestamp_from_filename(files[i + step]);
        int ind_first = find_closest_timestamp(sec, time_first);
        int ind_second = find_closest_timestamp(sec, time_second);

        std::cout << "rot.size(): " << rot.size() << std::endl;
        std::cout << "ind_first: " << ind_first << std::endl;
        std::cout << "ind_second: " << ind_second << std::endl;

        quat_gt = rot[ind_first].inverse()*rot[ind_second];
        quat_gt.normalize();
        t_gt = pos[ind_first] - pos[ind_second];

        pcl::PointCloud<pcl::PointXYZ>::Ptr ncenters(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>());
        asynch_visualizer viewer(ncenters, normals);
        gp_registration comp(first_cloud, 0.10f, 5, &viewer);
        viewer.display_cloud = comp.load_compressed();

        std::vector<double> quat_err;
        std::vector<double> t_err;
        std::vector<double> quat_norm;
        std::vector<double> t_norm;
        std::vector<double> likelihoods;
        std::vector<double> color_likelihoods;

        comp.add_cloud(filtered_cloud);
        pthread_t my_viewer_thread;
        pthread_create(&my_viewer_thread, NULL, viewer_thread, &viewer);
        do {
            comp.registration_step();
            comp.get_cloud_transformation(R, t);
            quat = Eigen::Quaterniond(R);
            quat.normalize();
            Eigen::Quaterniond temp = quat.inverse()*quat_gt;
            temp.normalize();
            quat_err.push_back(temp.vec().norm());
            t_err.push_back((t - t_gt).norm());
            quat_norm.push_back(quat.vec().norm());
            t_norm.push_back(t.norm());
            likelihoods.push_back(comp.get_likelihood());
            color_likelihoods.push_back(comp.get_color_likelihood());
        } while (!comp.registration_done());
        pthread_join(my_viewer_thread, NULL);

        octave_convenience oc;
        oc.eval_plot_vector_pair(t_err, t_norm);
        oc.eval_plot_vector_pair(quat_err, quat_norm);
        oc.eval_plot_vector(likelihoods);
        oc.eval_plot_vector(color_likelihoods);

        return 0;
    }

    return 0;
}
