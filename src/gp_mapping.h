#ifndef GP_MAPPING_H
#define GP_MAPPING_H

#include "gp_registration.h"

#include <Eigen/Dense>
#include <vector>

class gp_mapping : public gp_registration
{
private:
    int min_nbr;
    void insert_into_map();
    void transform_to_old(int i, const std::vector<int>& index_search,
                          int* occupied_indices);
    void transform_to_new(Eigen::Vector3d& center, const Eigen::Matrix3d& R, int i,
                          const std::vector<int>& index_search, int* occupied_indices);
    void train_processes(); // could probably be replaced in old one instead
public:
    void add_cloud(pointcloud::ConstPtr other_cloud);
    gp_mapping(pointcloud::ConstPtr cloud, double res = 0.1f, int sz = 10,
               asynch_visualizer* vis = NULL);
};

#endif // GP_MAPPING_H
