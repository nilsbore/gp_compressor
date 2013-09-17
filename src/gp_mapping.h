#ifndef GP_MAPPING_H
#define GP_MAPPING_H

#include "gp_registration.h"

class gp_mapping : public gp_registration
{
private:
    void insert_into_map();
    int min_nbr;
public:
    void add_cloud(pointcloud::ConstPtr other_cloud);
    gp_mapping(pointcloud::ConstPtr cloud, double res = 0.1f, int sz = 10,
               asynch_visualizer* vis = NULL);
};

#endif // GP_MAPPING_H
