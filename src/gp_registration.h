#ifndef GP_REGISTRATION_H
#define GP_REGISTRATION_H

#include "gp_compressor.h"
#include <Eigen/Dense>

class gp_registration : public gp_compressor
{
public:
    using gp_compressor::point;
    using gp_compressor::pointcloud;
private:

public:
    gp_registration(pointcloud::ConstPtr cloud, float res = 0.1f, int sz = 10);
};

#endif // GP_REGISTRATION_H
