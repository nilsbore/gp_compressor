#ifndef GP_REGISTRATION_H
#define GP_REGISTRATION_H

#include "sparse_gp.h"
#include <Eigen/Dense>

class gp_registration : public sparse_gp
{
public:
    gp_registration();
};

#endif // GP_REGISTRATION_H
