#ifndef GP_LEAF_H
#define GP_LEAF_H

#include <pcl/octree/octree_container.h>

class gp_leaf : public pcl::octree::OctreeContainerDataTVector<int>
{
public:
    int gp_index;
    gp_leaf(const OctreeContainerDataTVector<int>& source);
    gp_leaf();
};

#endif // GP_LEAF_H
