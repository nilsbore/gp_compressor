#ifndef GP_LEAF_H
#define GP_LEAF_H

#include <cassert>
#include <pcl/octree/octree_container.h>

class gp_leaf : public pcl::octree::OctreeContainerPointIndices
{
public:
    int gp_index;
    gp_leaf(const OctreeContainerPointIndices& source);
    gp_leaf();
};

#endif // GP_LEAF_H
