#include "gp_leaf.h"

gp_leaf::gp_leaf(const OctreeContainerDataTVector<int>& source) : OctreeContainerDataTVector<int>(source), gp_index(-1)
{

}

gp_leaf::gp_leaf() : OctreeContainerDataTVector(), gp_index(-1)
{

}
