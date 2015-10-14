#include "gp_leaf.h"

gp_leaf::gp_leaf(const OctreeContainerPointIndices& source) : OctreeContainerPointIndices(source), gp_index(-1)
{

}

gp_leaf::gp_leaf() : OctreeContainerPointIndices(), gp_index(-1)
{

}
