#ifndef GP_OCTREE_H
#define GP_OCTREE_H

#include "gp_leaf.h"
#include <pcl/octree/octree_impl.h>
#include <pcl/point_types.h>
#include <vector>

class gp_octree : public pcl::octree::OctreePointCloudSearch<pcl::PointXYZRGB, gp_leaf>
{
private:
    void rand_wo_replace(std::vector<int>& ind, int n, int m);
public:
    typedef pcl::octree::OctreePointCloudSearch<pcl::PointXYZRGB, gp_leaf>::OctreeT::LeafNodeIterator leaf_iterator;
    void generate_voxel_center(pcl::PointXYZRGB& center, const pcl::octree::OctreeKey& key);
    void update_points();
    void update_random_points(double percentage);
    void remove_just_points();
    gp_octree(const double resolution);
};

#endif // GP_OCTREE_H
