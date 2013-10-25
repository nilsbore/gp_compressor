#ifndef GP_OCTREE_H
#define GP_OCTREE_H

#include "gp_leaf.h"
#include <pcl/octree/octree_impl.h>
#include <pcl/point_types.h>
#include <vector>

class gp_octree : public pcl::octree::OctreePointCloudSearch<pcl::PointXYZRGB, gp_leaf>
{
public:
    typedef pcl::octree::OctreePointCloudSearch<pcl::PointXYZRGB, gp_leaf> super;
private:
    void rand_wo_replace(std::vector<int>& ind, int n, int m);
    int get_intersected_gps_recursive (
        double minX, double minY, double minZ, double maxX, double maxY, double maxZ, unsigned char a,
        const pcl::octree::OctreeNode* node, const pcl::octree::OctreeKey& key, std::vector<int> &k_indices, int maxVoxelCount) const;
public:
    typedef pcl::octree::OctreePointCloudSearch<pcl::PointXYZRGB, gp_leaf>::OctreeT::LeafNodeIterator leaf_iterator;
    void generate_voxel_center(pcl::PointXYZRGB& center, const pcl::octree::OctreeKey& key);
    void update_points();
    void update_random_points(double percentage);
    void remove_just_points();
    int get_intersected_gps (
        Eigen::Vector3f origin, Eigen::Vector3f direction, std::vector<int> &k_indices,
        int maxVoxelCount = 0) const;
    gp_octree(const double resolution);
};

#endif // GP_OCTREE_H
