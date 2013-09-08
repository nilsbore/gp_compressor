#include "gp_octree.h"

gp_octree::gp_octree(const double resolution) : pcl::octree::OctreePointCloudSearch<pcl::PointXYZRGB, gp_leaf>(resolution)
{

}

void gp_octree::generate_voxel_center(pcl::PointXYZRGB& center, const pcl::octree::OctreeKey& key)
{
    genLeafNodeCenterFromOctreeKey(key, center);
}

void gp_octree::update_points()
{
    for (int i = 0; i < input_->points.size (); i++)
    {
        if (pcl::isFinite (input_->points[i]))
        {
            // add points to octree
            this->addPointIdx(i);
        }
    }
}
