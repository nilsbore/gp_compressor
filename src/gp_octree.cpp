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

void gp_octree::rand_wo_replace(std::vector<int>& ind, int n, int m)
{
    ind.resize(m);
    int r;
    for (int i = 0; i < m; ++i) {
        //do {
            r = rand() % n;
        //}
        //while (std::find(ind.begin(), ind.end(), r) != ind.end());
        ind[i] = r;
    }
}

void gp_octree::update_random_points(double percentage)
{
    int m = int(percentage*double(input_->points.size()));
    std::vector<int> ind;
    rand_wo_replace(ind, input_->points.size(), m);
    for (int i = 0; i < m; i++)
    {
        if (pcl::isFinite (input_->points[ind[i]]))
        {
            // add points to octree
            this->addPointIdx(ind[i]);
        }
    }
}

void gp_octree::remove_just_points()
{
    leaf_iterator iter(*this);
    gp_leaf* leaf;
    while(*++iter) {
        leaf = dynamic_cast<gp_leaf*>(*iter);
        leaf->reset();
    }
}
