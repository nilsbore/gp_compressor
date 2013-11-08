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

int gp_octree::get_intersected_gps (
    Eigen::Vector3f origin, Eigen::Vector3f direction, std::vector<int> &k_indices,
    int maxVoxelCount) const
{
  pcl::octree::OctreeKey key;
  key.x = key.y = key.z = 0;

  k_indices.clear ();

  // Voxel childIdx remapping
  unsigned char a = 0;
  double minX, minY, minZ, maxX, maxY, maxZ;

  initIntersectedVoxel (origin, direction, minX, minY, minZ, maxX, maxY, maxZ, a);

  if (max (max (minX, minY), minZ) < min (min (maxX, maxY), maxZ))
    return get_intersected_gps_recursive (minX, minY, minZ, maxX, maxY, maxZ, a, this->rootNode_, key,
                                                k_indices, maxVoxelCount);
  return (0);
}

int gp_octree::get_intersected_gps_recursive (
    double minX, double minY, double minZ, double maxX, double maxY, double maxZ, unsigned char a,
    const pcl::octree::OctreeNode* node, const pcl::octree::OctreeKey& key, std::vector<int> &k_indices, int maxVoxelCount) const
{
  if (maxX < 0.0 || maxY < 0.0 || maxZ < 0.0)
    return (0);

  // If leaf node, get voxel center and increment intersection count
  if (node->getNodeType () == pcl::octree::LEAF_NODE)
  {
    const gp_leaf* leaf = dynamic_cast<const gp_leaf*> (node);
    if (leaf == NULL) {
        k_indices.push_back(-1);
    }
    else {
        k_indices.push_back(leaf->gp_index);
    }

    // decode leaf node into k_indices
    //leaf->getData (k_indices);

    return (1);
  }

  // Voxel intersection count for branches children
  int voxelCount = 0;

  // Voxel mid lines
  double midX = 0.5 * (minX + maxX);
  double midY = 0.5 * (minY + maxY);
  double midZ = 0.5 * (minZ + maxZ);

  // First voxel node ray will intersect
  int currNode = getFirstIntersectedNode (minX, minY, minZ, midX, midY, midZ);

  // Child index, node and key
  unsigned char childIdx;
  const pcl::octree::OctreeNode *childNode;
  pcl::octree::OctreeKey childKey;
  do
  {
    if (currNode != 0)
      childIdx = static_cast<unsigned char> (currNode ^ a);
    else
      childIdx = a;

    // childNode == 0 if childNode doesn't exist
    childNode = this->getBranchChildPtr (static_cast<const BranchNode&> (*node), childIdx);
    // Generate new key for current branch voxel
    childKey.x = (key.x << 1) | (!!(childIdx & (1 << 2)));
    childKey.y = (key.y << 1) | (!!(childIdx & (1 << 1)));
    childKey.z = (key.z << 1) | (!!(childIdx & (1 << 0)));

    // Recursively call each intersected child node, selecting the next
    //   node intersected by the ray.  Children that do not intersect will
    //   not be traversed.
    switch (currNode)
    {
      case 0:
        if (childNode)
          voxelCount += get_intersected_gps_recursive (minX, minY, minZ, midX, midY, midZ, a, childNode,
                                                             childKey, k_indices, maxVoxelCount);
        currNode = getNextIntersectedNode (midX, midY, midZ, 4, 2, 1);
        break;

      case 1:
        if (childNode)
          voxelCount += get_intersected_gps_recursive (minX, minY, midZ, midX, midY, maxZ, a, childNode,
                                                             childKey, k_indices, maxVoxelCount);
        currNode = getNextIntersectedNode (midX, midY, maxZ, 5, 3, 8);
        break;

      case 2:
        if (childNode)
          voxelCount += get_intersected_gps_recursive (minX, midY, minZ, midX, maxY, midZ, a, childNode,
                                                             childKey, k_indices, maxVoxelCount);
        currNode = getNextIntersectedNode (midX, maxY, midZ, 6, 8, 3);
        break;

      case 3:
        if (childNode)
          voxelCount += get_intersected_gps_recursive (minX, midY, midZ, midX, maxY, maxZ, a, childNode,
                                                             childKey, k_indices, maxVoxelCount);
        currNode = getNextIntersectedNode (midX, maxY, maxZ, 7, 8, 8);
        break;

      case 4:
        if (childNode)
          voxelCount += get_intersected_gps_recursive (midX, minY, minZ, maxX, midY, midZ, a, childNode,
                                                             childKey, k_indices, maxVoxelCount);
        currNode = getNextIntersectedNode (maxX, midY, midZ, 8, 6, 5);
        break;

      case 5:
        if (childNode)
          voxelCount += get_intersected_gps_recursive (midX, minY, midZ, maxX, midY, maxZ, a, childNode,
                                                             childKey, k_indices, maxVoxelCount);
        currNode = getNextIntersectedNode (maxX, midY, maxZ, 8, 7, 8);
        break;

      case 6:
        if (childNode)
          voxelCount += get_intersected_gps_recursive (midX, midY, minZ, maxX, maxY, midZ, a, childNode,
                                                             childKey, k_indices, maxVoxelCount);
        currNode = getNextIntersectedNode (maxX, maxY, midZ, 8, 8, 7);
        break;

      case 7:
        if (childNode)
          voxelCount += get_intersected_gps_recursive (midX, midY, midZ, maxX, maxY, maxZ, a, childNode,
                                                             childKey, k_indices, maxVoxelCount);
        currNode = 8;
        break;
    }
  } while ((currNode < 8) && (maxVoxelCount <= 0 || voxelCount < maxVoxelCount));

  return (voxelCount);
}
