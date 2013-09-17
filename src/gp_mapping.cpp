#include "gp_mapping.h"

#include <Eigen/Dense>

using namespace Eigen;

gp_mapping::gp_mapping(pointcloud::ConstPtr cloud, double res, int sz, asynch_visualizer* vis) : gp_registration(cloud, res, sz, vis), min_nbr(100)
{

}

void gp_mapping::add_cloud(pointcloud::ConstPtr other_cloud)
{
    // insert new point cloud
    gp_registration::add_cloud(other_cloud);
    do { // do registration until convergence
        registration_step();
    } while (!registration_done());
    // add point cloud to the map
    insert_into_map();
}

void gp_mapping::insert_into_map()
{
    //octree.setInputCloud(cloud);
    //octree.addPointsFromInputCloud();

    int n = octree.getLeafCount();

    S.resize(n); // increase number of patches
    to_be_added.resize(n);
    W.resize(sz*sz, n);
    rotations.resize(n);
    means.resize(n);

    // radius of the sphere encompassing the voxels
    double radius = sqrt(3.0f)/2.0f*res;

    std::vector<int> index_search; // indices in point cloud of sphere points
    std::vector<float> distances; // same, distances
    Eigen::Matrix3d R;
    Vector3d mid;
    int* occupied_indices = new int[cloud->width*cloud->height](); // maybe save this for later? just one bool

    Vector3d pt;
    point center;
    int i = to_be_added.size();
    leaf_iterator iter(octree);
    while(*++iter) {
        pcl::octree::OctreeKey key = iter.getCurrentOctreeKey();
        octree.generate_voxel_center(center, key);

        gp_leaf* leaf = dynamic_cast<gp_leaf*>(*iter);
        if (leaf == NULL) {
            std::cout << "doesn't work, exiting..." << std::endl;
            exit(0);
        }
        octree.radiusSearch(center, radius, index_search, distances); // search octree
        for (int m = 0; m < index_search.size(); ++m) {
            to_be_added[leaf->gp_index].push_back(cloud->points[index_search[m]].getVector3fMap().cast<double>());
        }
        if (leaf->gp_index == -1) {
            leaf->gp_index = i;
            ++i;
            if (to_be_added[leaf->gp_index].size() >= min_nbr) {
                MatrixXd points(4, index_search.size()); // 4 because of compute rotation
                points.row(3).setOnes();
                for (int m = 0; m < index_search.size(); ++m) {
                    points.block<3, 1>(0, m) = cloud->points[index_search[m]].getVector3fMap().cast<double>();
                }
                compute_rotation(R, points);
                mid = center.getVector3fMap().cast<double>();
                rotations[i] = R;
                means[i] = mid;
            }
        }
        if (to_be_added[leaf->gp_index].size() < min_nbr) {
            // add all points to occupied indices? not sure they will be used though
            // what is the point of occupied indices in gp representation anyways?
            continue;
        }
        project_points(mid, R, index_search, occupied_indices, i);
    }
    octree.remove_just_points();
    delete[] occupied_indices;
}

void gp_mapping::transform_new_points(int i)
{
    ArrayXi count(sz*sz);
    count.setZero(); // not needed anymore, only need weights
    Vector3d loc;
    int ind;
    int x, y;
    for (const Vector3d& glob : to_be_added[i]) {
        if (occupied_indices[index_search[m]]) { // have start point of new indices?
            //continue;
        }
        loc = rotations[i].toRotationMatrix().transpose()*(glob - means[i]); // transforming to the patch coordinate system
        if (loc(1) > res/2.0f || loc(1) < -res/2.0f || loc(2) > res/2.0f || loc(2) < -res/2.0f) {
            continue;
        }
        mn += loc(0);
        occupied_indices[index_search[m]] = 1;
        x = int(double(sz)*(loc(1)/res+0.5f)); // transforming into image patch coordinates
        y = int(double(sz)*(loc(2)/res+0.5f));
        ind = sz*x + y;
        S[i].push_back(loc);
        count(ind) += 1;
    }
    to_be_added[i].clear();
    W.col(i) = W.col(i) || (count > 0);
}

void gp_mapping::project_points(Vector3d& center, const Matrix3d& R,
                                const std::vector<int>& index_search,
                                int* occupied_indices, int i)
{
    ArrayXi count(sz*sz);
    count.setZero(); // not needed anymore, only need weights
    Vector3d loc;
    int ind;
    int x, y;
    double mn = 0;
    for (const Vector3d& glob : to_be_added[i]) {
        if (occupied_indices[index_search[m]]) { // have start point of new indices?
            //continue;
        }
        loc = R.transpose()*(glob - center); // transforming to the patch coordinate system
        if (loc(1) > res/2.0f || loc(1) < -res/2.0f || loc(2) > res/2.0f || loc(2) < -res/2.0f) {
            continue;
        }
        mn += loc(0);
        occupied_indices[index_search[m]] = 1;
        x = int(double(sz)*(loc(1)/res+0.5f)); // transforming into image patch coordinates
        y = int(double(sz)*(loc(2)/res+0.5f));
        ind = sz*x + y;
        S[i].push_back(loc);
        count(ind) += 1;
    }
    to_be_added[i].clear();
    mn /= S[i].size(); // check that mn != 0
    for (Vector3d& loc : S[i]) {
        loc(0) -= mn;
    }
    center += mn*R.col(0);
    W.col(i) = count > 0;
}
