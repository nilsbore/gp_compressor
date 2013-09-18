#include "gp_mapping.h"

using namespace Eigen;

gp_mapping::gp_mapping(pointcloud::ConstPtr cloud, double res, int sz, asynch_visualizer* vis) : gp_registration(cloud, res, sz, vis)
{
    min_nbr = 100;
}

// return new map here for debuggging?
// or just use the old method, make it public
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
    octree.update_points();
    int n = octree.getLeafCount();

    int i = to_be_added.size(); // first new leaf nbr

    S.resize(n); // increase number of patches
    to_be_added.resize(n);
    W.conservativeResize(sz*sz, n);
    rotations.resize(n);
    means.resize(n);

    // radius of the sphere encompassing the voxels
    double radius = sqrt(3.0f)/2.0f*res;

    std::vector<int> index_search; // indices in point cloud of sphere points
    std::vector<float> distances; // same, distances
    Eigen::Matrix3d R;
    Vector3d mid;
    int* occupied_indices = new int[cloud->width*cloud->height](); // maybe save this for later? just one bool

    bool is_new;
    point center;
    leaf_iterator iter(octree);
    while(*++iter) {
        pcl::octree::OctreeKey key = iter.getCurrentOctreeKey();
        octree.generate_voxel_center(center, key);

        gp_leaf* leaf = dynamic_cast<gp_leaf*>(*iter);
        if (leaf == NULL) {
            std::cout << "doesn't work, exiting..." << std::endl;
            exit(0);
        }
        is_new = (leaf->gp_index == -1);
        if (is_new) {
            leaf->gp_index = i;
            ++i;
        }
        octree.radiusSearch(center, radius, index_search, distances); // search octree
        std::cout << "index_search.size(): " << index_search.size() << std::endl;
        for (int m = 0; m < index_search.size(); ++m) {
            to_be_added[leaf->gp_index].push_back(cloud->points[index_search[m]].getVector3fMap().cast<double>());
        }
        std::cout << "to_be_added.size(): " << to_be_added[leaf->gp_index].size() << std::endl;
        if (!is_new && gps[leaf->gp_index].size() > 0) {
            transform_to_old(leaf->gp_index, index_search, occupied_indices); // here we can still transform them, doesn't matter if not enough
            std::cout << "Added to old one" << std::endl;
            continue;
        }
        if (to_be_added[leaf->gp_index].size() < min_nbr) {
            continue;
        }
        MatrixXd points(4, index_search.size()); // 4 because of compute rotation
        points.row(3).setOnes();
        for (int m = 0; m < index_search.size(); ++m) {
            points.block<3, 1>(0, m) = cloud->points[index_search[m]].getVector3fMap().cast<double>();
        }
        compute_rotation(R, points);
        mid = center.getVector3fMap().cast<double>();
        transform_to_new(mid, R, leaf->gp_index, index_search, occupied_indices);
        std::cout << "Added new one" << std::endl;
        rotations[i] = R;
        means[i] = mid;
    }
    octree.remove_just_points();
    delete[] occupied_indices;
    train_processes();
}

void gp_mapping::transform_to_old(int i, const std::vector<int>& index_search,
                                  int* occupied_indices)
{
    ArrayXi count(sz*sz);
    count.setZero(); // not needed anymore, only need weights
    Vector3d loc;
    int ind;
    int x, y;
    int m = 0;
    for (const Vector3d& glob : to_be_added[i]) {
        if (occupied_indices[index_search[m]]) { // have start point of new indices?
            ++m;
            continue;
        }
        loc = rotations[i].toRotationMatrix().transpose()*(glob - means[i]); // transforming to the patch coordinate system
        if (loc(1) > res/2.0f || loc(1) < -res/2.0f || loc(2) > res/2.0f || loc(2) < -res/2.0f) {
            ++m;
            continue;
        }
        occupied_indices[index_search[m]] = 1;
        x = int(double(sz)*(loc(1)/res+0.5f)); // transforming into image patch coordinates
        y = int(double(sz)*(loc(2)/res+0.5f));
        ind = sz*x + y;
        S[i].push_back(loc);
        count(ind) += 1;
        ++m;
    }
    to_be_added[i].clear();
    W.col(i) = (W.col(i).cast<int>() + count) > 0; // take at look at this, same as ||?
}

void gp_mapping::transform_to_new(Vector3d& center, const Matrix3d& R, int i,
                                  const std::vector<int>& index_search,
                                  int* occupied_indices)
{
    ArrayXi count(sz*sz);
    count.setZero(); // not needed anymore, only need weights
    Vector3d loc;
    int ind;
    int x, y;
    double mn = 0;
    int m = 0;
    for (const Vector3d& glob : to_be_added[i]) {
        if (occupied_indices[index_search[m]]) { // have start point of new indices?
            ++m;
            continue;
        }
        loc = R.transpose()*(glob - center); // transforming to the patch coordinate system
        if (loc(1) > res/2.0f || loc(1) < -res/2.0f || loc(2) > res/2.0f || loc(2) < -res/2.0f) {
            ++m;
            continue;
        }
        mn += loc(0);
        occupied_indices[index_search[m]] = 1;
        x = int(double(sz)*(loc(1)/res+0.5f)); // transforming into image patch coordinates
        y = int(double(sz)*(loc(2)/res+0.5f));
        ind = sz*x + y;
        S[i].push_back(loc);
        count(ind) += 1;
        ++m;
    }
    to_be_added[i].clear();
    mn /= S[i].size(); // check that mn != 0
    for (Vector3d& loc : S[i]) {
        loc(0) -= mn;
    }
    center += mn*R.col(0);
    W.col(i) = count > 0;
}

void gp_mapping::train_processes()
{
    std::cout << "Calling new train processes" << std::endl;
    MatrixXd X;
    VectorXd y;
    gps.resize(octree.getLeafCount()); // new ones
    int i;
    leaf_iterator iter(octree);
    while (*++iter) { // why iterate over leaves?
        gp_leaf* leaf = dynamic_cast<gp_leaf*>(*iter);
        if (leaf == NULL) {
            std::cout << "doesn't work, exiting..." << std::endl;
            exit(0);
        }
        i = leaf->gp_index;
        if (S[i].size() == 0) {
            std::cout << "Skipping for too few points" << std::endl;
            std::cout << "S[i].size(): " << S[i].size() << std::endl;
            continue;
        }
        X.resize(S[i].size(), 2);
        y.resize(S[i].size());
        int m = 0;
        for (const Vector3d& p : S[i]) {
            X.row(m) = p.tail<2>().transpose().cast<double>();
            y(m) = p(0);
            ++m;
        }
        gps[i].add_measurements(X, y);
        S[i].clear();
    }
}
