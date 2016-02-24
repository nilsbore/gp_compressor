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
    //int n = gps.size();
    // insert new point cloud
    gp_registration::add_cloud(other_cloud);
    transform_pointcloud(cloud, R_cloud, t_cloud);
    do { // do registration until convergence
        registration_step();
    } while (!registration_done());
    // add point cloud to the map
    if (step_nbr < max_steps) {
        insert_into_map();
    }
    else {
        cloud->clear();
    }

    /*S.resize(n); // increase number of patches
    to_be_added.resize(n);
    W.conservativeResize(sz*sz, n);
    rotations.resize(n);
    means.resize(n);
    gps.resize(n);*/
}

void gp_mapping::insert_into_map()
{
    octree.update_points();
    int n = octree.getLeafCount();

    if (n == 0) {
        octree.remove_just_points();
        return;
    }

    int i = to_be_added.size(); // first new leaf nbr

    S.resize(n); // increase number of patches
    //RGB.resize(n);
    to_be_added.resize(n);
    W.conservativeResize(sz*sz, n);
    free.conservativeResize(sz*sz, n);
    rotations.resize(n);
    means.resize(n);
    RGB_means.resize(n);

    // radius of the sphere encompassing the voxels
    double radius = sqrt(3.0f)/2.0f*res;

    std::vector<int> index_search; // indices in point cloud of sphere points
    std::vector<float> distances; // same, distances
    Eigen::Matrix3d R;
    Vector3d mid;
    int* occupied_indices = new int[cloud->width*cloud->height](); // maybe save this for later? just one bool
    int* gp_indices = new int[cloud->size()]; // can be used instead of occupied_indices
    for (int i = 0; i < cloud->size(); ++i) {
        gp_indices[i] = -1;
    }

    bool is_new;
    point center;
    point p;
    Vector3d c;
    leaf_iterator iter(&octree);
    while(*++iter) {
        pcl::octree::OctreeKey key = iter.getCurrentOctreeKey();
        octree.generate_voxel_center(center, key);

        //gp_leaf* leaf = dynamic_cast<gp_leaf*>(*iter);
        gp_octree::LeafNode* leaf = dynamic_cast<gp_octree::LeafNode*>(*iter);
        if (leaf == NULL) {
            std::cout << "inserting doesn't work, exiting..." << std::endl;
            exit(0);
        }
        //is_new = (leaf->gp_index == -1);
        is_new = (leaf->getContainer().gp_index == -1);
        if (is_new) {
            //leaf->gp_index = i;
            leaf->getContainer().gp_index = i;
            std::cout << "new number" << std::endl;
            std::cout << "S[i].size: " << S[i].size() << std::endl;
            std::cout << "to_be_added[i].size: " << to_be_added[i].size() << std::endl;
            ++i;
        }
        octree.radiusSearch(center, radius, index_search, distances); // search octree
        std::cout << "index_search.size(): " << index_search.size() << std::endl;
        for (int m = 0; m < index_search.size(); ++m) {
            p = cloud->points[index_search[m]];
            c(0) = p.r;
            c(1) = p.g;
            c(2) = p.b;
            //to_be_added[leaf->gp_index].push_back(
            //            point_pair(p.getVector3fMap().cast<double>(), c));
            std::cout << "index: " << leaf->getContainer().gp_index << std::endl;
            std::cout << "size: " << to_be_added.size() << std::endl;
            to_be_added[leaf->getContainer().gp_index].push_back(
                        point_pair(p.getVector3fMap().cast<double>(), c));

            //RGB[leaf->gp_index].push_back(c);
        }
        //std::cout << "to_be_added.size(): " << to_be_added[leaf->gp_index].size() << std::endl;
        std::cout << "to_be_added.size(): " << to_be_added[leaf->getContainer().gp_index].size() << std::endl;
        //if (!is_new && gps[leaf->gp_index].size() > 0) {
        if (!is_new && gps[leaf->getContainer().gp_index].size() > 0) {
            //transform_to_old(leaf->gp_index, index_search, occupied_indices, gp_indices); // here we can still transform them, doesn't matter if not enough
            transform_to_old(leaf->getContainer().gp_index, index_search, occupied_indices, gp_indices); // here we can still transform them, doesn't matter if not enough
            std::cout << "Added to old one" << std::endl;
            continue;
        }
        //if (!is_new && gps[leaf->gp_index].size() == 0) { // DEBUG
        if (!is_new && gps[leaf->getContainer().gp_index].size() == 0) { // DEBUG
            //continue; // this adds more strange processes for some reason
        }
        //if (to_be_added[leaf->gp_index].size() < min_nbr) {
        if (to_be_added[leaf->getContainer().gp_index].size() < min_nbr) {
            continue;
        }
        MatrixXd points(4, index_search.size()); // 4 because of compute rotation
        points.row(3).setOnes();
        for (int m = 0; m < index_search.size(); ++m) {
            points.block<3, 1>(0, m) = cloud->points[index_search[m]].getVector3fMap().cast<double>();
        }
        compute_rotation(R, points);
        mid = center.getVector3fMap().cast<double>();
        //transform_to_new(mid, R, leaf->gp_index, index_search, occupied_indices, gp_indices);
        transform_to_new(mid, R, leaf->getContainer().gp_index, index_search, occupied_indices, gp_indices);
        std::cout << "Added new one" << std::endl;
        //rotations[leaf->gp_index] = R;
        rotations[leaf->getContainer().gp_index] = R;
        //means[leaf->gp_index] = mid;
        means[leaf->getContainer().gp_index] = mid;
    }
    octree.remove_just_points();
    delete[] occupied_indices;

    // do free space stuff
    train_classification(gp_indices);
    delete[] gp_indices;

    train_processes();
}

void gp_mapping::train_classification(int* gp_indices)
{
    // save information about free space, probably subsampled
    Vector3f measurement;
    Vector3f delta;
    Vector3d intersection;
    Vector3d loc;
    Matrix3d R;
    Vector3d mid;
    int x, y;
    int ind;
    int m;
    std::vector<int> intersected_indices;
    for (int i = 0; i < cloud->size(); ++i) { // only add if gps already exists?
        measurement = cloud->points[i].getVector3fMap();
        delta = measurement - t_cloud.cast<float>(); // assumes original cloud has camera at (0, 0, 0)
        octree.get_intersected_gps(t_cloud.cast<float>(), delta, intersected_indices);
        if (intersected_indices.size() == 0) {
            continue;
        }
        bool reached_gp = false;
        for (int j = intersected_indices.size() - 1; j >= 0; --j) {
            if (intersected_indices[j] == -1 || gp_indices[i] == -1) {
                continue;
            }
            m = intersected_indices[j];
            if (gps[m].size() == 0) {
                continue;
            }
            if (!reached_gp) {
                if (m == gp_indices[i]) {
                    reached_gp = true;
                }
                else {
                    continue;
                }
            }
            R = rotations[m].toRotationMatrix();
            Vector3d normal = R.col(0);
            mid = means[m];
            double d = normal.dot(mid - t_cloud) / (normal.dot(delta.cast<double>()));
            intersection = t_cloud + d*delta.cast<double>();
            loc = R.transpose()*(intersection - mid);
            if (loc(1) > res/2.0f || loc(1) < -res/2.0f || loc(2) > res/2.0f || loc(2) < -res/2.0f) {
                continue;
            }
            x = int(double(sz)*(loc(1)/res+0.5f)); // transforming into image patch coordinates
            y = int(double(sz)*(loc(2)/res+0.5f));
            ind = sz*x + y;
            if (m == gp_indices[i]) {
                free(ind, m) = false;
            }
            else {
                free(ind, m) = true;
            }
        }
    }
}

void gp_mapping::transform_to_old(int i, const std::vector<int>& index_search,
                                  int* occupied_indices, int* gp_indices)
{
    ArrayXi count(sz*sz);
    count.setZero(); // not needed anymore, only need weights
    Vector3d loc;
    int ind;
    int x, y;
    int m = 0;
    for (const point_pair& glob : to_be_added[i]) {
        if (occupied_indices[index_search[m]]) { // have start point of new indices?
            ++m;
            continue;
        }
        loc = rotations[i].toRotationMatrix().transpose()*(glob.first - means[i]); // transforming to the patch coordinate system
        if (loc(1) > res/2.0f || loc(1) < -res/2.0f || loc(2) > res/2.0f || loc(2) < -res/2.0f) {
            ++m;
            continue;
        }
        occupied_indices[index_search[m]] = 1;
        gp_indices[index_search[m]] = i;
        x = int(double(sz)*(loc(1)/res+0.5f)); // transforming into image patch coordinates
        y = int(double(sz)*(loc(2)/res+0.5f));
        ind = sz*x + y;
        S[i].push_back(point_pair(loc, glob.second - RGB_means[i]));
        count(ind) += 1;
        ++m;
    }
    to_be_added[i].clear();
    W.col(i) = (W.col(i).cast<int>() + count) > 0; // take at look at this, same as ||?
}

void gp_mapping::transform_to_new(Vector3d& center, const Matrix3d& R, int i,
                                  const std::vector<int>& index_search,
                                  int* occupied_indices, int* gp_indices)
{
    ArrayXi count(sz*sz);
    count.setZero(); // not needed anymore, only need weights
    Vector3d loc;
    Vector3d c;
    int ind;
    int x, y;
    double mn = 0;
    Vector3d c_mn;
    c_mn.setZero();
    int last_inds;
    int m = 0;
    for (const point_pair& glob : to_be_added[i]) {
        last_inds = index_search.size() - to_be_added[i].size() + m;
        if (last_inds < 0 || occupied_indices[index_search[last_inds]]) { // have start point of new indices?
            ++m;
            continue;
        }
        loc = R.transpose()*(glob.first - center); // transforming to the patch coordinate system
        if (loc(1) > res/2.0f || loc(1) < -res/2.0f || loc(2) > res/2.0f || loc(2) < -res/2.0f) {
            ++m;
            continue;
        }
        mn += loc(0);
        occupied_indices[index_search[last_inds]] = 1;
        gp_indices[index_search[last_inds]] = i;
        x = int(double(sz)*(loc(1)/res+0.5f)); // transforming into image patch coordinates
        y = int(double(sz)*(loc(2)/res+0.5f));
        ind = sz*x + y;
        c_mn += glob.second;
        S[i].push_back(point_pair(loc, glob.second));
        count(ind) += 1;
        ++m;
    }
    to_be_added[i].clear();
    mn /= double(S[i].size()); // check that mn != 0
    c_mn /= double(S[i].size());
    for (point_pair& loc : S[i]) {
        loc.first(0) -= mn;
        loc.second -= c_mn;
    }
    center += mn*R.col(0);
    W.col(i) = count > 0;
}

void gp_mapping::train_processes()
{
    std::cout << "Calling new train processes" << std::endl;
    MatrixXd X;
    VectorXd y;
    MatrixXd C;
    gps.resize(octree.getLeafCount()); // new ones
    RGB_gps.resize(octree.getLeafCount());
    int i;
    leaf_iterator iter(&octree);
    while (*++iter) { // why iterate over leaves?
        //gp_leaf* leaf = dynamic_cast<gp_leaf*>(*iter);
        gp_octree::LeafNode* leaf = dynamic_cast<gp_octree::LeafNode*>(*iter);
        if (leaf == NULL) {
            std::cout << "training doesn't work, exiting..." << std::endl;
            exit(0);
        }
        //i = leaf->gp_index;
        i = leaf->getContainer().gp_index;
        if (S[i].size() == 0) {
            std::cout << "Skipping for too few points" << std::endl;
            std::cout << "S[i].size(): " << S[i].size() << std::endl;
            continue;
        }
        X.resize(S[i].size(), 2);
        y.resize(S[i].size());
        /*if (S[i].size() != RGB[i].size()) {
            std::cout << "S[i].size(): " << S[i].size() << std::endl;
            std::cout << "RGB[i].size(): " << RGB[i].size() << std::endl;
            std::cout << "S and RGB different size, stopping!" << std::endl;
            exit(0);
        }*/
        C.resize(S[i].size(), 3);
        int m = 0;
        for (const point_pair& p : S[i]) {
            X.row(m) = p.first.tail<2>().transpose();
            C.row(m) = p.second.transpose();
            y(m) = p.first(0);
            ++m;
        }
        /*m = 0;
        for (const Vector3d& c : RGB[i]) {
            C.row(m) = c.transpose();
            ++m;
        }*/
        gps[i].add_measurements(X, y);
        RGB_gps[i].add_measurements(X, C);
        S[i].clear();
        //RGB[i].clear();
    }
}
