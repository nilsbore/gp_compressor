#include "gp_compressor.h"

#include "gaussian_process.h"

#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/io/pcd_io.h>
#include <stdint.h>
#include <boost/thread/thread.hpp>

using namespace Eigen;

gp_compressor::gp_compressor(pointcloud::ConstPtr ncloud, double res, int sz) :
    cloud(new pointcloud()), octree(res), res(res), sz(sz)
{
    cloud->insert(cloud->begin(), ncloud->begin(), ncloud->end());
    iteration = 0;
    R_cloud.setIdentity();
    t_cloud.setZero();
}

void gp_compressor::save_compressed(const std::string& name)
{
    std::cout << "Size of original point cloud: " << cloud->width*cloud->height << std::endl;
    project_cloud();
    std::cout << "Number of patches: " << S.size() << std::endl;
    train_processes();
}

void gp_compressor::compute_rotation(Matrix3d& R, const MatrixXd& points)
{
    if (points.cols() < 4) {
        R.setIdentity();
        return;
    }
    JacobiSVD<MatrixXd> svd(points.transpose(), ComputeThinV); // kan ta U ist för transpose?
    Vector3d normal = svd.matrixV().block<3, 1>(0, 3);
    normal.normalize();
    Vector3d x(1.0f, 0.0f, 0.0f);
    Vector3d y(0.0f, 1.0f, 0.0f);
    Vector3d z(0.0f, 0.0f, 1.0f);
    if (fabs(normal(0)) > fabs(normal(1)) && fabs(normal(0)) > fabs(normal(2))) { // pointing in x dir
        if (normal(0) < 0) {
            normal *= -1;
        }
        R.col(0) = normal;
        R.col(1) = z.cross(normal);
    }
    else if (fabs(normal(1)) > fabs(normal(0)) && fabs(normal(1)) > fabs(normal(2))) { // pointing in y dir
        if (normal(1) < 0) {
            normal *= -1;
        }
        R.col(0) = normal;
        R.col(1) = x.cross(normal);
    }
    else { // pointing in z dir
        if (normal(2) < 0) {
            normal *= -1;
        }
        R.col(0) = normal;
        R.col(1) = y.cross(normal);
    }
    R.col(1).normalize();
    R.col(2) = normal.cross(R.col(1));
}

void gp_compressor::project_points(Vector3d& center, const Matrix3d& R, MatrixXd& points,
                                           const MatrixXd& colors,
                                           const std::vector<int>& index_search,
                                           int* occupied_indices, int i)
{
    ArrayXi count(sz*sz);
    count.setZero(); // not needed anymore, only need weights
    Vector3d pt;
    Vector3d c; // TEST
    int ind;
    int x, y;
    double mn = 0;
    Vector3d c_mn; // TEST
    c_mn.setZero(); // TEST
    for (int m = 0; m < points.cols(); ++m) {
        if (occupied_indices[index_search[m]]) {
            continue;
        }
        pt = R.transpose()*(points.block<3, 1>(0, m) - center); // transforming to the patch coordinate system
        if (pt(1) > res/2.0f || pt(1) < -res/2.0f || pt(2) > res/2.0f || pt(2) < -res/2.0f) {
            continue;
        }
        mn += pt(0);
        occupied_indices[index_search[m]] = 1;
        x = int(double(sz)*(pt(1)/res+0.5f)); // transforming into image patch coordinates
        y = int(double(sz)*(pt(2)/res+0.5f));
        ind = sz*x + y;
        //double current_count = count(ind);
        c = colors.col(m); // TEST
        c_mn += c; // TEST
        S[i].push_back(point_pair(pt, c));
        to_be_added[i].push_back(point_pair(pt, c));
        //RGB[i].push_back(c); // TEST
        count(ind) += 1;
    }
    mn /= double(to_be_added[i].size()); // check that mn != 0
    c_mn /= double(to_be_added[i].size()); // TEST
    for (point_pair& p : to_be_added[i]) {
        p.first(0) -= mn;
        p.second -= c_mn;
        //std::cout << p(0) << " " << std::endl;
    }
    /*for (Vector3d& p : S[i]) {
        p(0) -= mn;
        //std::cout << p(0) << " " << std::endl;
    }
    for (Vector3d& cc : RGB[i]) { // TEST
        cc -= c_mn;
    }*/
    RGB_means[i] = c_mn;
    center += mn*R.col(0); // should this be minus??
    W.col(i) = count > 0;
}

// do this for to_be_added instead, loop through leaves
void gp_compressor::train_processes()
{
    MatrixXd X;
    VectorXd y;
    MatrixXd C; // TEST
    gps.resize(to_be_added.size());
    RGB_gps.resize(to_be_added.size()); // TEST
    double mean = 0;
    double added = 0;
    int maxm = 0;
    int i;
    leaf_iterator iter(&octree);
    while (*++iter) {
        //gp_leaf* leaf = dynamic_cast<gp_leaf*>(*iter);
        gp_octree::LeafNode* leaf = dynamic_cast<gp_octree::LeafNode*>(*iter);
        if (leaf == NULL) {
            std::cout << "compressor train doesn't work, exiting..." << std::endl;
            exit(0);
        }
        //i = leaf->gp_index;
        i = leaf->getContainer().gp_index;
        if (to_be_added[i].size() == 0) {
            S[i].clear(); // DEBUG FOR MAPPING!
            continue;
        }
        X.resize(to_be_added[i].size(), 2);
        y.resize(to_be_added[i].size());
        C.resize(to_be_added[i].size(), 3); // TEST
        int m = 0;
        for (const point_pair& p : to_be_added[i]) {
            X.row(m) = p.first.tail<2>().transpose();
            C.row(m) = p.second.transpose();
            y(m) = p.first(0);
            ++m;
        }
        /*m = 0; // TEST
        for (const Vector3d& c : RGB[i]) { // TEST
            C.row(m) = c.transpose();
            ++m;
        }*/
        //gps[i].train_parameters(X, y);
        gps[i].add_measurements(X, y);
        RGB_gps[i].add_measurements(X, C);
        mean = (added*mean + gps[i].size())/(added + 1);
        if (gps[i].size() > maxm) {
            maxm = gps[i].size();
        }
        added += 1;
        to_be_added[i].clear();
        S[i].clear(); // DEBUG FOR MAPPING!?
        //RGB[i].clear(); // TEST
    }
    std::cout << "Mean added: " << mean << std::endl;
    std::cout << "Max added: " << maxm << std::endl;
}

void gp_compressor::project_cloud()
{
    octree.setInputCloud(cloud);
    octree.addPointsFromInputCloud();

    int n = octree.getLeafCount();

    S.resize(n);
    //RGB.resize(n); // TEST
    to_be_added.resize(n);
    W.resize(sz*sz, n);
    //RGB.resize(sz*sz, 3*centers.size());
    rotations.resize(n);
    means.resize(n);
    RGB_means.resize(n);
    //RGB_means.resize(centers.size());

    double radius = sqrt(3.0f)/2.0f*res; // radius of the sphere encompassing the voxels

    std::vector<int> index_search;
    std::vector<float> distances;
    Eigen::Matrix3d R;
    Vector3d mid;
    int* occupied_indices = new int[cloud->width*cloud->height]();

    point center;
    int i = 0;
    leaf_iterator iter(&octree);
    while(*++iter) {
        pcl::octree::OctreeKey key = iter.getCurrentOctreeKey();
        octree.generate_voxel_center(center, key);

        gp_octree::LeafNode* leaf = dynamic_cast<gp_octree::LeafNode*>(*iter);
        //gp_leaf* leaf = dynamic_cast<gp_leaf*>(*iter);
        if (leaf == NULL) {
            std::cout << "compressor project doesn't work, exiting..." << std::endl;
            //++i;
            //continue;
            exit(0);
        }
        //leaf->gp_index = i; // too early!
        leaf->getContainer().gp_index = i;

        octree.radiusSearch(center, radius, index_search, distances); // search octree
        //leaf->reset(); // remove references in octree
        if (index_search.size() == 0) { // MAPPING DEBUG
            ++i;
            continue;
        }
        MatrixXd points(4, index_search.size()); // 4 because of compute rotation
        points.row(3).setOnes();
        MatrixXd colors(3, index_search.size());
        for (int m = 0; m < index_search.size(); ++m) {
            points(0, m) = cloud->points[index_search[m]].x;
            points(1, m) = cloud->points[index_search[m]].y;
            points(2, m) = cloud->points[index_search[m]].z;
            colors(0, m) = double(cloud->points[index_search[m]].r);
            colors(1, m) = double(cloud->points[index_search[m]].g);
            colors(2, m) = double(cloud->points[index_search[m]].b);
        }
        compute_rotation(R, points);
        mid = Vector3d(center.x, center.y, center.z);
        project_points(mid, R, points, colors, index_search, occupied_indices, i);
        rotations[i] = R;
        means[i] = mid;
        ++i;
    }
    octree.remove_just_points();
    delete[] occupied_indices;

    free.resize(sz*sz, n); // crashes if put with the others
    free.setZero();
}

void gp_compressor::flatten_colors(Matrix<short, 3, 1>& rtn, const Vector3d& x)
{
    rtn = x.cast<short>();
    for (int i = 0; i < 3; ++i) {
        if (std::isnan(x(i)) || std::isinf(x(i))) {
            rtn(i) = 255;
        }
        else if (rtn(i) < 0) {
            rtn(i) = 0;
        }
        else if (rtn(i) > 255) {
            rtn(i) = 255;
        }
    }
}

gp_compressor::pointcloud::Ptr gp_compressor::load_compressed()
{
    if (iteration == 0) {
        iteration = gps.size();
    }
    int n = gps.size();
    pointcloud::Ptr ncloud(new pointcloud);
    pcl::PointCloud<pcl::PointXYZ>::Ptr ncenters(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    ncloud->width = n*sz*sz;
    ncenters->width = n;
    normals->width = n;
    ncloud->height = 1;
    ncenters->height = 1;
    normals->height = 1;
    ncloud->points.resize(ncloud->width * ncloud->height);
    ncenters->points.resize(ncenters->width * ncenters->height);
    normals->points.resize(normals->width * normals->height);
    Vector3d pt;
    int counter = 0;
    int points;
    int ind;
    int data_points = 0;
    VectorXd f; // DEBUGGING, computing rms error
    MatrixXd X_star;
    VectorXd f_star;
    MatrixXd C_star;
    VectorXd V_star;
    Vector3d c;
    Matrix<short, 3, 1> c_flat;
    double sum_squared_error = 0;
    for (int i = 0; i < n; ++i) {
        if (gps[i].size() == 0) { // S[i].size() MAPPING DEBUG
            continue;
        }

        // DEBUGGING, computing rms error
        /*X_star.resize(S[i].size(), 2);
        f.resize(S[i].size());
        int m = 0;
        for (const Vector3d& p : S[i]) {
            X_star.row(m) = p.tail<2>().transpose().cast<double>();
            f(m) = p(0);
            ++m;
        }
        data_points += S[i].size();
        gps[i].predict_measurements(f_star, X_star, V_star);
        sum_squared_error += (f - f_star).squaredNorm();*/
        // DEBUGGING, computing rms error

        X_star.resize(sz*sz, 2);
        points = 0;
        std::vector<bool> point_free;
        for (int y = 0; y < sz; ++y) { // ROOM FOR SPEEDUP
            for (int x = 0; x < sz; ++x) {
                ind = x*sz + y;
                /*if (!W(ind, i)) {
                    continue;
                }*/
                X_star(points, 0) = res*((double(x) + 0.5f)/double(sz) - 0.5f);
                X_star(points, 1) = res*((double(y) + 0.5f)/double(sz) - 0.5f);
                ++points;
                point_free.push_back(free(ind, i));
            }
        }
        X_star.conservativeResize(points, 2);
        gps[i].predict_measurements(f_star, X_star, V_star);
        RGB_gps[i].predict_measurements(C_star, X_star, V_star);
        for (int m = 0; m < points; ++m) {
            pt(0) = f_star(m);
            pt(1) = X_star(m, 0); // both at the same time
            pt(2) = X_star(m, 1);
            pt = rotations[i].toRotationMatrix()*pt + means[i];
            //std::cout << pt.transpose() << std::endl;
            ncloud->at(counter).x = pt(0);
            ncloud->at(counter).y = pt(1);
            ncloud->at(counter).z = pt(2);
            //int col = i % 3;
            /*if (col == 0) {
                ncloud->at(counter).g = 255;
            }
            else if (col == 1) {
                ncloud->at(counter).b = 255;
            }
            else {
                ncloud->at(counter).r = 255;
                ncloud->at(counter).b = 255;
            }*/
            /*if (i < iteration) {
                ncloud->at(counter).g = 255;
            }
            else {
                ncloud->at(counter).b = 255;
            }*/
            /*if (point_free[m]) {
                ncloud->at(counter).b = 255;
            }
            else {
                ncloud->at(counter).g = 255;
            }*/
            c =  C_star.row(m).transpose() + RGB_means[i];
            flatten_colors(c_flat, c);
            ncloud->at(counter).r = c_flat(0);
            ncloud->at(counter).g = c_flat(1);
            ncloud->at(counter).b = c_flat(2);
            ++counter;
        }
        ncenters->at(i).x = means[i](0);
        ncenters->at(i).y = means[i](1);
        ncenters->at(i).z = means[i](2);
        normals->at(i).normal_x = rotations[i].toRotationMatrix()(0, 0);
        normals->at(i).normal_y = rotations[i].toRotationMatrix()(1, 0);
        normals->at(i).normal_z = rotations[i].toRotationMatrix()(2, 0);
    }
    std::cout << "RMS error: " << sqrt(sum_squared_error / double(data_points)) << std::endl;
    ncloud->resize(counter);
    std::cout << "Size of transformed point cloud: " << ncloud->width*ncloud->height << std::endl;
    iteration = gps.size();
    return ncloud;
}
