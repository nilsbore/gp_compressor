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
}

void gp_compressor::save_compressed(const std::string& name)
{
    std::cout << "Size of original point cloud: " << cloud->width*cloud->height << std::endl;
    project_cloud();
    std::cout << "Number of patches: " << S.size() << std::endl;
    //compress_depths();
    //compress_colors();
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
                                           const Matrix<short, Dynamic, Dynamic>& colors,
                                           const std::vector<int>& index_search,
                                           int* occupied_indices, int i)
{
    ArrayXi count(sz*sz);
    count.setZero(); // not needed anymore, only need weights
    Vector3d pt;
    Matrix<short, 3, 1> c;
    int ind;
    int x, y;
    double mn = 0;
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
        S[i].push_back(pt);
        to_be_added[i].push_back(pt);
        c = colors.col(m);
        /*for (int n = 0; n < 3; ++n) {
            RGB(ind, n*S.cols() + i) = (current_count*RGB(ind, n*S.cols() + i) + double(c(n))) / (current_count + 1);
        }*/
        count(ind) += 1;
    }
    mn /= to_be_added[i].size(); // check that mn != 0
    for (Vector3d& p : to_be_added[i]) {
        p(0) -= mn;
        //std::cout << p(0) << " " << std::endl;
    }
    for (Vector3d& p : S[i]) {
        p(0) -= mn;
        //std::cout << p(0) << " " << std::endl;
    }
    center += mn*R.col(0); // should this be minus??
    /*mn = RGB.col(i).mean();
    RGB.col(i).array() -= mn;
    RGB_means[i](0) = mn;
    mn = RGB.col(S.cols() + i).mean();
    RGB.col(S.cols() + i).array() -= mn;
    RGB_means[i](1) = mn;
    mn = RGB.col(2*S.cols() + i).mean();
    RGB.col(2*S.cols() + i).array() -= mn;
    RGB_means[i](2) = mn;*/
    W.col(i) = count > 0;
}

// do this for to_be_added instead, loop through leaves
void gp_compressor::train_processes()
{
    MatrixXd X;
    VectorXd y;
    gps.resize(to_be_added.size());
    int i;
    leaf_iterator iter(octree);
    while (*++iter) {
        gp_leaf* leaf = dynamic_cast<gp_leaf*>(*iter);
        if (leaf == NULL) {
            std::cout << "doesn't work, exiting..." << std::endl;
            exit(0);
        }
        i = leaf->gp_index;
        if (to_be_added[i].size() == 0) {
            continue;
        }
        leaf->reset();
        X.resize(to_be_added[i].size(), 2);
        y.resize(to_be_added[i].size());
        int m = 0;
        for (const Vector3d& p : to_be_added[i]) {
            X.row(m) = p.tail<2>().transpose().cast<double>();
            y(m) = p(0);
            ++m;
        }
        gps[i].add_measurements(X, y);
        to_be_added[i].clear();
    }
    /*leaf_iterator iter1(octree);
    while (*++iter1) {
        gp_leaf* leaf = dynamic_cast<gp_leaf*>(*iter1);
        if (leaf == NULL) {
            std::cout << "doesn't work, exiting..." << std::endl;
            exit(0);
        }
        std::cout << "Leaf " << leaf->gp_index << ", size " << leaf->getSize() << std::endl;
    }*/
}

void gp_compressor::project_cloud()
{
    octree.setInputCloud(cloud);
    octree.addPointsFromInputCloud();

    int n = octree.getLeafCount();

    S.resize(n);
    to_be_added.resize(n);
    W.resize(sz*sz, n);
    //RGB.resize(sz*sz, 3*centers.size());
    rotations.resize(n);
    means.resize(n);
    //RGB_means.resize(centers.size());

    double radius = sqrt(3.0f)/2.0f*res; // radius of the sphere encompassing the voxels

    std::vector<int> index_search;
    std::vector<float> distances;
    Eigen::Matrix3d R;
    Vector3d mid;
    int* occupied_indices = new int[cloud->width*cloud->height]();

    point center;
    int i = 0;
    leaf_iterator iter(octree);
    while(*++iter) {
        pcl::octree::OctreeKey key = iter.getCurrentOctreeKey();
        octree.generate_voxel_center(center, key);

        gp_leaf* leaf = dynamic_cast<gp_leaf*>(*iter);
        if (leaf == NULL) {
            std::cout << "doesn't work, exiting..." << std::endl;
            exit(0);
        }
        leaf->gp_index = i;

        octree.radiusSearch(center, radius, index_search, distances);
        MatrixXd points(4, index_search.size()); // 4 because of compute rotation
        points.row(3).setOnes();
        Matrix<short, Dynamic, Dynamic> colors(3, index_search.size());
        for (int m = 0; m < index_search.size(); ++m) {
            points(0, m) = cloud->points[index_search[m]].x;
            points(1, m) = cloud->points[index_search[m]].y;
            points(2, m) = cloud->points[index_search[m]].z;
            colors(0, m) = cloud->points[index_search[m]].r;
            colors(1, m) = cloud->points[index_search[m]].g;
            colors(2, m) = cloud->points[index_search[m]].b;
        }
        compute_rotation(R, points);
        mid = Vector3d(center.x, center.y, center.z);
        project_points(mid, R, points, colors, index_search, occupied_indices, i);
        rotations[i] = R;
        means[i] = mid;
        ++i;
    }
    delete[] occupied_indices;
}

gp_compressor::pointcloud::Ptr gp_compressor::load_compressed()
{
    int n = S.size();
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
    VectorXd V_star;
    double sum_squared_error = 0;
    for (int i = 0; i < n; ++i) {
        if (S[i].size() == 0) {
            continue;
        }

        // DEBUGGING, computing rms error
        X_star.resize(S[i].size(), 2);
        f.resize(S[i].size());
        int m = 0;
        for (const Vector3d& p : S[i]) {
            X_star.row(m) = p.tail<2>().transpose().cast<double>();
            f(m) = p(0);
            ++m;
        }
        data_points += S[i].size();
        gps[i].predict_measurements(f_star, X_star, V_star);
        sum_squared_error += (f - f_star).squaredNorm();
        // DEBUGGING, computing rms error

        X_star.resize(sz*sz, 2);
        points = 0;
        for (int y = 0; y < sz; ++y) { // ROOM FOR SPEEDUP
            for (int x = 0; x < sz; ++x) {
                ind = x*sz + y;
                if (!W(ind, i)) {
                    continue;
                }
                X_star(points, 0) = res*((double(x) + 0.5f)/double(sz) - 0.5f);
                X_star(points, 1) = res*((double(y) + 0.5f)/double(sz) - 0.5f);
                ++points;
            }
        }
        X_star.conservativeResize(points, 2);
        gps[i].predict_measurements(f_star, X_star, V_star);
        for (int m = 0; m < points; ++m) {
            pt(0) = f_star(m);
            pt(1) = X_star(m, 0); // both at the same time
            pt(2) = X_star(m, 1);
            pt = rotations[i].toRotationMatrix()*pt + means[i];
            //std::cout << pt.transpose() << std::endl;
            ncloud->at(counter).x = pt(0);
            ncloud->at(counter).y = pt(1);
            ncloud->at(counter).z = pt(2);
            int col = i % 3;
            if (col == 0) {
                ncloud->at(counter).g = 150;
            }
            else if (col == 1) {
                ncloud->at(counter).g = 200;
            }
            else {
                ncloud->at(counter).g = 250;
            }
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
    /*if (display) {
        display_cloud(ncloud, ncenters, normals);
    }*/
    return ncloud;
}

void gp_compressor::compress_depths()
{
    /*MatrixXd X;
    VectorXd y;
    for (int i = 0; i < S.size(); ++i) {
        X.resize(S[i].size(), 2);
        y.resize(S[i].size());
        int m = 0;
        for (const Vector3d& p : S[i]) {
            X.row(m) = p.tail<2>().transpose();
            y(m) = p(0);
        }
        gaussian_process gp;
        gp.add_measurements(X, y);
    }*/
}

void gp_compressor::compress_colors()
{

}
