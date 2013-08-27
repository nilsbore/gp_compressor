#include "gp_compressor.h"

#include "gaussian_process.h"

#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/octree/octree_impl.h>
#include <pcl/io/pcd_io.h>
#include <stdint.h>
#include <boost/thread/thread.hpp>

using namespace Eigen;

gp_compressor::gp_compressor(pointcloud::ConstPtr cloud, float res, int sz) :
    cloud(cloud), res(res), sz(sz)
{

}

void gp_compressor::save_compressed(const std::string& name)
{
    std::cout << "Size of original point cloud: " << cloud->width*cloud->height << std::endl;
    project_cloud();
    std::cout << "Number of patches: " << S.size() << std::endl;
    compress_depths();
    compress_colors();
}

void gp_compressor::compute_rotation(Matrix3f& R, const MatrixXf& points)
{
    if (points.cols() < 4) {
        R.setIdentity();
        return;
    }
    JacobiSVD<MatrixXf> svd(points.transpose(), ComputeThinV); // kan ta U ist för transpose?
    Vector3f normal = svd.matrixV().block<3, 1>(0, 3);
    normal.normalize();
    Vector3f x(1.0f, 0.0f, 0.0f);
    Vector3f y(0.0f, 1.0f, 0.0f);
    Vector3f z(0.0f, 0.0f, 1.0f);
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

void gp_compressor::project_points(Vector3f& center, const Matrix3f& R, MatrixXf& points,
                                           const Matrix<short, Dynamic, Dynamic>& colors,
                                           const std::vector<int>& index_search,
                                           int* occupied_indices, int i)
{
    ArrayXi count(sz*sz);
    count.setZero(); // not needed anymore, only need weights
    Vector3f pt;
    Matrix<short, 3, 1> c;
    int ind;
    float mn = 0;
    for (int m = 0; m < points.cols(); ++m) {
        if (occupied_indices[index_search[m]]) {
            continue;
        }
        pt = R.transpose()*(points.block<3, 1>(0, m) - center);
        pt(1) += res/2.0f;
        pt(2) += res/2.0f;
        if (pt(1) > res || pt(1) < 0 || pt(2) > res || pt(2) < 0) {
            continue;
        }
        mn += pt(0);
        occupied_indices[index_search[m]] = 1;
        pt(1) *= float(sz)/res; // maybe just divide by res to get between 0 and 1
        pt(2) *= float(sz)/res;
        ind = sz*int(pt(1)) + int(pt(2));
        float current_count = count(ind);
        S[i].push_back(pt);
        c = colors.col(m);
        /*for (int n = 0; n < 3; ++n) {
            RGB(ind, n*S.cols() + i) = (current_count*RGB(ind, n*S.cols() + i) + float(c(n))) / (current_count + 1);
        }*/
        count(ind) += 1;
    }
    mn /= S[i].size(); // check that mn != 0
    for (Vector3f& p : S[i]) {
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

void gp_compressor::project_cloud()
{
    pcl::octree::OctreePointCloudSearch<point> octree(res);
    octree.setInputCloud(cloud);
    octree.addPointsFromInputCloud();

    std::vector<point, Eigen::aligned_allocator<point> > centers;
    octree.getOccupiedVoxelCenters(centers);

    S.resize(centers.size());
    W.resize(sz*sz, centers.size());
    //RGB.resize(sz*sz, 3*centers.size());
    rotations.resize(centers.size());
    means.resize(centers.size());
    //RGB_means.resize(centers.size());

    float radius = sqrt(3.0f)/2.0f*res; // radius of the sphere encompassing the voxels

    std::vector<int> index_search;
    std::vector<float> distances;
    Eigen::Matrix3f R;
    Vector3f mid;
    int* occupied_indices = new int[cloud->width*cloud->height]();
    point center;
    for (int i = 0; i < centers.size(); ++i) {
        center = centers[i];
        octree.radiusSearch(center, radius, index_search, distances);
        MatrixXf points(4, index_search.size());
        Matrix<short, Dynamic, Dynamic> colors(3, index_search.size());
        points.row(3).setOnes();
        for (int m = 0; m < index_search.size(); ++m) {
            points(0, m) = cloud->points[index_search[m]].x;
            points(1, m) = cloud->points[index_search[m]].y;
            points(2, m) = cloud->points[index_search[m]].z;
            colors(0, m) = cloud->points[index_search[m]].r;
            colors(1, m) = cloud->points[index_search[m]].g;
            colors(2, m) = cloud->points[index_search[m]].b;
        }
        compute_rotation(R, points);
        mid = Vector3f(center.x, center.y, center.z);
        project_points(mid, R, points, colors, index_search, occupied_indices, i);
        rotations[i] = R;
        means[i] = mid;
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
    Vector3f pt;
    int counter = 0;
    int points;
    int ind;
    int data_points = 0;
    MatrixXf X;
    VectorXf y;
    VectorXf f; // DEBUGGING, computing rms error
    MatrixXf X_star;
    VectorXf f_star;
    VectorXf V_star;
    float sum_squared_error = 0;
    for (int i = 0; i < n; ++i) {
        X.resize(S[i].size(), 2);
        y.resize(S[i].size());
        int m = 0;
        for (const Vector3f& p : S[i]) {
            if (rand() % 2 == 0 || rand() % 2 == 0 || rand() % 2 == 0 || rand() % 2 == 0) {
                continue;
            }
            X.row(m) = p.tail<2>().transpose();
            y(m) = p(0);
            ++m;
        }
        X.conservativeResize(m, 2);
        y.conservativeResize(m);
        gaussian_process gp;
        gp.add_measurements(X, y);

        // DEBUGGING, computing rms error
        X_star.resize(S[i].size(), 2);
        f.resize(S[i].size());
        m = 0;
        for (const Vector3f& p : S[i]) {
            X_star.row(m) = p.tail<2>().transpose();
            f(m) = p(0);
            ++m;
        }
        data_points += S[i].size();
        gp.evaluate_points(f_star, V_star, X_star);
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
                X_star(points, 0) = float(x);//(pt(1) + 0.5f)*res/float(sz) - res/2.0f; // both at the same time
                X_star(points, 1) = float(y);//(pt(2) + 0.5f)*res/float(sz) - res/2.0f;
                ++points;
            }
        }
        X_star.conservativeResize(points, 2);
        gp.evaluate_points(f_star, V_star, X_star);
        for (int m = 0; m < points; ++m) {
            pt(0) = f_star(m);
            pt(1) = (X_star(m, 0) + 0.5f)*res/float(sz) - res/2.0f; // both at the same time
            pt(2) = (X_star(m, 1) + 0.5f)*res/float(sz) - res/2.0f;
            pt = rotations[i].toRotationMatrix()*pt + means[i];
            //std::cout << pt.transpose() << std::endl;
            ncloud->at(counter).x = pt(0);
            ncloud->at(counter).y = pt(1);
            ncloud->at(counter).z = pt(2);
            int col = i % 3;
            if (col == 0) {
                ncloud->at(counter).r = 255;
            }
            else if (col == 1) {
                ncloud->at(counter).g = 255;
            }
            else {
                ncloud->at(counter).b = 255;
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
    std::cout << "RMS error: " << sqrt(sum_squared_error / float(data_points)) << std::endl;
    ncloud->resize(counter);
    std::cout << "Size of transformed point cloud: " << ncloud->width*ncloud->height << std::endl;
    /*if (display) {
        display_cloud(ncloud, ncenters, normals);
    }*/
    return ncloud;
}

void gp_compressor::compress_depths()
{
    /*MatrixXf X;
    VectorXf y;
    for (int i = 0; i < S.size(); ++i) {
        X.resize(S[i].size(), 2);
        y.resize(S[i].size());
        int m = 0;
        for (const Vector3f& p : S[i]) {
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

void gp_compressor::random_positions(std::vector<int>& rtn, int n, int m)
{
    if (n < m) {
        rtn.resize(n);
        for (int j = 0; j < n; ++j) {
            rtn[j] = j;
        }
        return;
    }
    if (m == 0) {
        return;
    }
    rtn.resize(m);
    std::srand(std::time(0)); // use current time as seed for random generator
    int ind;
    for (int j = 0; j < m; ++j) {
        do {
            ind = std::rand() % n;
        }
        while (std::find(rtn.begin(), rtn.end(), ind) != rtn.end());
        rtn[j] = ind;
    }
}
