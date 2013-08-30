#include "pointcloud_compressor.h"

#include "ksvd_decomposition.h"

#include <pcl/octree/octree_impl.h>
#include <stdint.h>

using namespace Eigen;

pointcloud_compressor::pointcloud_compressor(pointcloud::ConstPtr cloud, float res, int sz, int dict_size,
                                             int words_max, float proj_error, float stop_diff, int RGB_dict_size,
                                             int RGB_words_max, float RGB_proj_error, float RGB_stop_diff) :
    dictionary_representation(res, sz, dict_size, words_max, RGB_dict_size, RGB_words_max),
    cloud(cloud), proj_error(proj_error), RGB_proj_error(RGB_proj_error),
    stop_diff(stop_diff), RGB_stop_diff(RGB_stop_diff)
{

}

void pointcloud_compressor::save_compressed(const std::string& name)
{
    std::cout << "Size of original point cloud: " << cloud->width*cloud->height << std::endl;
    project_cloud();
    std::cout << "Number of patches: " << S.cols() << std::endl;
    compress_depths();
    compress_colors();
    write_to_file(name);
    std::cout << "RMS error: " << compute_rms_error() << std::endl;
}

void pointcloud_compressor::compute_rotation(Matrix3f& R, const MatrixXf& points)
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

void pointcloud_compressor::project_points(Vector3f& center, const Matrix3f& R, MatrixXf& points,
                                           const Matrix<short, Dynamic, Dynamic>& colors,
                                           const std::vector<int>& index_search,
                                           int* occupied_indices, int i)
{
    ArrayXi count(sz*sz);
    count.setZero();
    Vector3f pt;
    Matrix<short, 3, 1> c;
    int x, y, ind;
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
        occupied_indices[index_search[m]] = 1;
        x = int(float(sz)*pt(1)/res);
        y = int(float(sz)*pt(2)/res);
        ind = sz*x + y;
        float current_count = count(ind);
        patch_points[i].push_back(pt); // DEBUG, compute rms error
        S(ind, i) = (current_count*S(ind, i) + pt(0)) / (current_count + 1);
        c = colors.col(m);
        for (int n = 0; n < 3; ++n) {
            RGB(ind, n*S.cols() + i) = (current_count*RGB(ind, n*S.cols() + i) + float(c(n))) / (current_count + 1);
        }
        count(ind) += 1;
    }
    float mn = S.col(i).mean();
    S.col(i).array() -= mn;
    for (Vector3f& p : patch_points[i]) { // DEBUG, compute rms error
        p(0) -= mn;
    }
    center += mn*R.col(0); // should this be minus??
    mn = RGB.col(i).mean();
    RGB.col(i).array() -= mn;
    RGB_means[i](0) = mn;
    mn = RGB.col(S.cols() + i).mean();
    RGB.col(S.cols() + i).array() -= mn;
    RGB_means[i](1) = mn;
    mn = RGB.col(2*S.cols() + i).mean();
    RGB.col(2*S.cols() + i).array() -= mn;
    RGB_means[i](2) = mn;
    W.col(i) = count > 0;
}

void pointcloud_compressor::project_cloud()
{
    pcl::octree::OctreePointCloudSearch<point> octree(res);
    octree.setInputCloud(cloud);
    octree.addPointsFromInputCloud();

    std::vector<point, Eigen::aligned_allocator<point> > centers;
    octree.getOccupiedVoxelCenters(centers);

    patch_points.resize(centers.size());
    S.resize(sz*sz, centers.size());
    W.resize(sz*sz, centers.size());
    RGB.resize(sz*sz, 3*centers.size());
    rotations.resize(centers.size());
    means.resize(centers.size());
    RGB_means.resize(centers.size());

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

void pointcloud_compressor::compress_depths()
{
    ksvd_decomposition ksvd(X, I, D, number_words, S, W, dict_size, words_max, proj_error, stop_diff);
}

void pointcloud_compressor::compress_colors()
{
    Matrix<bool, Dynamic, Dynamic> RGB_W(sz*sz, RGB.cols());
    for (int n = 0; n < 3; ++n) {
        RGB_W.block(0, n*S.cols(), sz*sz, S.cols()) = W;
    }
    ksvd_decomposition(RGB_X, RGB_I, RGB_D, RGB_number_words, RGB,
                       RGB_W, RGB_dict_size, RGB_words_max, RGB_proj_error, RGB_stop_diff); // 1e3, 1e2
}

float pointcloud_compressor::compute_rms_error()
{
    VectorXf s_err(sz*sz);
    int x, y, ind;
    float sum_squared_error = 0;
    float error;
    int n = 0;
    for (int i = 0; i < S.cols(); ++i) {
        s_err.setZero();
        for (int k = 0; k < number_words[i]; ++k) {
            s_err += X(k, i)*D.col(I(k, i));
        }
        // s_err = S.col(i); // for testing the influence of the averaging
        for (const VectorXf& p : patch_points[i]) {
            x = int(float(sz)*p(1)/res);
            y = int(float(sz)*p(2)/res);
            ind = sz*x + y;
            error = s_err(ind) - p(0);
            sum_squared_error += error * error;
        }
        n += patch_points[i].size();
    }
    return sqrt(sum_squared_error / float(n));
}
