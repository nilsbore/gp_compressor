#include "gp_registration.h"

using namespace Eigen;

gp_registration::gp_registration(pointcloud::ConstPtr cloud, double res, int sz,
                                 pcl::PointCloud<pcl::PointXYZ>::Ptr ncenters,
                                 pcl::PointCloud<pcl::Normal>::Ptr normals) :
    gp_compressor(cloud, res, sz),
    accumulated_weight(0), step(0.1), ncenters(ncenters), normals(normals)
{
    covariance.setZero();
    mean1.setZero();
    mean2.setZero();
    project_cloud();
    std::cout << "Number of patches: " << S.size() << std::endl;
    train_processes();
}

void gp_registration::add_cloud(pointcloud::ConstPtr other_cloud)
{
    cloud->clear();
    cloud->insert(cloud->end(), other_cloud->begin(), other_cloud->end());
    octree.update_points();
    compute_transformation();
    Matrix3d R;
    Vector3d t;
    get_transformation(R, t);
}

void gp_registration::get_local_points(MatrixXd& points, int* occupied_indices, const std::vector<int>& index_search, int i)
{
    Vector3d pt;
    int ind;
    int k = 0;
    for (int m = 0; m < points.cols(); ++m) {
        ind = index_search[m];
        if (occupied_indices[ind]) {
            continue;
        }
        pt = rotations[i].toRotationMatrix().transpose()*(points.col(m) - means[i]); // transforming to the patch coordinate system
        if (pt(1) > res/2.0f || pt(1) < -res/2.0f || pt(2) > res/2.0f || pt(2) < -res/2.0f) {
            continue;
        }
        occupied_indices[ind] = 1;
        points.col(k) = pt;
        ++k;
    }
    points.conservativeResize(3, k);
}

void gp_registration::compute_transformation()
{
    if (ncenters) {
        ncenters->resize(cloud->size());
        normals->resize(cloud->size());
    }
    double radius = sqrt(3.0f)/2.0f*res; // radius of the sphere encompassing the voxels

    std::vector<int> index_search;
    std::vector<float> distances;
    int* occupied_indices = new int[cloud->width*cloud->height](); // cloud.size()?

    Matrix3d R;
    Vector3d t;
    point center;
    int i;
    int k = 0; // for normal plotting
    leaf_iterator iter(octree);
    while(*++iter) {
        pcl::octree::OctreeKey key = iter.getCurrentOctreeKey();
        octree.generate_voxel_center(center, key);

        gp_leaf* leaf = dynamic_cast<gp_leaf*>(*iter);
        if (leaf == NULL) {
            std::cout << "doesn't work, exiting..." << std::endl;
            exit(0);
        }

        if (leaf->gp_index == -1) {
            continue;
        }
        octree.radiusSearch(center, radius, index_search, distances);
        i = leaf->gp_index;
        MatrixXd points(3, index_search.size());
        for (int m = 0; m < index_search.size(); ++m) {
            points(0, m) = cloud->points[index_search[m]].x;
            points(1, m) = cloud->points[index_search[m]].y;
            points(2, m) = cloud->points[index_search[m]].z;
        }
        get_local_points(points, occupied_indices, index_search, i);
        MatrixXd dX;
        gps[i].compute_derivatives(dX, points.block(0, 0, 2, points.cols()).transpose().cast<double>(),
                                   points.row(2).transpose().cast<double>());
        // transform points and derivatives to global system
        R = rotations[i].toRotationMatrix();
        t = means[i];
        for (int m = 0; m < points.cols(); ++m) {
            points.col(m) = R*points.col(m) + t;
        }
        dX *= R.transpose(); // transpose because vectors transposed
        if (ncenters) {
            for (int m = 0; m < points.cols(); ++m) {
                ncenters->at(k).x = points(0, m);
                ncenters->at(k).y = points(1, m);
                ncenters->at(k).z = points(2, m);
                normals->at(k).normal_x = 1e-5f*dX(m, 0);
                normals->at(k).normal_y = 1e-5f*dX(m, 1);
                normals->at(k).normal_z = 1e-5f*dX(m, 2);
                ++k;
            }
        }

        // get derivatives of points from gp
        add_derivatives(points.transpose().cast<double>(), dX);
        std::cout << "Number of points: " << points.cols() << std::endl;
        std::cout << "Index search: " << index_search.size() << std::endl;
    }
    if (ncenters) {
        ncenters->resize(k);
        normals->resize(k);
    }
    delete[] occupied_indices;
}

void gp_registration::add_derivatives(const MatrixXd& X, const MatrixXd& dX)
{
    Vector3d diff1;
    Vector3d diff2;
    Vector3d cx;
    Vector3d x;
    Vector3d dx;
    for (int i = 0; i < X.rows(); ++i) {
        x = X.row(i).transpose();
        dx = dX.row(i).transpose();
        double weight = dx.norm();
        if (weight == 0.0f) {
            continue;
        }

        cx = x + step/weight*dx;

        accumulated_weight += weight;
        double alpha = weight / accumulated_weight;

        diff1 = x - mean1;
        diff2 = cx - mean2;
        covariance = (1.0f - alpha)*(covariance + alpha * (diff2 * diff1.transpose()));

        mean1 += alpha*diff1;
        mean2 += alpha*diff2;
    }
}

void gp_registration::get_transformation(Matrix3d& R, Vector3d& t)
{
    JacobiSVD<Matrix3d> svd(covariance, Eigen::ComputeFullU | Eigen::ComputeFullV);
    const Matrix3d& u = svd.matrixU();
    const Matrix3d& v = svd.matrixV();
    Matrix3d s;
    s.setIdentity();
    if (u.determinant()*v.determinant() < 0.0f) {
        s(2, 2) = -1.0f;
    }
    R = u * s * v.transpose();
    t = mean2 - R*mean1;
}
