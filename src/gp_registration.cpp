#include "gp_registration.h"

using namespace Eigen;

gp_registration::gp_registration(pointcloud::ConstPtr cloud, double res, int sz,
                                 asynch_visualizer* vis) :
    gp_compressor(cloud, res, sz), step(1e-3f), vis(vis), P(6)
{
    /*covariance.setZero();
    mean1.setZero();
    mean2.setZero();
    accumulated_weight = 0;*/
    project_cloud();
    std::cout << "Number of patches: " << S.size() << std::endl;
    train_processes();
}

void gp_registration::get_cloud_transformation(Matrix3d& R, Vector3d& t)
{
    R = R_cloud;
    t = t_cloud;
}

void gp_registration::transform_pointcloud(pcl::PointCloud<pcl::PointXYZ>::Ptr c, const Matrix3d& R, const Vector3d& t)
{
    int n = c->size();
    for (int i = 0; i < n; ++i) {
        c->points[i].getVector3fMap() = (R*c->points[i].getVector3fMap().cast<double>() + t).cast<float>();
    }
}

void gp_registration::transform_pointcloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr c, const Matrix3d& R, const Vector3d& t)
{
    int n = c->size();
    for (int i = 0; i < n; ++i) {
        c->points[i].getVector3fMap() = (R*c->points[i].getVector3fMap().cast<double>() + t).cast<float>();
    }
}

void gp_registration::get_transform_jacobian(MatrixXd& J, const Vector3d& x)
{
    J.block<3, 3>(0, 0).setIdentity();
    J(0, 4) = x(2);
    J(0, 5) = -x(1);
    J(1, 3) = -x(2);
    J(1, 5) = x(0);
    J(2, 3) = x(1);
    J(2, 4) = -x(0);
}

void gp_registration::gradient_step(Matrix3d& R, Vector3d& t)
{
    Matrix3d Rx = AngleAxisd(step*delta(3), Vector3d::UnitX()).matrix();
    Matrix3d Ry = AngleAxisd(step*delta(4), Vector3d::UnitY()).matrix();
    Matrix3d Rz = AngleAxisd(step*delta(5), Vector3d::UnitZ()).matrix();
    R = Rx*Ry*Rz;
    t = step*delta.head<3>().transpose();
}

void gp_registration::add_cloud(pointcloud::ConstPtr other_cloud)
{
    cloud->clear();
    cloud->insert(cloud->end(), other_cloud->begin(), other_cloud->end());
    step_nbr = 0;
    R_cloud.setIdentity();
    t_cloud.setZero();
}

bool gp_registration::registration_done()
{
    return (delta.head<3>().norm() < 0.2 && delta.tail<3>().norm() < 0.2);
}

void gp_registration::registration_step()
{
    Matrix3d R;
    Vector3d t;
    octree.update_random_points(0.1f);
    //octree.update_points();
    compute_transformation();
    gradient_step(R, t);
    R_cloud = R*R_cloud; // add to total rotation
    t_cloud += t; // add to total translation
    transform_pointcloud(cloud, R, t);
    std::cout << "Doing step number " << step_nbr << std::endl;
    std::cout << "P derivative " << delta << std::endl;
    std::cout << "P angles norm " << delta.tail<3>().norm() << std::endl;
    std::cout << "P translation norm " << delta.head<3>().norm() << std::endl;
    ++step_nbr;
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
    if (vis != NULL) {
        vis->lock();
        vis->display_centers->resize(cloud->size());
        vis->display_normals->resize(cloud->size());
        vis->unlock();
    }
    double radius = sqrt(3.0f)/2.0f*res; // radius of the sphere encompassing the voxels

    std::vector<int> index_search;
    std::vector<float> distances;
    int* occupied_indices = new int[cloud->width*cloud->height](); // cloud.size()?

    Matrix3d R;
    Vector3d t;
    MatrixXd J(3, 6);
    MatrixXd dX;
    //MatrixXd dX_temp;
    VectorXd l;
    point center;
    int i;
    P.setZero();
    ls = 0;
    added_derivatives = 0;
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
        i = leaf->gp_index;
        if (i == -1 || gps[i].size() == 0) { // size check to be unnecessary, just initialize gps > 100 points in leaf
            continue;
        }
        octree.radiusSearch(center, radius, index_search, distances); // search in octree
        //leaf->reset(); // remove references in octree
        MatrixXd points(3, index_search.size());
        for (int m = 0; m < index_search.size(); ++m) {
            points(0, m) = cloud->points[index_search[m]].x;
            points(1, m) = cloud->points[index_search[m]].y;
            points(2, m) = cloud->points[index_search[m]].z;
        }

        get_local_points(points, occupied_indices, index_search, i);
        gps[i].compute_derivatives_fast(dX, points.block(1, 0, 2, points.cols()).transpose().cast<double>(),
                                   points.row(0).transpose().cast<double>());
        /*gps[i].compute_derivatives_fast(dX_temp, points.block(1, 0, 2, points.cols()).transpose().cast<double>(),
                                   points.row(0).transpose().cast<double>());
        double diff = (dX-dX_temp).array().abs().sum();
        if (diff > 1e-6f) {
            std::cout << dX-dX_temp << std::endl;
            std::cout << "Didn't work, result: " << diff << std::endl;
        }*/
        //gps[i].compute_likelihoods(l, points.block(1, 0, 2, points.cols()).transpose().cast<double>(),
          //                         points.row(0).transpose().cast<double>());
        // transform points and derivatives to global system
        R = rotations[i].toRotationMatrix();
        t = means[i];
        dX *= R.transpose(); // transpose because vectors transposed
        for (int m = 0; m < points.cols(); ++m) {
            points.col(m) = R*points.col(m) + t;
            get_transform_jacobian(J, points.col(m));
            //std::cout << P << std::endl;
            //std::cout << dX.row(m)*J << std::endl;
            //ls = (added_derivatives/(added_derivatives+1.0f))*ls + 1.0f/(added_derivatives+1.0f)*l(m);
            P = (added_derivatives/(added_derivatives+1.0f))*P + 1.0f/(added_derivatives+1.0f)*dX.row(m)*J;
            ++added_derivatives;
        }

        if (vis != NULL) {
            vis->lock();
            for (int m = 0; m < points.cols(); ++m) {
                vis->display_centers->at(k).x = points(0, m);
                vis->display_centers->at(k).y = points(1, m);
                vis->display_centers->at(k).z = points(2, m);
                vis->display_normals->at(k).normal_x = dX(m, 0);
                vis->display_normals->at(k).normal_y = dX(m, 1);
                vis->display_normals->at(k).normal_z = dX(m, 2);
                ++k;
            }
            vis->unlock();
        }

        // get derivatives of points from gp
        //add_derivatives(points.transpose().cast<double>(), dX);
        //std::cout << "Number of points: " << points.cols() << std::endl;
        //std::cout << "Index search: " << index_search.size() << std::endl;
    }
    octree.remove_just_points(); // not sure if this has any effect, removes points but not leaves
    if (vis != NULL) {
        vis->lock();
        vis->display_centers->resize(k);
        vis->display_normals->resize(k);
        vis->has_transformed = true;
        vis->unlock();
    }
    delete[] occupied_indices;

    //JacobiSVD<MatrixXd> H(P.transpose()*P, ComputeThinU | ComputeThinV);
    //delta = ls*H.solve(P.transpose());
    delta = P.transpose();
}

/*void gp_registration::add_derivatives(const MatrixXd& X, const MatrixXd& dX)
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
}*/
