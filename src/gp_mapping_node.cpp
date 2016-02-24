#include <iostream>
#include <string>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/utils.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include "gp_mapping.h"
#include "asynch_visualizer.h"
#include <dirent.h>

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_ros/point_cloud.h>

using namespace std;

class gp_mapping_node {
private:
    gp_mapping* comp;
    pcl::PointCloud<pcl::PointXYZ>::Ptr ncenters;
    pcl::PointCloud<pcl::Normal>::Ptr normals;
    asynch_visualizer viewer;
    ros::NodeHandle n;
    ros::Publisher pub;
    ros::Subscriber sub;

public:
    // if it does not work with the visualizer I can just publish the cloud on a topic, it's actually nicer
    void callback(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& msg_cloud)
    {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
        for (const pcl::PointXYZ& p : msg_cloud->points) {
            cloud->push_back(pcl::PointXYZRGB());
            cloud->back().getVector3fMap() = p.getVector3fMap();
        }

        if (comp == NULL) {
            comp = new gp_mapping(cloud, 2.0f, 20, NULL);
            cout << "Initialized gp processes" << endl;
            //viewer.display_cloud = comp->load_compressed();
            //viewer.create_thread();
            return;
        }

        comp->add_cloud(cloud);
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr display_cloud = comp->load_compressed();
        display_cloud->header.frame_id = "base_link";
        display_cloud->height = 1;
        display_cloud->width = display_cloud->size();
        pub.publish(display_cloud);
        //viewer.lock();
        //viewer.display_cloud = display_cloud;
        //viewer.map_has_transformed = true;
        //viewer.unlock();
    }
    gp_mapping_node() : comp(NULL), ncenters(new pcl::PointCloud<pcl::PointXYZ>()),
                        normals(new pcl::PointCloud<pcl::Normal>()), viewer(ncenters, normals)
    {
        pub = n.advertise<pcl::PointCloud<pcl::PointXYZRGB> >("/gp_cloud", 1);
        sub = n.subscribe("/velodyne_points", 1000, &gp_mapping_node::callback, this);
    }
    ~gp_mapping_node()
    {
        viewer.join_thread();
    }
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "gp_mapping");

    gp_mapping_node node;

    ros::spin();

    return 0;
}

