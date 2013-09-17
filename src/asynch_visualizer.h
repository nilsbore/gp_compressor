#ifndef ASYNCH_VISUALIZER_H
#define ASYNCH_VISUALIZER_H

#include <boost/thread/thread.hpp>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

class asynch_visualizer
{
private:
    pthread_mutex_t mutex;
public:
    bool has_transformed;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr display_cloud;
    pcl::PointCloud<pcl::PointXYZ>::Ptr display_centers;
    pcl::PointCloud<pcl::Normal>::Ptr display_normals;
    void lock();
    void unlock();
    void run_visualizer();
    asynch_visualizer(pcl::PointCloud<pcl::PointXYZ>::Ptr display_centers,
                      pcl::PointCloud<pcl::Normal>::Ptr display_normals) :
        display_centers(display_centers), display_normals(display_normals)
    {
        has_transformed = false;
        if (pthread_mutex_init(&mutex, NULL) == 0) {
            std::cout << "mutex init failed" << std::endl;
        }
        //pthread_mutex_lock(&mutex);
    }
};

void* viewer_thread(void* ptr);

#endif // ASYNCH_VISUALIZER_H
