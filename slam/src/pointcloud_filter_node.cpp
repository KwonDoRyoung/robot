/**
 * pointcloud_filter_node.cpp
 *
 * Pre-filters LiDAR point cloud before NDT localization.
 * Fixes: noise spikes, reflective artifacts, dense ground points
 *
 * Pipeline: PassThrough (distance) → VoxelGrid → StatisticalOutlierRemoval
 */

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.h>

#include <pcl_ros/point_cloud.h>
#include <pcl_ros/transforms.h>
#include <pcl_conversions/pcl_conversions.h>

#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/radius_outlier_removal.h>

using PointT = pcl::PointXYZI;
using Cloud  = pcl::PointCloud<PointT>;

class PointCloudFilterNode
{
public:
  PointCloudFilterNode() : nh_("~"), tf_listener_(tf_buffer_)
  {
    // Parameters
    nh_.param<std::string>("input_topic",  input_topic_,  "/livox/lidar_192_168_1_100");
    nh_.param<std::string>("output_topic", output_topic_, "/livox/lidar_filtered");
    nh_.param<std::string>("fixed_frame",  fixed_frame_,  "base_footprint");

    nh_.param("distance_near_threshold", dist_near_, 0.5);
    nh_.param("distance_far_threshold",  dist_far_,  50.0);
    nh_.param("voxel_leaf_size",         voxel_leaf_, 0.1);
    nh_.param("sor_mean_k",              sor_mean_k_,   20);
    nh_.param("sor_std_dev_mul_threshold", sor_std_dev_, 2.0);
    nh_.param("ror_enable",              ror_enable_,   false);
    nh_.param("ror_radius_search",       ror_radius_,   0.5);
    nh_.param("ror_min_neighbors",       ror_min_nb_,   5);
    nh_.param("height_filter_enable",    height_enable_, true);
    nh_.param("height_min",              height_min_,  -0.5);
    nh_.param("height_max",              height_max_,   2.0);

    sub_ = nh_.subscribe(input_topic_, 1, &PointCloudFilterNode::cloudCallback, this);
    pub_ = nh_.advertise<sensor_msgs::PointCloud2>(output_topic_, 1);

    ROS_INFO("[PCFilter] %s -> %s | voxel=%.2f sor_k=%d",
             input_topic_.c_str(), output_topic_.c_str(), voxel_leaf_, sor_mean_k_);
  }

private:
  void cloudCallback(const sensor_msgs::PointCloud2::ConstPtr& msg)
  {
    Cloud::Ptr cloud(new Cloud);
    pcl::fromROSMsg(*msg, *cloud);

    if (cloud->empty()) return;

    // 1. Distance filter (remove points outside [near, far])
    {
      pcl::PassThrough<PointT> pass;
      pass.setInputCloud(cloud);
      pass.setFilterFieldName("x");
      pass.setFilterLimits(-dist_far_, dist_far_);
      pass.filter(*cloud);

      pass.setFilterFieldName("y");
      pass.setFilterLimits(-dist_far_, dist_far_);
      pass.filter(*cloud);
    }

    // Remove blind zone near the sensor
    {
      Cloud::Ptr filtered(new Cloud);
      filtered->reserve(cloud->size());
      for (const auto& pt : *cloud) {
        float d = std::sqrt(pt.x * pt.x + pt.y * pt.y + pt.z * pt.z);
        if (d >= dist_near_ && d <= dist_far_) {
          filtered->push_back(pt);
        }
      }
      *cloud = *filtered;
    }

    // 2. Height filter (focus on wall-level features for localization)
    if (height_enable_) {
      pcl::PassThrough<PointT> pass;
      pass.setInputCloud(cloud);
      pass.setFilterFieldName("z");
      pass.setFilterLimits(height_min_, height_max_);
      pass.filter(*cloud);
    }

    if (cloud->empty()) return;

    // 3. Voxel downsampling (uniform density, noise reduction)
    {
      pcl::VoxelGrid<PointT> vg;
      vg.setInputCloud(cloud);
      vg.setLeafSize(voxel_leaf_, voxel_leaf_, voxel_leaf_);
      vg.filter(*cloud);
    }

    if (cloud->empty()) return;

    // 4. Statistical Outlier Removal (removes floating noise)
    {
      pcl::StatisticalOutlierRemoval<PointT> sor;
      sor.setInputCloud(cloud);
      sor.setMeanK(sor_mean_k_);
      sor.setStddevMulThresh(sor_std_dev_);
      sor.filter(*cloud);
    }

    // 5. Radius Outlier Removal (optional, for reflective surfaces)
    if (ror_enable_ && !cloud->empty()) {
      pcl::RadiusOutlierRemoval<PointT> ror;
      ror.setInputCloud(cloud);
      ror.setRadiusSearch(ror_radius_);
      ror.setMinNeighborsInRadius(ror_min_nb_);
      ror.filter(*cloud);
    }

    // Publish
    sensor_msgs::PointCloud2 out_msg;
    pcl::toROSMsg(*cloud, out_msg);
    out_msg.header = msg->header;
    pub_.publish(out_msg);
  }

  ros::NodeHandle nh_;
  ros::Subscriber sub_;
  ros::Publisher  pub_;

  tf2_ros::Buffer            tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  std::string input_topic_, output_topic_, fixed_frame_;

  double dist_near_, dist_far_;
  double voxel_leaf_;
  int    sor_mean_k_;
  double sor_std_dev_;
  bool   ror_enable_;
  double ror_radius_;
  int    ror_min_nb_;
  bool   height_enable_;
  double height_min_, height_max_;
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "pointcloud_filter_node");
  PointCloudFilterNode node;
  ros::spin();
  return 0;
}
