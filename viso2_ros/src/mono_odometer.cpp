#include <ros/ros.h>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/camera_subscriber.h>
#include <image_transport/image_transport.h>
#include <image_geometry/pinhole_camera_model.h>

#include <viso_mono.h>

#include <viso2_ros/VisoInfo.h>

#include "odometer_base.h"
#include "odometry_params.h"

namespace viso2_ros
{

class MonoOdometer : public OdometerBase
{

private:

  boost::shared_ptr<VisualOdometryMono> visual_odometer_;
  VisualOdometryMono::parameters visual_odometer_params_;

  image_transport::CameraSubscriber camera_sub_;

  ros::Publisher info_pub_;
  ros::Publisher image_pub_;

  bool replace_;

  int max_y;

public:

  MonoOdometer(const std::string& transport) : OdometerBase(), replace_(false)
  {
    // Read local parameters
    ros::NodeHandle local_nh("~");
    odometry_params::loadParams(local_nh, visual_odometer_params_);

    max_y = 600;
    local_nh.getParam("max_y", max_y);

    ros::NodeHandle nh;
    image_transport::ImageTransport it(nh);
    camera_sub_ = it.subscribeCamera("image", 1, &MonoOdometer::imageCallback, this, transport);

    info_pub_ = local_nh.advertise<VisoInfo>("info", 1);

    // (jcg) Output matches debug image
    image_pub_ = local_nh.advertise<sensor_msgs::Image>("image_matches", 10);
  }

protected:

  void imageCallback(
      const sensor_msgs::ImageConstPtr& image_msg,
      const sensor_msgs::CameraInfoConstPtr& info_msg)
  {
    ros::WallTime start_time = ros::WallTime::now();
 
    bool first_run = false;
    // create odometer if not exists
    if (!visual_odometer_)
    {
      first_run = true;
      // read calibration info from camera info message
      // to fill remaining odometer parameters
      image_geometry::PinholeCameraModel model;
      model.fromCameraInfo(info_msg);
      visual_odometer_params_.calib.f  = model.fx();
      visual_odometer_params_.calib.cu = model.cx();
      visual_odometer_params_.calib.cv = model.cy();
      visual_odometer_.reset(new VisualOdometryMono(visual_odometer_params_));
      if (image_msg->header.frame_id != "") setSensorFrameId(image_msg->header.frame_id);
      ROS_INFO_STREAM("Initialized libviso2 mono odometry "
                      "with the following parameters:" << std::endl << 
                      visual_odometer_params_);
    }

    cv_bridge::CvImageConstPtr masked_cv_ptr;
    masked_cv_ptr = cv_bridge::toCvShare(image_msg, sensor_msgs::image_encodings::MONO8);
    cv::Mat_<uint8_t> gray(masked_cv_ptr->image);
    maskImage(gray, max_y);

    // convert image if necessary
    uint8_t *image_data;
    int step;
    cv_bridge::CvImageConstPtr cv_ptr;
    // TODO: Handle Mono8 directly in the future
    // if (image_msg->encoding == sensor_msgs::image_encodings::MONO8)
    // {
    //   image_data = const_cast<uint8_t*>(&(image_msg->data[0]));
    //   step = image_msg->step;
    // }
    // else
    {
      cv_ptr = cv_bridge::toCvShare(image_msg, sensor_msgs::image_encodings::MONO8);
      image_data = masked_cv_ptr->image.data;
      step = masked_cv_ptr->image.step[0];
    }

    // run the odometer
    int32_t dims[] = {image_msg->width, image_msg->height, step};
    // on first run, only feed the odometer with first image pair without
    // retrieving data
    if (first_run)
    {
      visual_odometer_->process(image_data, dims);
      tf::Transform delta_transform;
      delta_transform.setIdentity();
      integrateAndPublish(delta_transform, image_msg->header.stamp);
    }
    else
    {
      bool success = visual_odometer_->process(image_data, dims);
      if(success)
      {
        replace_ = false;
        Matrix camera_motion = Matrix::inv(visual_odometer_->getMotion());
        ROS_DEBUG("Found %i matches with %i inliers.", 
                  visual_odometer_->getNumberOfMatches(),
                  visual_odometer_->getNumberOfInliers());
        ROS_DEBUG_STREAM("libviso2 returned the following motion:\n" << camera_motion);

        tf::Matrix3x3 rot_mat(
          camera_motion.val[0][0], camera_motion.val[0][1], camera_motion.val[0][2],
          camera_motion.val[1][0], camera_motion.val[1][1], camera_motion.val[1][2],
          camera_motion.val[2][0], camera_motion.val[2][1], camera_motion.val[2][2]);
        tf::Vector3 t(camera_motion.val[0][3], camera_motion.val[1][3], camera_motion.val[2][3]);
        tf::Transform delta_transform(rot_mat, t);

        integrateAndPublish(delta_transform, image_msg->header.stamp);
      }
      else
      {
        ROS_DEBUG("Call to VisualOdometryMono::process() failed. Assuming motion too small.");
        replace_ = true;
        tf::Transform delta_transform;
        delta_transform.setIdentity();
        integrateAndPublish(delta_transform, image_msg->header.stamp);
      }

      // create and publish viso2 info msg
      VisoInfo info_msg;
      info_msg.header.stamp = image_msg->header.stamp;
      info_msg.got_lost = !success;
      info_msg.change_reference_frame = false;
      info_msg.num_matches = visual_odometer_->getNumberOfMatches();
      info_msg.num_inliers = visual_odometer_->getNumberOfInliers();
      ros::WallDuration time_elapsed = ros::WallTime::now() - start_time;
      info_msg.runtime = time_elapsed.toSec();
      info_pub_.publish(info_msg);


      cv_bridge::CvImagePtr cv_ptr;
      try
      {
        cv_ptr = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::BGR8);
      }
      catch (const cv_bridge::Exception& e)
      {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
      }

      cv::Mat_<cv::Vec3b> color(cv_ptr->image);
      std::vector<Matcher::p_match> matches = visual_odometer_->getMatches();
      maskImage(color, max_y);
      drawMatches(color, matches);

      image_pub_.publish(cv_ptr->toImageMsg());
    }
  }

  // (jcg) As a simple hack, mask out requested regions of image

  void maskImage(cv::Mat& mat, int max_y)
  {
    // Mask out portions of image beyond min/max x and y
    // Only max_y is defined at the moment
    int height = std::min(mat.rows, std::max(0, mat.rows - max_y));
    cv::Rect roi(0, max_y,mat.cols, height); // x,y,width,height
    cv::Mat sub_mat(mat, roi);

    if (mat.channels() == 1)
    {
      sub_mat.setTo(cv::Scalar(0));
    }
    else if(mat.channels() == 3)
    {
      sub_mat.setTo(cv::Scalar(0,0,0));
    }
    else
    {
      ROS_ERROR("Had non-standard number of channels (not 1 or 3)");
    }
  }

  void drawMatches(cv::Mat_<cv::Vec3b>& color, std::vector<Matcher::p_match> matches)
  {
    // Test, draw line along center of image
    for (const auto match : matches)
    {
      // Assume for mono that left image previous and current values are filled out
      cv::line(color, 
              cv::Point(match.u1p, match.v1p),// Match in prev. left image
              cv::Point(match.u1c, match.v1c),// Match in current left image
              cv::Scalar(255,0,0), // Red
              1, // Thickness
              cv::LINE_8); // Line type
      // cv::Vec3b& bgr = color(color.rows/2.0, x);
      // bgr[0] = static_cast<uint8_t>(25);
      // bgr[1] = static_cast<uint8_t>(233);
      // bgr[2] = static_cast<uint8_t>(75);
    }
  }

};

} // end of namespace


int main(int argc, char **argv)
{
  ros::init(argc, argv, "mono_odometer");
  if (ros::names::remap("image").find("rect") == std::string::npos) {
    ROS_WARN("mono_odometer needs rectified input images. The used image "
             "topic is '%s'. Are you sure the images are rectified?",
             ros::names::remap("image").c_str());
  }

  std::string transport = argc > 1 ? argv[1] : "raw";
  viso2_ros::MonoOdometer odometer(transport);
  
  ros::spin();
  return 0;
}

