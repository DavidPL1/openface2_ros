#include <ros/ros.h>
#include <ros/package.h>
#include <stdio.h>
#include <unistd.h>

#include <string>
#include <vector>
#include <unordered_map>
#include <iostream>
#include <memory>
#include <tuple>
#include <set>
#include <sstream>
#include <thread>
#include <atomic>
#include <mutex>
#include <future>
#include <exception>

#include <tbb/tbb.h>

#include <opencv2/videoio/videoio.hpp>
#include <opencv2/videoio/videoio_c.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "openface2_ros_msgs/ActionUnits.h"
#include "openface2_ros_msgs/ActionUnit.h"
#include "openface2_ros_msgs/Face.h"
#include "openface2_ros_msgs/Faces.h"

#include <sensor_msgs/Image.h>

#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>

#include "LandmarkCoreIncludes.h"
#include "Face_utils.h"
#include "FaceAnalyser.h"
#include "Visualizer.h"
#include "VisualizationUtils.h"

#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/TransformStamped.h>

using namespace std;
using namespace ros;
using namespace cv;

namespace
{
  static geometry_msgs::Quaternion toQuaternion(double pitch, double roll, double yaw)
  {
    double t0 = std::cos(yaw * 0.5f);
    double t1 = std::sin(yaw * 0.5f);
    double t2 = std::cos(roll * 0.5f);
    double t3 = std::sin(roll * 0.5f);
    double t4 = std::cos(pitch * 0.5f);
    double t5 = std::sin(pitch * 0.5f);

    geometry_msgs::Quaternion q;
    q.w = t0 * t2 * t4 + t1 * t3 * t5;
    q.x = t0 * t3 * t4 - t1 * t2 * t5;
    q.y = t0 * t2 * t5 + t1 * t3 * t4;
    q.z = t1 * t2 * t4 - t0 * t3 * t5;
    return q;
  }

  static geometry_msgs::Quaternion operator *(const geometry_msgs::Quaternion &a, const geometry_msgs::Quaternion &b)
  {
    geometry_msgs::Quaternion q;
    
    q.w = a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z;  // 1
    q.x = a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y;  // i
    q.y = a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x;  // j
    q.z = a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w;  // k
    return q;
  }

 void NonOverlapingDetections(const vector<LandmarkDetector::CLNF>& clnf_models, vector<cv::Rect_<float> >& face_detections)
 {
    // Go over the model and eliminate detections that are not informative (there already is a tracker there)
    for (size_t model = 0; model < clnf_models.size(); ++model)
    {

      // See if the detections intersect
      cv::Rect_<float> model_rect = clnf_models[model].GetBoundingBox();

      for (int detection = face_detections.size() - 1; detection >= 0; --detection)
      {
        double intersection_area = (model_rect & face_detections[detection]).area();
        double union_area = model_rect.area() + face_detections[detection].area() - 2 * intersection_area;

        // If the model is already tracking what we're detecting ignore the detection, this is determined by amount of overlap
        if (intersection_area / union_area > 0.5)
        {
          face_detections.erase(face_detections.begin() + detection);
        }
      }
    }
  }
}


namespace openface2_ros
{
  class OpenFace2Ros
  {
  public:
    OpenFace2Ros(NodeHandle &nh, FaceAnalysis::FaceAnalyserParameters &face_analysis_params)
      : nh_(nh)
      , it_(nh_)
      , visualizer(true, false, false, true)
      , face_analyser(face_analysis_params)
    {
      NodeHandle pnh("~");

      if(!pnh.getParam("image_topic", image_topic_)) pnh.param<string>("image_topic", image_topic_, "/realsense_face/color/image_raw");
      
      const auto base_path = package::getPath("openface2_ros");

      //pnh.param<bool>("publish_viz", publish_viz_, false);

      //if(!pnh.getParam("max_faces", max_faces_)) pnh.param<int>("max_faces", max_faces_, 4);
      //if(max_faces_ <= 0) throw invalid_argument("~max_faces must be > 0");
      max_faces_ = 1;

      //float rate = 0;
      //if(!pnh.getParam("rate", rate)) pnh.param<float>("rate", rate, 4.0);
      //if(rate <= 0) throw invalid_argument("~rate must be > 0");
      //rate_ = round(30/rate);

      camera_sub_ = it_.subscribeCamera(image_topic_, 1, &OpenFace2Ros::process_incoming_, this);
      aus_pub_ = nh_.advertise<openface2_ros_msgs::ActionUnits>("openface2/aus", 10);
      viz_pub_ = it_.advertise("openface2/image", 1);
      init_openface_();
    }
    
    ~OpenFace2Ros()
    {
    }
    
  private:
    void init_openface_()
    {
      	vector<string> arguments(1,"");
      	LandmarkDetector::FaceModelParameters det_params(arguments);
      	// This is so that the model would not try re-initialising itself
      	//det_params.reinit_video_every = -1;

      	det_params.curr_face_detector = LandmarkDetector::FaceModelParameters::MTCNN_DETECTOR;

      	det_parameters = det_params;

      	LandmarkDetector::CLNF face_model(det_parameters.model_location);

      	if (!face_model.loaded_successfully)
      	{
        	cout << "ERROR: Could not load the landmark detector" << endl;
      	}

      	// Loading the face detectors
      	face_model.face_detector_HAAR.load(det_parameters.haar_face_detector_location);
      	face_model.haar_face_detector_location = det_parameters.haar_face_detector_location;
      	face_model.face_detector_MTCNN.Read(det_parameters.mtcnn_face_detector_location);
      	face_model.mtcnn_face_detector_location = det_parameters.mtcnn_face_detector_location;

      	if (!face_model.eye_model)
      	{
        	cout << "WARNING: no eye model found" << endl;
      	}

        fps_tracker.AddFrame();

        ROS_INFO("OpenFace initialized!");
    }

    void process_incoming_(const sensor_msgs::ImageConstPtr &img, const sensor_msgs::CameraInfoConstPtr &cam)
    {
      if(viz_pub_.getNumSubscribers() > 0 ) {
        publish_viz_ = true;
      } else {
        publish_viz_ = false;
      }

      cv_bridge::CvImagePtr cv_ptr_rgb;
      cv_bridge::CvImagePtr cv_ptr_mono;
      try
      {
        cv_ptr_rgb = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::BGR8);
        cv_ptr_mono = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
      }
      catch(const cv_bridge::Exception &e)
      {
       	ROS_ERROR("cv_bridge exception: %s", e.what());
       	return;
      }

      double fx = cam->K[0];
      double fy = cam->K[4];
      double cx = cam->K[2];
      double cy = cam->K[5];


      if(fx == 0 || fy == 0)
      {
        fx = 500.0 * cv_ptr_rgb->image.cols / 640.0;
        fy = 500.0 * cv_ptr_rgb->image.rows / 480.0;
        fx = (fx + fy) / 2.0;
        fy = fx;
      }

      if(cx == 0) cx = cv_ptr_rgb->image.cols / 2.0;
      if(cy == 0) cy = cv_ptr_rgb->image.rows / 2.0;

      bool detection_success = LandmarkDetector::DetectLandmarksInVideo(cv_ptr_rgb->image, face_model, det_parameters, cv_ptr_mono->image);

      // Keeping track of FPS
      fps_tracker.AddFrame();

      decltype(cv_ptr_rgb->image) viz_img = cv_ptr_rgb->image.clone();
      if(publish_viz_) visualizer.SetImage(viz_img, fx, fy, cx, cy);

      openface2_ros_msgs::ActionUnits aus_msg;
      aus_msg.header.frame_id = img->header.frame_id;
      aus_msg.header.stamp = Time::now();
      if(detection_success) {

      face_analyser.PredictStaticAUsAndComputeFeatures(cv_ptr_rgb->image, face_model.detected_landmarks);

      auto aus_reg = face_analyser.GetCurrentAUsReg();
      auto aus_class = face_analyser.GetCurrentAUsClass();

      unordered_map<string, openface2_ros_msgs::ActionUnit> aus;
      for(const auto &au_reg : aus_reg)
      {
        auto it = aus.find(get<0>(au_reg));
        if(it == aus.end())
        {
          openface2_ros_msgs::ActionUnit u;
          u.name = get<0>(au_reg);
          u.intensity = get<1>(au_reg);
          aus.insert({ get<0>(au_reg), u});
          continue;
        }
        it->second.intensity = get<1>(au_reg);
      }

      for(const auto &au_class : aus_class)
      {
        auto it = aus.find(get<0>(au_class));
        if(it == aus.end())
        {
          openface2_ros_msgs::ActionUnit u;
          u.name = get<0>(au_class);
          u.presence = get<1>(au_class);
          aus.insert({ get<0>(au_class), u});
          continue;
        }
        it->second.presence = get<1>(au_class);
      }

      for(const auto &au : aus) aus_msg.action_units.push_back(get<1>(au));

      aus_pub_.publish(aus_msg);

      if (publish_viz_)
      {
        visualizer.SetObservationActionUnits(aus_reg, aus_class);
      }

      // we only publish faces if we have any. this way a simple rostopic hz reveals whether faces are found
      }

      if(publish_viz_)
      {
        visualizer.SetFps(fps_tracker.GetFPS());
        //visualizer.ShowObservation();
        auto viz_msg = cv_bridge::CvImage(img->header, "bgr8", visualizer.GetVisImage()).toImageMsg();
        viz_pub_.publish(viz_msg);
      }
    }
    
    tf2_ros::TransformBroadcaster tf_br_;
    
    // The modules that are being used for tracking
	  LandmarkDetector::CLNF face_model;
    LandmarkDetector::FaceModelParameters det_parameters;

    FaceAnalysis::FaceAnalyser face_analyser;
    Utilities::Visualizer visualizer;
    Utilities::FpsTracker fps_tracker;

    string image_topic_;
    int max_faces_;
    unsigned rate_;

    bool published_markers;
    NodeHandle nh_;
    image_transport::ImageTransport it_;
    image_transport::CameraSubscriber camera_sub_;
    Publisher aus_pub_;

    bool publish_viz_;
    image_transport::Publisher viz_pub_;
  };
}

int main(int argc, char *argv[])
{
  init(argc, argv, "openface2_ros");
  
  using namespace openface2_ros;

  NodeHandle nh;

  FaceAnalysis::FaceAnalyserParameters face_analysis_params;
	face_analysis_params.OptimizeForImages();

  try
  {
    OpenFace2Ros openface_(nh, face_analysis_params);
    MultiThreadedSpinner().spin();
  }
  catch(const exception &e)
  {
    ROS_FATAL("%s", e.what());
    return EXIT_FAILURE;
  }
  
  return EXIT_SUCCESS; 
}
