//
// Created by prashant on 2/14/20.
//

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>

#include <geometry_msgs/TransformStamped.h>

#include <cv_bridge/cv_bridge.h>

#include <opencv2/aruco.hpp>
#include <opencv2/aruco/charuco.hpp>
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>

#include <tf2_ros/transform_broadcaster.h>
#include "tf2_geometry_msgs/tf2_geometry_msgs.h"
#include <tf2_eigen/tf2_eigen.h>
#include <eigen3/Eigen/Dense>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>



class CharucoBoard{

private:
    cv::Mat inImage, resultImage;
    cv::Mat cameraMatrix, distortionCoeffs;
    bool draw_markers;
    bool draw_axis;
    bool publish_tf;
    bool publish_corners;
    bool cam_info_received;

    int nMarkerDetectThreshold;
    int nMarkers;
    int dictionary_id;

    image_transport::Publisher image_pub;
    ros::Publisher transform_pub;
    ros::Publisher pose_pub;
    ros::Publisher corner_pub;
    ros::Subscriber cam_info_sub;

    double board_scale;
    std::vector< cv::Mat > idcorners;
    std::vector< cv::Mat > idcornerspx;
    std::vector< float> board_ids;

    ros::NodeHandle nh;
    image_transport::ImageTransport it;
    image_transport::Subscriber image_sub;

    cv::Mat board;
    cv::Ptr<cv::aruco::Dictionary> dictionary;
    cv::Ptr<cv::aruco::DetectorParameters> detectorParams;
    std::ostringstream vector_to_marker;

    int marker_id;
    int borderBits;
    int sidePixels;
    float markerLengthMeters;


public:

    CharucoBoard() : cam_info_received(false),
    nh("~"),
    it(nh),
    nMarkerDetectThreshold(0)    {

        image_sub = it.subscribe("/image", 1, &CharucoBoard::image_callback, this);
        cam_info_sub = nh.subscribe("/camera_info", 1, &CharucoBoard::cam_info_callback, this);

        image_pub = it.advertise("result", 1);
        transform_pub = nh.advertise<geometry_msgs::TransformStamped>("transform", 100);
        pose_pub = nh.advertise<geometry_msgs::PoseStamped>("pose", 100);



        nh.param<int>("dictionary_id", dictionary_id, 0);
        nh.param<int>("marker_id", marker_id, 1);
        nh.param<int>("borderBits", borderBits, 3);
        nh.param<int>("sidePixels", sidePixels, 30);
        nh.param<float>("markerLengthMeters", markerLengthMeters, 0.18);

        nh.param<bool>("draw_markers", draw_markers, true);
        nh.param<bool>("draw_axis", draw_axis, true);
        nh.param<bool>("publish_tf", publish_tf, false);
        nh.param<bool>("publish_corners", publish_corners, true);

        ROS_INFO_STREAM("Initializing dictionary id " <<dictionary_id << " marker id " << marker_id <<"Marker length "<< markerLengthMeters<<" borderBits " << borderBits << " sidePixels " << sidePixels );

        dictionary = cv::aruco::getPredefinedDictionary(dictionary_id);
        detectorParams = cv::aruco::DetectorParameters::create();
        cv::aruco::drawMarker(dictionary,marker_id,sidePixels,board,borderBits);
        detectorParams->doCornerRefinement = false;

        ROS_INFO_STREAM("Image of the aruco marker have been printed as ~/test.jpg" );
        cv::imwrite("/home/prashant/test.jpg", board);
        detectorParams->cornerRefinementMaxIterations = 30;
    }

    void image_callback(const sensor_msgs::ImageConstPtr& msg) {

        if(!cam_info_received) return;

        static tf2_ros::TransformBroadcaster br;

        cv_bridge::CvImagePtr cv_ptr;
        try {
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::RGB8);
            inImage = cv_ptr->image;

            std::vector< int > ids;
            std::vector< std::vector< cv::Point2f > > corners, rejected;
            cv::Vec3d tvec(0, 0, 1);
            cv::Vec3d rvec(0, 0, 0);
//            cv::Mat guessRotMat = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
//            cv::Rodrigues(guessRotMat, rvec);
            // detect markers
            cv::aruco::detectMarkers(inImage, dictionary, corners, ids, detectorParams, rejected);

            if (ids.size() < 1)
                return;

            cv::aruco::drawDetectedMarkers(inImage, corners, ids);
            std::vector<cv::Vec3d> rvecs, tvecs;
            cv::aruco::estimatePoseSingleMarkers(corners, markerLengthMeters,cameraMatrix, distortionCoeffs, rvecs, tvecs);

            Eigen::Affine3d transform_result;
            getTF(rvecs[0], tvecs[0], transform_result);

            geometry_msgs::TransformStamped transform_msg;
            transform_msg = tf2::eigenToTransform(transform_result);
            transform_msg.header.stamp = msg->header.stamp;
            transform_msg.header.frame_id = "aruco_marker";
            transform_msg.child_frame_id = msg->header.frame_id;
            transform_pub.publish(transform_msg);

            if (publish_tf)
                br.sendTransform(transform_msg);

            geometry_msgs::PoseStamped poseMsg;
            poseMsg.pose = tf2::toMsg(transform_result);
            poseMsg.header = msg->header;
            pose_pub.publish(poseMsg);

            resultImage = cv_ptr->image.clone();
            if (draw_markers)
                cv::aruco::drawDetectedMarkers(resultImage, corners);

            if (draw_axis)
                cv::aruco::drawAxis(resultImage, cameraMatrix, distortionCoeffs, rvecs, tvecs, 2*markerLengthMeters);

            if(draw_axis || draw_markers) {
                if (image_pub.getNumSubscribers() > 0) {
                    //show input with augmented information
                    cv_bridge::CvImage out_msg;
                    out_msg.header.frame_id = msg->header.frame_id;
                    out_msg.header.stamp = msg->header.stamp;
                    out_msg.encoding = sensor_msgs::image_encodings::RGB8;
                    out_msg.image = resultImage;
                    image_pub.publish(out_msg.toImageMsg());
                }
            }


        } catch (cv_bridge::Exception& e) {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }
    }

    void cam_info_callback(const sensor_msgs::CameraInfo &msg) {

        if (msg.K[0] == 0) {
            std::cout << msg << std::endl;
            ROS_ERROR("Camera Info message is zero --> Cannot use an uncalibrated camera!");
            return;
        }

        cameraMatrix = cv::Mat::zeros(3, 3, CV_32FC1);
        distortionCoeffs = cv::Mat::zeros(5, 1, CV_32FC1);

        for (int i = 0; i < 9; ++i)
            cameraMatrix.at<float>(i / 3, i % 3) = msg.K[i];

        for (int i = 0; i < 5; ++i)
            distortionCoeffs.at<float>(i, 0) = msg.D[i];

        cam_info_received = true;
        cam_info_sub.shutdown();
    }



    void getTF(const cv::Vec3d &rvec, const cv::Vec3d &tvec, Eigen::Affine3d &board_to_camera){

        cv::Mat rot;
        cv::Rodrigues(rvec, rot);

        board_to_camera.linear() << rot.at<double>(0, 0), rot.at<double>(0, 1), rot.at<double>(0, 2),
                             rot.at<double>(1, 0), rot.at<double>(1, 1), rot.at<double>(1, 2),
                             rot.at<double>(2, 0), rot.at<double>(2, 1), rot.at<double>(2, 2);
        board_to_camera.translation() << tvec(0) , tvec(1), tvec(2) ;

    }
};


int main(int argc, char **argv) {
    ros::init(argc, argv, "charuco_ros");

    CharucoBoard node;

    ros::spin();
}
