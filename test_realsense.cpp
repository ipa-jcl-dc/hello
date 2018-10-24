//C++
#include <curl/curl.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <istream>
#include <string>
#include <unordered_set>
#include <set>
//PCL
#include <pcl/common/common_headers.h>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/obj_io.h>
#include <pcl/io/vtk_io.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <boost/thread/thread.hpp>
#include <pcl/console/parse.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <pcl/common/transforms.h>
#include <pcl/features/board.h>
#include <pcl/console/parse.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>
#include <pcl/registration/transformation_estimation_point_to_plane.h>
#include <pcl/surface/impl/mls.hpp>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/surface/marching_cubes_rbf.h>
#include <pcl/features/normal_3d.h>
#include <pcl/surface/gp3.h>
#include <pcl/surface/vtk_smoothing/vtk_mesh_smoothing_laplacian.h>
#include <pcl/surface/marching_cubes_rbf.h>

#include "filters.hpp"
#include "segmentation.hpp"
#include "loader.hpp"
#include "aruco_marker.hpp"
#include "matrix_utils.hpp"
#include "realsense2.hpp"
#include "configuration.hpp"
#include "util/random_utils.h"
#include "texturem.hpp"
#include "util2.hpp"
#include "mesh_proc.hpp"
#include "realsense.hpp"

int
main (int argc, char** argv)
{ 
  //    const char* const input_file_name = "testure_mesh_opt_ver.obj";
  //    const char* const output_file_name = "round.obj" ;
  //    // to write back into the same file, use the same file name for both input and output

  //    std::vector<std::string> all_the_lines, new_lines ;
  //    {
  //        // Open the file for input, read the lines in the file into a vector; close the file.
  //        std::ifstream file(input_file_name) ;
  //        std::string line ;
  //        while( std::getline( file, line ) ) all_the_lines.push_back(line) ;
  //    }

  //    {
  //        // Modify the contents of the vector
  //        for( std::string& line : all_the_lines ) //std::reverse( std::begin(line), std::end(line) ) ;
  //        {
  //            stringstream s(line); // Used for breaking words
  //            string word; // to store individual words
  //            s>>word;
  //            if (word.compare("f")==0)
  //            {
  //                string w1,w2,w3;
  //                s>>w1;s>>w2;s>>w3;
  //                stringstream ss;
  //                ss<< "f "<<w1<<" "<<w3<<" "<<w2;
  //                string str;
  //                str=ss.str();
  //                line=str;
  ////                new_lines.push_back(str);
  //            }
  ////            else new_lines.push_back(line);

  //        }
  //    }

  //    {
  //        // Open the file for output, write the contents of the vector into the file; close the file.
  //        std::ofstream file(output_file_name) ;
  //        for( const std::string& line : all_the_lines ) file << line << '\n' ;
  ////        for( const std::string& line : new_lines ) file << line << '\n' ;
  //    }

  //-------------------------------------------------------------------------------------------------------------

  //      cv::Ptr<cv::aruco::Dictionary> dictionary_ = cv::aruco::getPredefinedDictionary(cv::aruco::PREDEFINED_DICTIONARY_NAME(cv::aruco::DICT_4X4_1000));

  //      int num_squares=30;
  //      cv::Size image_size;
  //      int margins = 50;
  //      image_size.width = num_squares * 100 + margins*2;
  //      image_size.height = num_squares * 100 + margins*2;

  //      cv::Ptr<cv::aruco::CharucoBoard> charuco_board_ = cv::aruco::CharucoBoard ::create(num_squares,num_squares,0.02,0.02*0.6,dictionary_);

  //      cv::Mat board_image;
  //      charuco_board_->draw(image_size, board_image,margins,1);
  //      cv::imwrite("charuco.png",board_image);

  //-------------------------------------------------------------------------------------------------------------

//  cv::Ptr<cv::aruco::Dictionary> dictionary_ = cv::aruco::getPredefinedDictionary(cv::aruco::PREDEFINED_DICTIONARY_NAME(cv::aruco::DICT_4X4_1000));

//  int num_squares_x=7,num_squares_y=10;
//  cv::Size image_size;
//  int margins = 30;
//  image_size.width = num_squares_x * 100 + margins*2;
//  image_size.height = num_squares_y * 100 + margins*2;

//  cv::Ptr<cv::aruco::CharucoBoard> charuco_board_ = cv::aruco::CharucoBoard ::create(num_squares_x,num_squares_y,0.02,0.02*0.6,dictionary_);

//  cv::Mat board_image;
//  charuco_board_->draw(image_size, board_image,margins,1);
//  cv::imwrite("charuco_a4.png",board_image);

  //-------------------------------------------------------------------------------------------------------------

  std::string charuco_file;
  charuco_file = "src/cob_object_perception_experimental/config/charuco_board.xml";
  boost::shared_ptr<Marker> aruco_marker = boost::make_shared<Marker>();

  RealSense realsense( 480, 640, 1080, 1920);
  // Get list of cameras
  rs2::device_list devices = realsense.get_device_list();
  //Enable all camera and store serial numbers for further processing
  for (rs2::device device : devices)
  {
    realsense.enable_device(device);
  }

  while(true)
  {
    realsense.pollRawFrames();
    for(int i=0;i</*realsense.devices_.size()*/1;i++)
    {
      cv::Mat color_image;
      realsense.getColorImage(realsense.devices_[i],color_image);

      aruco_marker->LoadParameters(charuco_file);
      aruco_marker->createChArucoMarker();

      aruco_marker->setCameraMatrix(realsense.getColorCameraMatrix(realsense.devices_[i]));
      // wait several frames for the effect to take place

      aruco_pose pose;
      int pose_flag = aruco_marker->estimatePoseCharuco(color_image,pose);
    }
  }
  cv::destroyAllWindows();
  realsense.stopAll();
  cout<<"data collection finished"<<endl;

  return (0);
}
