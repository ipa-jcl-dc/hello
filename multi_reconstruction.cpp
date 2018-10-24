//C/C++
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <cstdlib>
#include <unistd.h>
#include <sstream>
#include <stdlib.h>
#include <vector>
#include <thread>
#include <chrono>
#include <boost/filesystem.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
//CUDA
#include "cuda_headers.hpp"
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
//OpenCV
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>
#include <eigen3/Eigen/Dense>
//Header files
#include "curlserver.hpp"
#include "util.hpp"
#include "registrator.hpp"
#include "filters.hpp"
#include "segmentation.hpp"
#include "loader.hpp"
#include "aruco_marker.hpp"
#include "matrix_utils.hpp"
#include "realsense.hpp"
#include "configuration.hpp"
#include "util/random_utils.h"
#include "texturem.hpp"
#include "data_types.hpp"
#include "util2.hpp"
#include "mesh_proc.hpp"
#include "surface_reconstruction.hpp"

//Declare helpers globally
boost::shared_ptr<Webserver2> webserver = boost::make_shared<Webserver2>();
boost::shared_ptr<Marker> aruco_marker = boost::make_shared<Marker>();
Filters filters;
Segmentation segmentation;
MeshProc mp;

//Declare some variables globally
std::vector<std::vector<cv::Mat> >  color_image_vec;
std::vector<std::vector<cv::Mat> > depth_image_vec;
std::vector<std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>> cloud_vec; //vector to store pointclouds
std::string charuco_file;
//cv::Mat camera_matrix;
std::vector<std::vector<Eigen::Matrix4f>> camera2marker; //vector stores matrix from camera to board
std::vector<std::vector<int>> pose_flags; //vector stores successful found pose of aruco board(0:not found, 1: found)
std::string data_path;

void collect360Degrees(const int num_captures,RealSense realsense)
{
  Timer log;
  aruco_marker->LoadParameters(charuco_file);
  aruco_marker->createChArucoMarker();
  webserver->rotateDegRel(360);
  //Turn off all lasers
  for (auto&& view : realsense.devices_)
    realsense.turnOffLaser(view);

  auto now=std::chrono::system_clock::now();
  int64_t wait_time=28000000/(num_captures);//27000000
  for(int n=0;n<num_captures;n++)
  {
    std::ostringstream ss;      ss<<n;
    //        auto now=std::chrono::system_clock::now();
    Timer timer;
    int i=0;
    for (auto&& view : realsense.devices_)// i th camera
    {
      std::cout<<"view "<<n<<", camera "<<i<<endl;
      //turn on the laser to be used
      realsense.turnOnLaser(view);
      //Set intrinsic parameters of each camera
      aruco_marker->setCameraMatrix(realsense.getColorCameraMatrix(view));
      // wait several frames for the effect to take place
      for(int t=0;t<3; t++)realsense.pollRawFrames();
      cv::Mat color_image,depth_image;

      realsense.getRectifiedDepthImage(view,depth_image);
      cv::Scalar mean = cv::mean( depth_image);
      while (mean[0]<1)
      {
        realsense.pollRawFrames();
        realsense.getRectifiedDepthImage(view,depth_image);
        mean = cv::mean( depth_image);
      }

      pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
      realsense.getRawPointCloud(view,cloud);
      realsense.getColorImage(view,color_image);
      //            realsense.getRectifiedDepthImage(view,depth_image);
      cloud_vec[i].push_back(cloud);
      color_image_vec[i].push_back(color_image.clone());
      depth_image_vec[i].push_back(depth_image.clone());
      aruco_pose pose;
      int pose_flag = aruco_marker->estimatePoseCharuco(color_image,pose);
      Eigen::Vector3f T;
      Eigen::Matrix3f R;
      Eigen::Matrix4f marker_to_camera, camera_to_marker;
      marker_to_camera.setIdentity();   // Set to Identity to make bottom row of Matrix 0,0,0,1
      //Only convert to eigen matrix with valid pose
      if(pose_flag==1)
      {
        cv::cv2eigen(pose.rot,R);
        cv::cv2eigen(pose.trans,T);
        marker_to_camera.block<3,3>(0,0) = R;
        marker_to_camera.block<3,1>(0,3) = T;
      }
      else  std::cerr<<"cannot get camera pose for this view"<<std::endl;
      camera_to_marker = marker_to_camera.inverse();
      pose_flags[i].push_back(pose_flag);
      camera2marker[i].push_back(camera_to_marker);
      //turn off the laser used
      realsense.turnOffLaser(view);
      std::ostringstream s; s<<i;
      cv::imwrite(data_path+"color-frame-"+s.str()+"-"+ss.str()+".png",color_image_vec[i][n]);
      cv::imwrite(data_path+"depth-frame-"+s.str()+"-"+ss.str()+".png",depth_image_vec[i][n]);
      pcl::io::savePCDFile(data_path+"cloud_"+s.str()+"-"+ss.str()+".pcd",*cloud_vec[i][n],true);
      i++;
    }

    cv::imshow("first camera",color_image_vec[0].back());
    cv::waitKey(1);

    timer.tok_();
    //        std::this_thread::sleep_until(now + std::chrono::microseconds(wait_time));
    std::this_thread::sleep_until(now + std::chrono::microseconds(wait_time*(n+1)));
  }
  log.tok_();
}

int main(int argc, char *argv[])
{
  //ChAruco board parameters
  //    charuco_file = "../config/charuco_board.xml";
  charuco_file = "src/cob_object_perception_experimental/config/charuco_board.xml";
  Configuration config;
  Timer timer;
  //Set angular speep for station, max 14.5, default 12
  webserver->setAngularSpeed(14.0);
  data_path = "../data/" + currentDateTime() + "/";
  int mkdir=system(("mkdir -p " + data_path).c_str());

  RealSense realsense( 480, 640, 1080, 1920);
  // Get list of cameras
  rs2::device_list devices = realsense.get_device_list();
  //Enable all camera and store serial numbers for further processing
  for (rs2::device device : devices)
  {
    realsense.enable_device(device);
    //            serial_numbers.push_back(realsense.get_serial_number(device));
  }

  //Resize data vector = number of connected cameras
  cloud_vec.resize(realsense.devices_.size());
  color_image_vec.resize(realsense.devices_.size());
  depth_image_vec.resize(realsense.devices_.size());
  camera2marker.resize(realsense.devices_.size());
  pose_flags.resize(realsense.devices_.size());

  //    realsense.setConfiguration("../config/realsense_parameters.yaml");
  //    realsense.getConfiguration();
  std::printf("To collect data, press Enter \n Press ESC to exit \n");
  char c =0;
  while((c=(char)cv::waitKey(1)) !=27)//ESC
  {
    realsense.pollRawFrames();
    //Visualize streams
    for(int i=0;i<realsense.devices_.size();i++)
    {
      cv::Mat img;
      realsense.getColorImage(realsense.devices_[i],img);
      std::ostringstream ss; ss<<i;
      cv::imshow("color "+ss.str(),img);
    }
    //Collect data
    if(c==13) //Enter
    {
      cv::destroyAllWindows();
      //Continiously collect
      collect360Degrees(config.n_capture_,realsense);
      break;
    }
  }
  if(c==27) return 0;
  cv::destroyAllWindows();
  realsense.stopAll();
  cout<<"data collection finished"<<endl;

  std::cout<<"checking abnormal extrinsics.."<<std::endl;
  for(int i=0;i<realsense.devices_.size();i++){
    removeZOutliers(camera2marker[i],pose_flags[i]);
  }
  std::vector<std::vector<Eigen::Matrix4f>> camera2marker_original = camera2marker;
  //    if(config.plane_equalized_)//plane_equalized_==0
  //    {
  //        //Do least square 3d plane-equalization, modify transformation matrices
  //        Eigen::Vector3f plane_coeffs;
  //        plane_coeffs =leastSquare3d(camera2marker,pose_flags);
  //        //Transform translation
  //        projectTransformTranslationsToPlane(plane_coeffs,camera2marker,pose_flags); //project the translations onto the plane
  //    }
    timer.tok_();

  std::vector<std::vector<Eigen::Matrix4f>> depth2marker; //vector stores matrix from depth camera to board
  depth2marker.resize(camera2marker.size());
  std::vector<CameraParameters> cam_params,depth_cam_params;
  for (int i=0;i<realsense.devices_.size();i++)
  {
    cam_params.push_back(realsense.getColorParameters(realsense.devices_[i]));
    depth_cam_params.push_back(realsense.getDepthParameters(realsense.devices_[i]));
    Eigen::Matrix4f depth_to_color = realsense.getExtrinsics(realsense.devices_[i]);
    depth2marker[i].resize(camera2marker[i].size());
    for(int j=0;j<camera2marker[i].size();j++)
      depth2marker[i][j] = camera2marker[i][j]*depth_to_color;
  }

  if(config.debug_)
  {
    std::cout<<"saving raw data.."<<std::endl;
#pragma omp parallel for schedule(dynamic)
    for(int i=0;i<realsense.devices_.size();i++)
    {
      std::ostringstream s;           s<<i;
      std::string cam_info_file = data_path + "cam.info.txt";
      realsense.saveCameraParameters(cam_info_file,config.n_capture_);
      writeTransformsYml(data_path+"transforms_"+s.str()+".yaml",camera2marker_original[i],pose_flags[i]);
      writeTransformsTxt(data_path+"transforms_"+s.str()+".txt",camera2marker[i],pose_flags[i]);
      //            writeTransformsYml(data_path+"transform_plane_equalization.yaml",camera2marker,pose_flags);
      writeTransformsYml(data_path+"depth_transform_"+s.str()+".yaml",depth2marker[i],pose_flags[i]);
      std::vector<std::string> vector_img_name,vector_cloud_name;
      for(int k=0;k<color_image_vec[i].size();k++)
      {
        std::ostringstream ss;                ss<<k;
        std::string img_name,cloud_name;
        img_name = "frame-"+s.str()+"-"+ss.str()+".png";
        cloud_name = s.str()+"-"+ss.str()+".pcd";
        vector_img_name.push_back(img_name);
        vector_cloud_name.push_back(cloud_name);
      }
      saveNameTxt(data_path+"input_image_"+s.str()+".txt",vector_img_name);
      saveNameTxt(data_path+"input_cloud_"+s.str()+".txt",vector_cloud_name);
    }
    cout<<"data is save in folder "<<data_path<<endl;
    timer.tok_();
  }//end saving data

  //        vector<vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>> filtered_cloud_vec;
  //        filtered_cloud_vec.resize(realsense.devices_.size());
  //        cout<<"Transform points from camera to marker coordinate"<<endl;
  //        for(int i=0;i<cloud_vec.size();i++)
  //        {
  //            for(int k=0;k<cloud_vec[i].size();k++)
  //            {
  //                //Only tranform cloud with valid pose
  //                if(pose_flags[i][k]==1)
  //                {
  //                    pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  //                    pcl::transformPointCloud(*cloud_vec[i][k],*transformed_cloud,depth2marker[i][k]);
  //                    filtered_cloud_vec[i].push_back(transformed_cloud);
  //                }
  //            }
  //            cout<<"number of transformed clouds="<<filtered_cloud_vec[i].size()<<endl;
  //        }
  //        timer.tok_();

  //        pcl::PointCloud<pcl::PointXYZRGB>::Ptr output(new pcl::PointCloud<pcl::PointXYZRGB>); // fused cloud
  //        cout<<"Crop-box filter and remove outliers and fuse clouds..." <<endl;
  //        for(int i=0;i<filtered_cloud_vec.size();i++)
  //        {
  //            std::ostringstream s;           s<<i;
  //            for(int j=0;j<filtered_cloud_vec[i].size();j++)
  //            {
  //                filters.CropBox(filtered_cloud_vec[i][j],config.min_x_,config.min_y_,config.min_z_,config.max_x_,config.max_y_,config.max_z_);
  //                std::ostringstream ss;                ss<<j;
  //                pcl::io::savePCDFile(data_path+"croped_cloud_"+s.str()+"-"+ss.str()+".pcd",*filtered_cloud_vec[i][j],true);
  ////                filters.removeOutliers(filtered_cloud_vec[i][j],0.6,60);//100
  ////                segmentation.euclideanCluster(filtered_cloud_vec[i][j],filtered_cloud_vec[i][j]->size()/3,0.001,filtered_cloud_vec[i][j]);
  //                if((filtered_cloud_vec[i][j]->points.size())<1) cout<<"???????????"<<i<<"-"<<j<<endl;
  //                pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_rgb(new pcl::PointCloud<pcl::PointXYZRGB>);
  //                if (i==0) segmentation.cloudAddColor(filtered_cloud_vec[i][j],cloud_rgb,255,0,0);
  //                if (i==1) segmentation.cloudAddColor(filtered_cloud_vec[i][j],cloud_rgb,0,255,0);
  //                *output += * cloud_rgb;
  //            }
  //        }

  ////        cout<<"remove outliers again..." <<endl;
  ////        filters.removeOutliers(output,1.0,60);//100
  ////        cout<<"number of points "<<output->points.size()<<endl;
  ////        cout<<"euclideanCluster..." <<endl;//keep big clusters
  ////        segmentation.euclideanCluster(output,output->size()/3,0.001,output);
  //        cout<<"number of points "<<output->points.size()<<endl;
  //        printf("Fused cloud is done \n");

  //        //    utils->bbox3DPCA(output);
  //        //    utils->bbox3DInertia(output);
  //        pcl::io::savePCDFile(data_path+"fused_cloud.pcd",*output,true);

  int num_devices=realsense.devices_.size();
  pcl::PolygonMesh pclmesh;
  int method=1;//1=mc,2=poisson

  if (method==2){
    cout<<"tsdf+poisson..."<<endl;
    SurfaceRecontruction sr(config,depth2marker[0][0]);
    VolumeData volume(config.volume_resolution_,config.voxel_size_);
    for (int i=0;i<num_devices;i++)
    {
      sr.volumeItegration(volume,depth_image_vec[i],depth2marker[i],
                          pose_flags[i],depth_cam_params[i],i);
    }
    PointCloud cloud_tsdf = Cuda::hostExtractPointCloud(volume,config.pointcloud_buffer_size_,
                                                        config.original_distance_x_,config.original_distance_y_,config.original_distance_z_);
    volume.release();
    //    export_ply(data_path+"tsdf.ply",cloud_tsdf);
    //TSDF cloud to PCL format cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    saveTSDFCloudPCD(data_path + "/tsdf.pcd",cloud_tsdf,cloud );

    //Transform points to marker coordinate
    pcl::transformPointCloud(*cloud,*cloud,depth2marker[0][0]);
    std::cout<<"Crop points that are outside of 3D bounding box.."<<std::endl;
    filters.BoxPassThrough(cloud,config.min_x_,config.min_y_,config.min_z_,
                           config.max_x_,config.max_y_,config.max_z_);
    pcl::io::savePCDFile(data_path+"cropbox.pcd",*cloud,true);

    timer.tok_();
    std::cout<<"remove small clusters with euclideanCluster.."<<std::endl;
    segmentation.euclideanCluster(cloud,cloud->size()/2,config.voxel_size_*1.5f,cloud);
    pcl::io::savePCDFile(data_path+"euclideanCluster.pcd",*cloud,true);
    std::cout<<"smooth cloud with movingLeastSquares.."<<std::endl;
    cloud = filters.movingLeastSquares(cloud,config.mls_);
    pcl::io::savePCDFile(data_path+"movingLeastSquares.pcd",*cloud,true);
    std::cout<<"Close bottom of cloud"<<std::endl;
    segmentation.closeCloud(cloud,config.voxel_size_/2);
    pcl::io::savePCDFile(data_path+"closed_cloud.pcd",*cloud,true);
    std::cout<<"generate mesh.."<<std::endl;
    pclmesh = mp.poissonReconstruction(cloud,config.voxel_size_*2);
    pcl::io::savePolygonFilePLY(data_path+"poisson_mesh.ply",pclmesh);
    timer.tok_();
  }

  if(method==1){
    cout<<"tsdf+mc..."<<endl;
    SurfaceRecontruction sr(config,depth2marker[0][0]);
    VolumeData volume(config.volume_resolution_,config.voxel_size_);
    for (int i=0;i<num_devices;i++)
    {
      sr.volumeItegration(volume,depth_image_vec[i],depth2marker[i],
                          pose_flags[i],depth_cam_params[i],i);
    }
    sr.extractTSDF(volume);
    volume.release();
    sr.extractPclMesh(pclmesh);
    timer.tok_();
    cout<<"saving mesh from mc..."<<endl;
    pcl::io::savePolygonFilePLY(data_path+"mesh_mc.ply",pclmesh);
    cout<<"filter out isolated meshes..."<<endl;
    mp.meshFilter(pclmesh,config.voxel_size_*2/*1.4*/);
    //        pcl::io::savePolygonFilePLY(data_path+"mesh_filtered.ply",pclmesh);
    //        cout<<"generate bottom surface..."<<endl;
    //        mp.getBottomMesh(pclmesh,config.voxel_size_/2);
    //        pcl::io::savePolygonFilePLY(data_path+"mesh_bottom.ply",pclmesh);
    timer.tok_();
  }

  std::cout<<"Laplacian Smoothing of mesh.."<<std::endl;
  mp.meshSmoothingLaplacian(pclmesh,config.relaxion_);
  pcl::io::savePolygonFilePLY(data_path+"mesh_smoothed.ply",pclmesh);
  timer.tok_();
  std::cout<<"simplication of mesh.."<<std::endl;
  mp.meshSimplication(pclmesh,config.simplified_percent_);
  pcl::io::savePolygonFilePLY(data_path+"simplified_mesh.ply",pclmesh);
  timer.tok_();
  //    std::cout<<"reproduce of mesh.."<<std::endl;
  //    mp.reproduceMesh(pclmesh);
  //    pcl::io::savePolygonFilePLY(data_path+"double_layers_mesh.ply",pclmesh);

  std::cout<<"Texture Mapping.."<<std::endl;
  TextureMapping tm;
  tm.initializeMesh(pclmesh);
  PCL_INFO ("Loading textures and camera poses...\n");
  tm.readCamParam(data_path,cam_params,camera2marker,pose_flags);
  //    cout<<"Create materials for each texture image and one extra for occluded faces"<<endl;
  //    tm.createMaterials();
  PCL_INFO ("Sorting faces by cameras...\n");
  tm.viewSelection();
  //    tm.textureMeshwithMultipleCameras();
  tm.textureMeshWithAtlas();
  std::cout<<"createAtlasMaterials .."<<std::endl;
  tm.createAtlasMaterials();
  tm.shiftXYOrigin_addVerticesNormal();
  PCL_INFO ("Saving textured mesh..\n");
  pcl::io::saveOBJFile(data_path+"textured_atlas.obj",tm.output_mesh,5);
  timer.tok_();

  if(config.optimization_){
    cout<<"perform optimization.."<<endl;
    tm.grayAndGradient();
    timer.tok_();
    //    tm.optimization();
    tm.poseOptimization();
    timer.tok_();
    tm.viewSelection();
    tm.textureMeshWithAtlas();
    tm.createAtlasMaterials();
    tm.shiftXYOrigin_addVerticesNormal();
    PCL_INFO ("Saving optimised textured mesh..\n");
    pcl::io::saveOBJFile(data_path+"textured_atlas_opt.obj",tm.output_mesh);
    timer.tok_();
  }


  return 0;
}
