#include <curl/curl.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <istream>
#include <string>
#include "segmentation.hpp"
#include "filters.hpp"
#include "util.hpp"
#include "matrix_utils.hpp"
#include "aruco_marker.hpp"
#include "realsense2.hpp"
#include "realsense.hpp"
#include "curlserver.hpp"
#include "util/random_utils.h"
#include "loader.hpp"
#include "util2.hpp"
#include "surface_reconstruction.hpp"
#include "mesh_proc.hpp"
#include "cuda_headers.hpp"
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>
#include "configuration.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>

int main(int argc,char** argv)
{
  //    string data_path = "../data/2018-10-05-16-58-23/";
  string data_path = "src/cob_object_perception_experimental/data/2018-10-08-17-59-24/";
  Configuration config;
  Loader loader;
  Filters filters;
  Segmentation segmentation;
  MeshProc mp;
  Timer timer;

  std::vector<CameraParameters> cam_params,depth_cam_params;
  int num_devices;
  loader.readMultiCameraParameters(data_path+"/cam.info.txt",cam_params,depth_cam_params,num_devices);
  std::vector<std::vector<cv::Mat> >  color_image_vec(num_devices),depth_image_vec(num_devices);
  std::vector<std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>> cloud_vec(num_devices);
  std::vector<std::vector<Eigen::Matrix4f>> camera2marker(num_devices),depth2marker(num_devices);
  std::vector<std::vector<int>> pose_flags(num_devices);

  for (int i=0;i<num_devices;i++)
  {
    std::ostringstream s;           s<<i;
    readTransforms(data_path+"/transforms_"+s.str()+".yaml",camera2marker[i],pose_flags[i],cam_params[i].num_view);
    pose_flags[i].clear();
    readTransforms(data_path+"/depth_transform_"+s.str()+".yaml",depth2marker[i],pose_flags[i],depth_cam_params[i].num_view);
    loader.loadColorImg(color_image_vec[i],data_path,"input_image_"+s.str()+".txt");
    loader.loadDepthImg(depth_image_vec[i],data_path,"input_image_"+s.str()+".txt");
    loader.loadPointClouds(cloud_vec[i],data_path,"input_cloud_"+s.str()+".txt");
  }
  timer.tok_();
  //    visualize_raw_depth(depth_image_vec);
  //    correctRotations(camera2marker,pose_flags);
  //    correctRotations(depth2marker,pose_flags);

  //////////////////////////////////////////////////////////////////////////////////////////////////////////
  //        vector<vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>> filtered_cloud_vec;
  //        filtered_cloud_vec.resize(num_devices);
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

  //        pcl::PointCloud<pcl::PointXYZRGB>::Ptr output(new pcl::PointCloud<pcl::PointXYZRGB>);
  //        cout<<"Crop-box filter and remove outliers and fuse clouds..." <<endl;
  //        for(int i=0;i<filtered_cloud_vec.size();i++)
  //        {
  //            std::ostringstream s;           s<<i;
  //            for(int j=0;j<filtered_cloud_vec[i].size();j++)
  //            {
  //    //            filters.CropBox(filtered_cloud_vec[i][j],config.min_x_,config.min_y_,config.min_z_,config.max_x_,config.max_y_,config.max_z_);
  //                filters.BoxPassThrough(filtered_cloud_vec[i][j],config.min_x_,config.min_y_,config.min_z_,config.max_x_,config.max_y_,config.max_z_);
  //                std::ostringstream ss;                ss<<j;
  //                pcl::io::savePCDFile(data_path+"croped_cloud_"+s.str()+"-"+ss.str()+".pcd",*filtered_cloud_vec[i][j],true);
  ////                filters.removeOutliers(filtered_cloud_vec[i][j],0.7,50);//0.5,100
  //                if((filtered_cloud_vec[i][j]->points.size())<1) cout<<i<<"-"<<j<<endl;
  //                pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_rgb(new pcl::PointCloud<pcl::PointXYZRGB>);
  //                if (i==0) segmentation.cloudAddColor(filtered_cloud_vec[i][j],cloud_rgb,255,0,0);
  //                if (i==1) segmentation.cloudAddColor(filtered_cloud_vec[i][j],cloud_rgb,0,255,0);
  //                *output += * cloud_rgb;
  //            }
  //        }

  ////        cout<<"remove outliers again..." <<endl;
  //    //    filters.removeOutliers(output,1.0,50);//100
  ////        cout<<"number of points "<<output->points.size()<<endl;
  ////        cout<<"euclideanCluster..." <<endl;//keep big clusters
  ////        segmentation.euclideanCluster(output,output->size()/2,0.001,output);
  //        cout<<"number of points "<<output->points.size()<<endl;
  //        printf("Fused cloud is done \n");
  //        pcl::io::savePCDFile("fused_cloud.pcd",*output,true);
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

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
    pcl::io::savePolygonFilePLY(data_path+"mesh_filtered.ply",pclmesh);
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
  //    pcl::io::savePLYFile(data_path+"simplified_mesh_ascii.ply",pclmesh);
  //    std::cout<<"reproduce of mesh.."<<std::endl;
  //    mp.reproduceMesh(pclmesh);
  //    pcl::io::savePolygonFilePLY(data_path+"double_layers_mesh.ply",pclmesh);
  timer.tok_();

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
