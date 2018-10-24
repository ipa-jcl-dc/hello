#include "surface_reconstruction.hpp"
#include "marching_cubes_table.hpp"
#include <cmath>
#include <utility>

SurfaceRecontruction::SurfaceRecontruction(const Configuration &config,Eigen::Matrix4f& base2world):
  config_(config),base2world_(base2world),
  origin_(config.original_distance_x_,config.original_distance_y_,config.original_distance_z_)
{}

void SurfaceRecontruction::volumeItegration(VolumeData& volume, std::vector<cv::Mat>& depth_images,
                                            const std::vector<Eigen::Matrix4f>& transforms,
                                            const std::vector<int>& pose_flags,
                                            CameraParameters cam_params, int index)
{
  bool debug=false;
  std::ostringstream s;    s<<index;
  for(auto i=0;i<depth_images.size();i++)
  {
    std::ostringstream ss;    ss<<i;
    if(pose_flags[i] ==0) continue;
    std::cout<<"fusing "<<index<<"-"<<i<<std::endl;
    cv::cuda::GpuMat gpu_depth_frame(depth_images[i]);
    //        cv::cuda::GpuMat gpu_vertex_map = cv::cuda::createContinuous(cam_params.image_height,cam_params.image_width,CV_32FC3);
    //        cv::cuda::GpuMat gpu_normal_map = cv::cuda::createContinuous(cam_params.image_height,cam_params.image_width,CV_32FC3);
    cv::cuda::GpuMat gpu_diff_map = cv::cuda::createContinuous(cam_params.image_height,cam_params.image_width,CV_32FC1);

    //        Cuda::hostVertexMap(gpu_depth_frame,gpu_vertex_map,config_.depth_cutoff_,cam_params);
    //        gpu_normal_map.setTo(0.f); Cuda::hostNormalMap(gpu_vertex_map,gpu_normal_map);

    if (config_.weight_mode==0) gpu_diff_map.setTo(1.f);
    else if (config_.weight_mode==1) {
      gpu_diff_map.setTo(0.f);
      Cuda::hostDiffMap(gpu_depth_frame,gpu_diff_map,cam_params.depth_scale,config_.discontinuity_kernel);
    }
    else abort();

    cv::Mat cpu_diff_map; gpu_diff_map.download(cpu_diff_map);
    cpu_diff_map=cpu_diff_map/100;
    if (debug) cv::imwrite("diff_map-"+s.str()+"-"+ss.str()+".png",cpu_diff_map);

    //        bool filter_depth_img = false;
    //        if (filter_depth_img)
    //        {
    //            Eigen::Matrix3f rvect = transforms[i].block(0,0,3,3);
    //            Eigen::Vector3f tvect = transforms[i].block(0,3,3,1);
    //            //Min max threshold to crop depth img
    //            Eigen::Vector3f min_threshold(config_.min_x_,config_.min_y_,config_.min_z_);
    //            Eigen::Vector3f max_threshold(config_.max_x_,config_.max_y_,config_.max_z_);
    //            cv::cuda::GpuMat gpu_mask = cv::cuda::createContinuous(cam_params.image_height,cam_params.image_width,CV_8UC1);
    //            Cuda::hostTransformAndCropPoints(gpu_vertex_map,gpu_mask,rvect,tvect,min_threshold,max_threshold);
    //            cv::Mat cpu_mask; gpu_mask.download(cpu_mask);
    //            cv::imwrite("mask-"+s.str()+"-"+ss.str()+".png",cpu_mask);
    //            Cuda::hostSegmentedDepth(gpu_mask,gpu_depth_frame);
    //            gpu_mask.release();
    //        }

    Eigen::Matrix4f cam2base,cam2world,world2base;
    world2base=base2world_.inverse();
    cam2world=transforms[i];
    cam2base=world2base*cam2world;
    cv::Mat cam2base_cv = cv::Mat(cv::Size(4,4),CV_32FC1);
    cv::eigen2cv(cam2base,cam2base_cv);
    cv::cuda::GpuMat gpu_cam2base;
    //gpu_cam2base = cv::cuda::createContinuous(4,4,CV_32FC1);
    gpu_cam2base.upload(cam2base_cv);
    Cuda::hostTSDFVolume(gpu_depth_frame,volume,
                         cam_params,gpu_cam2base,config_.truncation_distance_,config_.depth_cutoff_,
                         config_.original_distance_x_,config_.original_distance_y_,config_.original_distance_z_,
                         cam_params.depth_scale,/*gpu_normal_map*/gpu_diff_map);

    //        gpu_vertex_map.release();
    gpu_cam2base.release();
    gpu_depth_frame.release();
    //        gpu_normal_map.release();
  }
}

//extract tsdf and weights from gpu volum to cpu vector
void SurfaceRecontruction::extractTSDF(const VolumeData &volume)
{
  cv::Mat tsdf_mat,weight_mat;
  volume.tsdf_volume.download(tsdf_mat);
  volume.weight_volume.download(weight_mat);
  tsdf_.resize(volume.volume_res_*volume.volume_res_*volume.volume_res_);
  weight_.resize(volume.volume_res_*volume.volume_res_*volume.volume_res_);
#pragma omp parallel for schedule(dynamic)
  for(int x=0;x<volume.volume_res_;x++){
    for(int y=0;y<volume.volume_res_;y++){
      for(int z=0;z<volume.volume_res_;z++){
        int idx = z*volume.volume_res_*volume.volume_res_ + y *volume.volume_res_+ x;
        float tsdf = tsdf_mat.at<float>(z*volume.volume_res_+y,x);
        float weight = weight_mat.at<float>(z*volume.volume_res_+y,x);
        tsdf_[idx] = tsdf;
        weight_[idx] = weight;
      }
    }
  }
}

std::shared_ptr<TriangleMesh> SurfaceRecontruction::extractTriangleMesh(){
  auto mesh =  std::make_shared<TriangleMesh>();
  //    std::unordered_map<Eigen::Vector4i, int, hash_eigen::hash<Eigen::Vector4i>> global_edge_to_vertex_index;
  std::unordered_map<double, int> global_edge_to_vertex_index;
  int local_edge_to_vertex_index[12];
  for(int x=0;x<config_.volume_resolution_-1;x++){
    for(int y=0;y<config_.volume_resolution_ -1;y++){
      for(int z=0;z<config_.volume_resolution_ -1;z++){

        Eigen::Vector4f pt_base(config_.voxel_size_*(1+x)+origin_(0), config_.voxel_size_*(1+y)+origin_(1),
                                config_.voxel_size_*(1+z)+origin_(2), 1);
        Eigen::Vector4f pt_world=base2world_*pt_base;
        if(pt_world(0)>config_.max_x_||pt_world(0)<config_.min_x_
           ||pt_world(1)>config_.max_y_||pt_world(1)<config_.min_y_
           ||pt_world(2)>config_.max_z_||pt_world(2)<config_.min_z_)
          continue;

        int cube_index = 0;
        float f[8];
        //check a cube with 8 voxels
        for(int i=0;i<8;i++)
        {
          Eigen::Vector3i idx = Eigen::Vector3i(x,y,z) + shift[i];
          if (weight_[IndexOf(idx)] == 0.0f)
          {
            cube_index = 0;
            break;
          }
          else
          {
            f[i] = tsdf_[IndexOf(idx)];
            if (f[i] < 0.0f) {
              cube_index |= (1 << i);
            }
          }
        }
        if (cube_index == 0 || cube_index == 255)    continue;
        for(int i=0;i<12;i++)
        {
          if(edge_table[cube_index]&(1<<i))
          {
            Eigen::Vector4i edge_index =  Eigen::Vector4i(x, y, z, 0) + edge_shift[i];
            //                        if(global_edge_to_vertex_index.find(edge_index)==global_edge_to_vertex_index.end())
            if(global_edge_to_vertex_index.find(
                 edge_index(0)*config_.volume_resolution_*config_.volume_resolution_
                 +edge_index(1)*config_.volume_resolution_
                 +edge_index(2)+0.1*edge_index(3))==global_edge_to_vertex_index.end())
            {
              local_edge_to_vertex_index[i] = (int)mesh->vertices_.size();
              //                            global_edge_to_vertex_index[edge_index] = (int)mesh->vertices_.size();
              global_edge_to_vertex_index[
                  edge_index(0)*config_.volume_resolution_*config_.volume_resolution_
                  +edge_index(1)*config_.volume_resolution_
                  +edge_index(2)+0.1*edge_index(3)] = (int)mesh->vertices_.size();
              Eigen::Vector3f pt(config_.voxel_size_*(0.5+edge_index(0)),
                                 config_.voxel_size_*(0.5+edge_index(1)),
                                 config_.voxel_size_*(0.5+edge_index(2)));
              float f0 = std::fabs((float)f[edge_to_vert[i][0]]);
              float f1 = std::fabs((float)f[edge_to_vert[i][1]]);
              pt(edge_index(3)) += f0 * config_.voxel_size_ / (f0 + f1);
              mesh->vertices_.push_back(pt + origin_);
            }
            else
              //                            local_edge_to_vertex_index[i] = global_edge_to_vertex_index.find(edge_index)->second;
              local_edge_to_vertex_index[i] = global_edge_to_vertex_index.find(
                    edge_index(0)*config_.volume_resolution_*config_.volume_resolution_
                    +edge_index(1)*config_.volume_resolution_
                    +edge_index(2)+0.1*edge_index(3))->second;
          }
        }
        for (int i = 0; tri_table[cube_index][i] != -1; i += 3)
        {
          mesh->triangles_.push_back(Eigen::Vector3i(
                                       local_edge_to_vertex_index[tri_table[cube_index][i]],
                                     local_edge_to_vertex_index[tri_table[cube_index][i + 2]],
              local_edge_to_vertex_index[tri_table[cube_index][i + 1]]));
        }
      }
    }
  }
  return mesh;
}

void SurfaceRecontruction::extractPclMesh(pcl::PolygonMesh& pclmesh)//,vector<Eigen::Matrix4f>& matrix_camera_to_marker_vec)
{
  std::shared_ptr<TriangleMesh> mcmesh = extractTriangleMesh();
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
  cloud->points.resize(mcmesh->vertices_.size());
#pragma omp parallel for schedule(dynamic)
  for(int i=0;i<mcmesh->vertices_.size();i++)
  {
    cloud->points[i] = pcl::PointXYZ(mcmesh->vertices_[i](0),mcmesh->vertices_[i](1),mcmesh->vertices_[i](2));
  }
  //    pcl::io::savePCDFile(data_path+"tsdf_cloud.pcd",*cloud,true);
  pclmesh.polygons.resize(mcmesh->triangles_.size());
#pragma omp parallel for schedule(dynamic)
  for(int i=0;i<mcmesh->triangles_.size();i++){
    pcl::Vertices v;
    v.vertices.push_back (mcmesh->triangles_[i](0));
        v.vertices.push_back (mcmesh->triangles_[i](1));
        v.vertices.push_back (mcmesh->triangles_[i](2));
        pclmesh.polygons[i] = v;
  }
  pcl::transformPointCloud(*cloud,*cloud,base2world_);//matrix_camera_to_marker_vec[0]);
  pcl::toPCLPointCloud2 (*cloud,pclmesh.cloud);

  //    pcl::NormalEstimationOMP<pcl::PointXYZ,pcl::Normal> ne; //compute normals
  //    ne.setNumberOfThreads(8);
  //    ne.setInputCloud(cloud);
  //    ne.setRadiusSearch(config_.voxel_size_*2);
  //    Eigen::Vector4f centroid;
  //    pcl::compute3DCentroid(*cloud,centroid);
  //    ne.setViewPoint(centroid[0],centroid[1],centroid[2]);
  //    pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>());
  //    ne.compute(*normals);
  //    for(size_t i =0;i<normals->size();i++)
  //    {
  //        normals->points[i].normal_x *=-1;
  //        normals->points[i].normal_y *=-1;
  //        normals->points[i].normal_z *=-1;
  //    }
  //    pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals(new  pcl::PointCloud<pcl::PointNormal>);
  //    pcl::concatenateFields(*cloud,*normals,*cloud_with_normals);
  //    pcl::toPCLPointCloud2 (*cloud_with_normals,pclmesh.cloud);
}


void SurfaceRecontruction::updateTSDFvolume(VolumeData& volume, const cv::Mat& depth_image,
                                            const Eigen::Matrix4f& transform, CameraParameters cam_params)
{
  bool debug=false;
  cv::cuda::GpuMat gpu_depth_frame(depth_image);
  cv::cuda::GpuMat gpu_diff_map = cv::cuda::createContinuous(cam_params.image_height,cam_params.image_width,CV_32FC1);

  if (config_.weight_mode==0) gpu_diff_map.setTo(1.f);
  else if (config_.weight_mode==1) {
    gpu_diff_map.setTo(0.f);
    Cuda::hostDiffMap(gpu_depth_frame,gpu_diff_map,cam_params.depth_scale,config_.discontinuity_kernel);
  }
  else abort();

  if (debug) {
    cv::Mat cpu_diff_map; gpu_diff_map.download(cpu_diff_map);
    cpu_diff_map=cpu_diff_map/100;
    cv::imwrite("diff_map.png",cpu_diff_map);
  }

  Eigen::Matrix4f cam2base,cam2world,world2base;
  world2base=base2world_.inverse();
  cam2world=transform;
  cam2base=world2base*cam2world;
  cv::Mat cam2base_cv = cv::Mat(cv::Size(4,4),CV_32FC1);
  cv::eigen2cv(cam2base,cam2base_cv);
  cv::cuda::GpuMat gpu_cam2base;
  //gpu_cam2base = cv::cuda::createContinuous(4,4,CV_32FC1);
  gpu_cam2base.upload(cam2base_cv);
  Cuda::hostTSDFVolume(gpu_depth_frame,volume,
                       cam_params,gpu_cam2base,config_.truncation_distance_,config_.depth_cutoff_,
                       config_.original_distance_x_,config_.original_distance_y_,config_.original_distance_z_,
                       cam_params.depth_scale,/*gpu_normal_map*/gpu_diff_map);

  gpu_cam2base.release();
  gpu_depth_frame.release();
}
