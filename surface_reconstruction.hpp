#pragma once
#include <cmath>
#include <utility>
#include <unordered_map>
#include <thread>

//CUDA
#include "data_types.hpp"
#include "cuda_headers.hpp"
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
//OpenCV
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Core>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>
//PCL
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <boost/thread/thread.hpp>
#include <pcl/console/parse.h>
#include <pcl/common/transforms.h>
#include <pcl/features/board.h>
#include <pcl/common/common_headers.h>
#include <pcl/PolygonMesh.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
//Headers
#include "configuration.hpp"
#include "matrix_utils.hpp"
struct TriangleMesh{
    std::vector<Eigen::Vector3f> vertices_;
    std::vector<Eigen::Vector3f> vertex_normals_;
    //std::vector<Eigen::Vector3d> vertex_colors_;
    std::vector<Eigen::Vector3i> triangles_;
    //std::vector<Eigen::Vector3d> triangle_normals_;
};

class SurfaceRecontruction{
public:
    typedef cv::Ptr<SurfaceRecontruction> Ptr;
    SurfaceRecontruction(const Configuration& config,Eigen::Matrix4f& base2world);//,const CameraParameters& cam_params);
    void volumeItegration(VolumeData& volume,std::vector<cv::Mat>& depth_images,
                          const std::vector<Eigen::Matrix4f>& transforms,
                          const std::vector<int>& pose_flags,
                          CameraParameters cam_params_,int index);
    void updateTSDFvolume(VolumeData& volume, const cv::Mat& depth_image,
                                                const Eigen::Matrix4f& transform, CameraParameters cam_params);
    void extractTSDF(const VolumeData &volume);
    ///
    /// \brief extractTriangleMesh extract mesh model from TSDF volume
    /// \return mesh model
    std::shared_ptr<TriangleMesh> extractTriangleMesh();

    void extractPclMesh(pcl::PolygonMesh& pclmesh);//,vector<Eigen::Matrix4f>& matrix_camera_to_marker_vec);

private:
    std::vector<float> tsdf_;
    std::vector<float> weight_;
    Eigen::Vector3f origin_; //base coordinate of the volume/grid origin
    const Configuration config_;
    const Eigen::Matrix4f base2world_;

    inline int IndexOf(int x, int y, int z) const {return z * config_.volume_resolution_ * config_.volume_resolution_ + y * config_.volume_resolution_ + x; }
    inline int IndexOf(const Eigen::Vector3i &xyz) const { return IndexOf(xyz(0), xyz(1), xyz(2));}
};
