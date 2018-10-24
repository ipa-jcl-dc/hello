#include "texturem.hpp"
const float MARGIN = 10;

void TextureMapping::readCamParam(const std::string& data_path,const CameraParameters& cam_params,
                                  std::vector<Eigen::Matrix4f>& transforms,std::vector<int>& pose_flags)
{
    for (int i=0;i<transforms.size();i=i+1)
    {
        if (pose_flags[i]!=1) continue;
        //-------------------------------------------------------------//
    }
}

void TextureMapping::readCamParam(const std::string& data_path,const std::vector<CameraParameters>& cam_params,
                                  std::vector<std::vector<Eigen::Matrix4f>>& transforms,std::vector<std::vector<int>>& pose_flags)
{
    data_path_=data_path;
    depth_scale_=cam_params[0].depth_scale;
    for (int i=0;i<transforms.size();i=i+1)
    {
        for (int j=0;j<transforms[i].size();j=j+1)
        {
            if (pose_flags[i][j]!=1) continue;

            CameraView cam;
            cam.K=cam_params[i].K;
            cam.focal_length_w = cam_params[i].focal_x;
            cam.focal_length_h = cam_params[i].focal_y;
            cam.center_w = cam_params[i].c_x;
            cam.center_h = cam_params[i].c_y;
            std::ostringstream ss;        ss<<i;
            std::ostringstream s;        s<<j;
            cam.texture_file = "color-frame-"+ss.str()+"-"+s.str()+".png";
            cam.image=cv::imread(data_path+cam.texture_file);
            cam.height = cam_params[i].image_height;
            cam.width = cam_params[i].image_width;
            cam.world_to_cam=transforms[i][j].inverse();

            //            cam.depth_img=cv::imread(data_path + "/depth-frame-"+ss.str()+"-"+s.str()+".png",cv::IMREAD_ANYDEPTH);

            cameras.push_back(cam);
        }
    }
}

bool TextureMapping::createMaterials()
{
    mesh.tex_materials.resize (cameras.size () + 1);
    for(int i = 0 ; i <= cameras.size() ; ++i)
    {
        pcl::TexMaterial mesh_material;
        mesh_material.tex_Ka.r = 0.2f;
        mesh_material.tex_Ka.g = 0.2f;
        mesh_material.tex_Ka.b = 0.2f;
        mesh_material.tex_Kd.r = 0.8f;
        mesh_material.tex_Kd.g = 0.8f;
        mesh_material.tex_Kd.b = 0.8f;
        mesh_material.tex_Ks.r = 1.0f;
        mesh_material.tex_Ks.g = 1.0f;
        mesh_material.tex_Ks.b = 1.0f;
        mesh_material.tex_d = 1.0f;
        mesh_material.tex_Ns = 75.0f;
        mesh_material.tex_illum = 2;

        std::stringstream tex_name;
        tex_name << "material_" << i;
        tex_name >> mesh_material.tex_name;

        if(i < cameras.size ())
            mesh_material.tex_file = cameras[i].texture_file;
        else
            mesh_material.tex_file = cameras[0].texture_file;//mesh_material.tex_file = "occluded.jpg";

        mesh.tex_materials[i] = mesh_material;
    }
}

void TextureMapping::textureMeshwithMultipleCameras ()
{
    output_mesh=mesh;
    output_mesh.tex_polygons.clear ();
    output_mesh.tex_polygons.resize(cameras.size()+1);
    output_mesh.tex_coordinates.resize(cameras.size()+1);

    //    viewSelection();
    pcl::PointCloud<pcl::PointXYZ>::Ptr mesh_cloud (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromPCLPointCloud2 (mesh.cloud, *mesh_cloud);

    for (int i=0;i<facesLable.size();i++)
    {
        int cam_idx=facesLable[i];
        if (cam_idx==cameras.size()) //occluded for all cameras
        {
            //            continue;
            Eigen::Vector2f UV;
            UV(0) = -1.0; UV(1) = -1.0;
            output_mesh.tex_coordinates[cameras.size()].push_back(UV);
            output_mesh.tex_coordinates[cameras.size()].push_back(UV);
            output_mesh.tex_coordinates[cameras.size()].push_back(UV);

            output_mesh.tex_polygons[cam_idx].push_back(mesh.tex_polygons[0][i]);
        }
        else
        {
            pcl::PointCloud<pcl::PointXYZ>::Ptr global_face_cloud (new pcl::PointCloud<pcl::PointXYZ>);
            global_face_cloud->push_back(mesh_cloud->points[mesh.tex_polygons[0][i].vertices[0]]);
            global_face_cloud->push_back(mesh_cloud->points[mesh.tex_polygons[0][i].vertices[1]]);
            global_face_cloud->push_back(mesh_cloud->points[mesh.tex_polygons[0][i].vertices[2]]);
            for (int j=0;j<3;j++)
            {
                Eigen::Vector2f uv=cameras[cam_idx].getUVCoords4GlobalPt(global_face_cloud->points[j]);
                output_mesh.tex_coordinates[cam_idx].push_back(uv);
            }
            output_mesh.tex_polygons[cam_idx].push_back(mesh.tex_polygons[0][i]);
        }
    }
    //    for(int i = 0 ; i < mesh.tex_polygons.size() ; ++i)
    //        PCL_INFO ("\tSub mesh %d contains %d faces and %d UV coordinates.\n", i, mesh.tex_polygons[i].size (), mesh.tex_coordinates[i].size ());
}

void TextureMapping::shiftXYOrigin_addVerticesNormal()
{
    //    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    //    pcl::fromPCLPointCloud2(output_mesh.cloud, *cloud);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_shifted(new pcl::PointCloud<pcl::PointXYZ>);
    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*cloud,centroid);
    Eigen::Matrix4f translation = Eigen::Matrix4f::Identity();
    translation(0,3)=-centroid(0); translation(1,3)=-centroid(1); translation(2,3)=-config_.min_z_;
    pcl::transformPointCloud (*cloud, *cloud_shifted, translation);

    pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals (new pcl::PointCloud<pcl::PointNormal>);
    pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud (cloud_shifted);
    pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> n;
    n.setNumberOfThreads(8);
    n.setInputCloud (cloud_shifted);
    n.setSearchMethod (tree);
    n.setKSearch (10);
    //    n.setRadiusSearch(0.002);
    //    n.setViewPoint(centroid[0],centroid[1],centroid[2]);
    n.compute (*normals);
    //    for(size_t i =0;i<normals->size();i++)
    //    {
    //        normals->points[i].normal_x *=-1;
    //        normals->points[i].normal_y *=-1;
    //        normals->points[i].normal_z *=-1;
    //    }

    pcl::concatenateFields (*cloud_shifted, *normals, *cloud_with_normals);
    pcl::toPCLPointCloud2 (*cloud_with_normals, output_mesh.cloud);
}

void TextureMapping::createMveScene(const std::string& output_path, const std::vector<std::vector<cv::Mat> >& image_data_vec,
                                    std::vector<std::vector<int>> pose_flags, std::vector<std::vector<Eigen::Matrix4f>> transforms, std::vector<CameraParameters> cam_params)
{
    boost::filesystem::remove_all(output_path+"scene");
    std::string path=output_path+"scene/views";
    int index=0;

    for(int i=0;i<image_data_vec.size();i++)
    {
        //#pragma omp parallel for schedule(dynamic)
        for(int j=0;j<image_data_vec[i].size();j++)
        {
            if (pose_flags[i][j]!=1) continue;
            std::stringstream ss;
            ss<<index++;
            std::string view_dir=path+"/view"+ss.str()+".mve";
            boost::filesystem::create_directories(view_dir);
            cv::imwrite(view_dir+"/original.png",image_data_vec[i][j]);

            Eigen::Matrix4f transf=transforms[i][j].inverse();

            std::ofstream fs;
            fs.open (view_dir+"/meta.ini");
            fs<<"[camera]\n";
            float focal_length=cam_params[i].focal_x/cam_params[i].image_width;
            fs<<"focal_length = "<<focal_length<<std::endl;
            fs<<"pixel_aspect = 1"<<std::endl;
            float px=cam_params[i].c_x/cam_params[i].image_width;
            float py=cam_params[i].c_y/cam_params[i].image_height;
            fs<<"principal_point = "<<px<<" "<<py<<std::endl;
            fs<<"radial_distortion = 0 0\n";
            fs<<"rotation = ";
            for (int r=0;r<3;r++)
                for (int c=0;c<3;c++)
                    fs<<transf(r,c)<<" ";
            fs<<std::endl;
            fs<<"translation = "<<transf(0,3)<<" "<<transf(1,3)<<" "<<transf(2,3)<<std::endl;

            fs<<"\n[view]\n";
            fs<<"id = "<<index<<std::endl;
            fs<<"name = color-"<<index<<std::endl;

            fs.close();
        }
    }
}

void TextureMapping::viewSelection ()
{
    //    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
    //    pcl::fromPCLPointCloud2 (mesh.cloud, *cloud);
    visiblity_vertex_to_image.resize(cloud->points.size());
    visiblity_image_to_vertex.resize(cameras.size());
    facesLable.clear();
    facesLable.resize(mesh.tex_polygons[0].size ());

    // calculate face normals in marker board coordinates
    std::vector<Eigen::Vector3f> normals(mesh.tex_polygons[0].size ());
#pragma omp parallel for schedule(dynamic)
    for (int idx_face=0;idx_face<normals.size();idx_face++)
    {
        pcl::PointXYZ v1,v2,v3;
        v1=cloud->points[mesh.tex_polygons[0][idx_face].vertices[0]];
        v2=cloud->points[mesh.tex_polygons[0][idx_face].vertices[1]];
        v3=cloud->points[mesh.tex_polygons[0][idx_face].vertices[2]];
        Eigen::Vector3f s1(v2.x - v1.x,v2.y - v1.y,v2.z - v1.z);
        Eigen::Vector3f s2(v3.x - v2.x,v3.y - v2.y,v3.z - v2.z);
        Eigen::Vector3f n=s1.cross(s2);
        float norm=n.norm();
        if (norm==0) norm=1e-10;
        normals[idx_face]=n/norm;
    }
    // faces to cameras costs
    std::vector<std::vector<float>> costs;
    costs.resize(mesh.tex_polygons[0].size ());
    for(int i=0;i<costs.size();i++) costs[i].resize(cameras.size());
    std::cout<<"total faces: "<<mesh.tex_polygons[0].size ()<<std::endl;
    cout<<"total cameras: "<<cameras.size()<<endl;

#pragma omp parallel for schedule(dynamic)
    for (int current_cam = 0; current_cam < static_cast<int> (cameras.size ()); ++current_cam)
    {
        //        PCL_INFO ("Processing camera %d of %d.\n", current_cam+1, cameras.size ());

        // transform mesh into camera's frame
        pcl::PointCloud<pcl::PointXYZ>::Ptr camera_cloud (new pcl::PointCloud<pcl::PointXYZ>);
        pcl::transformPointCloud (*cloud, *camera_cloud, cameras[current_cam].world_to_cam);

        pcl::PointCloud<pcl::PointXY>::Ptr projections (new pcl::PointCloud<pcl::PointXY>);
        //            std::vector<pcl::Vertices>::iterator current_face;
        std::vector<bool> visibility(mesh.tex_polygons[0].size());
        std::vector<UvIndex> uv_indexes;

        pcl::PointXY nan_point;
        nan_point.x = std::numeric_limits<float>::quiet_NaN ();
        nan_point.y = std::numeric_limits<float>::quiet_NaN ();
        UvIndex u_null;
        u_null.idx_cloud = -1;
        u_null.idx_face = -1;

        int cpt_invisible=0;
        for (int idx_face = 0; idx_face <  static_cast<int> (mesh.tex_polygons[0].size ()); ++idx_face)
        {
            pcl::PointXY uv_coord1;
            pcl::PointXY uv_coord2;
            pcl::PointXY uv_coord3;
            //project each vertice, if one is out of view, stop
            if (isFaceProjected2 (cameras[current_cam],
                                  camera_cloud->points[mesh.tex_polygons[0][idx_face].vertices[0]],
                                  camera_cloud->points[mesh.tex_polygons[0][idx_face].vertices[1]],
                                  camera_cloud->points[mesh.tex_polygons[0][idx_face].vertices[2]],
                                  uv_coord1, uv_coord2,  uv_coord3))
            {
                // face is in the camera's FOV

                // add UV coordinates
                projections->points.push_back (uv_coord1);
                projections->points.push_back (uv_coord2);
                projections->points.push_back (uv_coord3);

                // remember corresponding face
                UvIndex u1, u2, u3;
                u1.idx_cloud = mesh.tex_polygons[0][idx_face].vertices[0];
                u2.idx_cloud = mesh.tex_polygons[0][idx_face].vertices[1];
                u3.idx_cloud = mesh.tex_polygons[0][idx_face].vertices[2];
                u1.idx_face = idx_face; u2.idx_face = idx_face; u3.idx_face = idx_face;
                uv_indexes.push_back (u1);
                uv_indexes.push_back (u2);
                uv_indexes.push_back (u3);

                //keep track of visibility
                visibility[idx_face] = true;
            }
            else
            {
                projections->points.push_back (nan_point);
                projections->points.push_back (nan_point);
                projections->points.push_back (nan_point);
                uv_indexes.push_back (u_null);
                uv_indexes.push_back (u_null);
                uv_indexes.push_back (u_null);
                //keep track of visibility
                visibility[idx_face] = false;
                cpt_invisible++;
            }
        }

        // projections contains all UV points of the current faces
        // uv_indexes links a uv point to its point in the camera cloud
        // visibility contains tells if a face was in the camera FOV (false = skip)

        // TODO handle case were no face could be projected
        if (visibility.size () - cpt_invisible !=0)
        {
            //create kdtree
            pcl::KdTreeFLANN<pcl::PointXY> kdtree;
            kdtree.setInputCloud (projections);
            std::vector<int> idxNeighbors;
            std::vector<float> neighborsSquaredDistance;

            //check for self occlusions. At this stage, we skip faces that were already marked as occluded
            for (int idx_face = 0; idx_face <  static_cast<int> (mesh.tex_polygons[0].size ()); ++idx_face)
            {
                if (!visibility[idx_face])
                {
                    // we are now checking for self occlusions within the current faces
                    // the current face was already declared as occluded.
                    // therefore, it cannot occlude another face anymore => we skip it
                    continue;
                }

                pcl::PointXY uv_coord1;
                pcl::PointXY uv_coord2;
                pcl::PointXY uv_coord3;
                // face is in the camera's FOV
                uv_coord1=projections->points[idx_face*3 + 0];
                uv_coord2=projections->points[idx_face*3 + 1];
                uv_coord3=projections->points[idx_face*3 + 2];

                //get its circumsribed circle
                double radius;
                pcl::PointXY center;
                getTriangleCircumscribedCircleCentroid(uv_coord1, uv_coord2, uv_coord3, center, radius); // this function yields faster results than getTriangleCircumcenterAndSize

                // get points inside circ.circle
                if (kdtree.radiusSearch (center, radius, idxNeighbors, neighborsSquaredDistance) > 0 )
                {
                    // for each neighbor
                    for (size_t i = 0; i < idxNeighbors.size (); ++i)
                    {
                        if (std::max (camera_cloud->points[mesh.tex_polygons[0][idx_face].vertices[0]].z,
                                      std::max (camera_cloud->points[mesh.tex_polygons[0][idx_face].vertices[1]].z,
                                                camera_cloud->points[mesh.tex_polygons[0][idx_face].vertices[2]].z))
                                < camera_cloud->points[uv_indexes[idxNeighbors[i]].idx_cloud].z)
                        {
                            // neighbor is farther than all the face's points. Check if it falls into the triangle
                            if (checkPointInsideTriangle(uv_coord1, uv_coord2, uv_coord3, projections->points[idxNeighbors[i]]))
                            {
                                // current neighbor is inside triangle and is closer => the corresponding face
                                visibility[uv_indexes[idxNeighbors[i]].idx_face] = false;
                                //TODO we could remove the projections of this face from the kd-tree cloud, but I fond it slower, and I need the point to keep ordered to querry UV coordinates later
                            }
                        }
                    }
                }
            }
        }

        //        PCL_INFO("\t%f percent faces are visible in camera %d \n",
        //                 100.0f*std::count(visibility.begin(),visibility.end(),true)/visibility.size(),current_cam);
        std::vector<int> visible_vertices;
        for (int idx_face = 0; idx_face <  static_cast<int> (visibility.size ()); ++idx_face)
        {
            if(visibility[idx_face])
            {
                {
                    visible_vertices.push_back(mesh.tex_polygons[0][idx_face].vertices[0]);
                    visible_vertices.push_back(mesh.tex_polygons[0][idx_face].vertices[1]);
                    visible_vertices.push_back(mesh.tex_polygons[0][idx_face].vertices[2]);
                }
            }
        }
        //use set?????????
        std::sort(visible_vertices.begin(),visible_vertices.end());
        visible_vertices.erase(std::unique(visible_vertices.begin(),visible_vertices.end()),visible_vertices.end());
        //        PCL_INFO("\t%f percent points are visible in camera %d \n",
        //                 100.0f*visible_vertices.size()/camera_cloud->points.size(),current_cam);

#pragma omp critical
        {
            for(int i=0;i<visible_vertices.size();i++)
            {
                int idx = visible_vertices[i];
                visiblity_vertex_to_image[idx].push_back(current_cam);
                //                visiblity_image_to_vertex[current_cam].push_back(idx);
            }
            visiblity_image_to_vertex[current_cam]=visible_vertices;
        }

        for (int idx_face = 0; idx_face <  static_cast<int> (mesh.tex_polygons[0].size ()); ++idx_face)
        {
            if (visibility[idx_face]){
                Eigen::Matrix4f transform=cameras[current_cam].world_to_cam;
                Eigen::Matrix3f rotation=transform.block<3,3>(0,0);
                Eigen::Vector3f normal=rotation*normals[idx_face];

                pcl::PointXYZ v1,v2,v3;
                v1=camera_cloud->points[mesh.tex_polygons[0][idx_face].vertices[0]];
                v2=camera_cloud->points[mesh.tex_polygons[0][idx_face].vertices[1]];
                v3=camera_cloud->points[mesh.tex_polygons[0][idx_face].vertices[2]];

                Eigen::Vector3f view_dir(0-(v1.x+v2.x+v3.x)/3,0-(v1.y+v2.y+v3.y)/3,0-(v1.z+v2.z+v3.z)/3);
                float norm=view_dir.norm();
                if (norm==0) norm=1e-10;
                view_dir=view_dir/norm;

                float dot=normal.dot(view_dir);
                //                if (dot<0) dot*=-1;
                // cost = 1 - dot(face_normal, viewing_direction)
                costs[idx_face][current_cam] = 1.0 - dot;
            }
            else costs[idx_face][current_cam]=99;
        }
    }  // we have been through all the cameras.

    if (graph.adj_lists.size()!=costs.size()) abort();
    for (int iter=0;iter<config_.cost_iterations;iter++){
#pragma omp parallel for schedule(dynamic)
        for (int i=0;i<costs.size();i++)
        {
            auto min_pt = std::min_element(costs[i].begin(), costs[i].end());
            if (*min_pt==99) facesLable[i] = cameras.size();
            else facesLable[i] = std::distance(costs[i].begin(), min_pt);
        }
        for (int i=0;i<facesLable.size();i++){
            int cam_idx=facesLable[i];
            for(int j=0;j<graph.adj_lists[i].size();j++){
                int adj_idx=graph.adj_lists[i][j];
                costs[adj_idx][cam_idx]*=config_.cost_multiplicand;
            }
        }
    }

    std::vector<float> min_costs(costs.size());
#pragma omp parallel for schedule(dynamic)
    for (int i=0;i<costs.size();i++)
    {
        auto min_pt = std::min_element(costs[i].begin(), costs[i].end());
        if (*min_pt==99) facesLable[i] = cameras.size();
        else facesLable[i] = std::distance(costs[i].begin(), min_pt);
        min_costs[i]=*min_pt;
    }

    std::ofstream fs;
    if(optimised) fs.open("costs_opt.txt");
    else fs.open("costs_raw.txt");
    for (int i=0;i<costs.size();i++)
    {
        fs<<"min_index:"<<facesLable[i]<<"  min_value:"<<min_costs[i]<<std::endl;
    }
    fs.close();
}

void TextureMapping::textureMeshWithAtlas ()
{
    /* MVE: Label 0 is undefined. */
    //    viewSelection();
    if(facesLable.size()!=graph.num_nodes()) exit(EXIT_FAILURE);
    for (std::size_t i = 0; i < facesLable.size(); ++i) {
        const std::size_t label = facesLable[i];
        if (label > cameras.size()) exit(EXIT_FAILURE);
        graph.set_label(i, label);
    }
    tex::TexturePatches texture_patches;
    tex::VertexProjectionInfos vertex_projection_infos;
    std::cout << "Generating texture patches:" << std::endl;
    tex::generate_texture_patches(graph, mveMesh,/* mesh_info,*/ cameras,
                                  &vertex_projection_infos, &texture_patches);
    tex::generate_texture_atlases(&texture_patches, &texture_atlases);
    for (int i=0;i<texture_atlases.size();i++){
        std::stringstream s; s<<i;
        if (optimised) cv::imwrite(data_path_+"atlas_opt"+s.str()+".png",texture_atlases[i]->image);
        else cv::imwrite(data_path_+"atlas"+s.str()+".png",texture_atlases[i]->image);
    }

    output_mesh=mesh;
    output_mesh.tex_polygons.clear ();
    output_mesh.tex_polygons.resize(texture_atlases.size()/*+1*/);
    output_mesh.tex_coordinates.resize(texture_atlases.size()/*+1*/);
    for (int i=0;i<texture_atlases.size();i++)
    {
        for (int j=0;j<texture_atlases[i]->faces.size();j++)
        {
            for (int k=0;k<3;k++)
            {
                output_mesh.tex_coordinates[i].push_back(texture_atlases[i]->texcoords[j*3+k]);
            }
            output_mesh.tex_polygons[i].push_back(mesh.tex_polygons[0][texture_atlases[i]->faces[j]]);
        }
    }
}

bool TextureMapping::createAtlasMaterials()
{
    output_mesh.tex_materials.resize (texture_atlases.size () /*+ 1*/);
    for(int i = 0 ; i </*=*/ output_mesh.tex_materials.size() ; ++i)
    {
        pcl::TexMaterial mesh_material;
        mesh_material.tex_Ka.r = 0.2f;
        mesh_material.tex_Ka.g = 0.2f;
        mesh_material.tex_Ka.b = 0.2f;
        mesh_material.tex_Kd.r = 0.8f;
        mesh_material.tex_Kd.g = 0.8f;
        mesh_material.tex_Kd.b = 0.8f;
        mesh_material.tex_Ks.r = 1.0f;
        mesh_material.tex_Ks.g = 1.0f;
        mesh_material.tex_Ks.b = 1.0f;
        mesh_material.tex_d = 1.0f;
        mesh_material.tex_Ns = 75.0f;
        mesh_material.tex_illum = 2;

        std::stringstream tex_name;
        tex_name << "material_" << i;
        tex_name >> mesh_material.tex_name;

        //        if(i < cameras.size ())
        std::stringstream s; s<<i;
        if (optimised) mesh_material.tex_file = "atlas_opt"+s.str()+".png";
        else mesh_material.tex_file = "atlas"+s.str()+".png";
        //        else mesh_material.tex_file = cameras[0].texture_file;//mesh_material.tex_file = "occluded.jpg";

        output_mesh.tex_materials[i] = mesh_material;
    }
}

void TextureMapping::grayAndGradient()
{
    bool debug = false;
    for(int i=0;i<cameras.size();i++)
    {
        cv::Mat img = cameras[i].image;
        cv::cuda::GpuMat gpuImg(img);
        cv::cuda::GpuMat gpuImg_gray,gpuImg_gray_float, gpuImg_gx, gpuImg_gy;
        cv::cuda::cvtColor(gpuImg, gpuImg_gray,CV_BGR2GRAY);
        gpuImg_gray.convertTo(gpuImg_gray_float,CV_32FC1);
        cv::Ptr<cv::cuda::Filter> filter ;
        //        = cv::cuda::createGaussianFilter(gpuImg_gray_float.type(),gpuImg_gray_float.type(), cv::Size(3,3), 0);
        //        filter->apply(gpuImg_gray_float,gpuImg_gray_float);
        cv::Mat img_gray,img_x,img_y;
        gpuImg_gray_float.download(img_gray);
        filter = cv::cuda::createScharrFilter(gpuImg_gray_float.type(),gpuImg_gray_float.type(),1,0);
        filter->apply(gpuImg_gray_float,gpuImg_gx);
        gpuImg_gx.download(img_x);
        img_x=img_x/32;
        filter = cv::cuda::createScharrFilter(gpuImg_gray_float.type(),gpuImg_gray_float.type(),0,1);
        filter->apply(gpuImg_gray_float,gpuImg_gy);
        gpuImg_gy.download(img_y);
        img_y=img_y/32;
        if(debug)
        {
            std::ostringstream ss;    ss<<i;
            cv::imwrite("gray-frame-"+ss.str()+".png",img_gray);
            cv::imwrite("ScharrX-frame-"+ss.str()+".png",img_x);
            cv::imwrite("ScharrY-frame-"+ss.str()+".png",img_y);
        }
        images_gray.push_back(img_gray);
        grays_gx.push_back(img_x);
        grays_gy.push_back(img_y);
        gpuImg_gy.release();
        gpuImg_gx.release();
        gpuImg_gray.release();
        gpuImg.release();
        gpuImg_gray_float.release();
    }
}

//cv::Mat TextureMapping::makeDepthDiscontinuity(const int current_cam)
//{
//    // discontinue=255, smooth=0;
//    bool debug = true;
//    assert(!cameras[current_cam].depth_img.empty());
//    cv::cuda::GpuMat gpu_depth_frame(cameras[current_cam].depth_img);
//    cv::cuda::GpuMat gpu_depth;
//    gpu_depth_frame.convertTo(gpu_depth,CV_32FC1,depth_scale_); // in meter
//    //    cv::Mat cdc;gpu_depth.download(cdc);cdc=cdc*1000;cv::imwrite("cdc.png",cdc);

//    //    cv::cuda::GpuMat gpu_depth_nan = cv::cuda::createContinuous(gpu_depth_frame.rows,gpu_depth_frame.cols,CV_32FC1);
//    //    Cuda::hostMakeNanDepth(gpu_depth,gpu_depth_nan);//0->nan
//    //    gpu_depth_nan.convertTo(gpu_depth,CV_32FC1);

//    cv::cuda::GpuMat sobel_dx, sobel_dy;
//    cv::Ptr<cv::cuda::Filter> filter = cv::cuda::createSobelFilter(gpu_depth.type(),gpu_depth.type(),1,0,3,1,cv::BORDER_DEFAULT);
//    filter->apply(gpu_depth,sobel_dx);
//    filter = cv::cuda::createSobelFilter(gpu_depth.type(),gpu_depth.type(),0,1,3,1,cv::BORDER_DEFAULT);
//    filter->apply(gpu_depth,sobel_dy);
//    cv::cuda::GpuMat mask_gpu = cv::cuda::createContinuous(gpu_depth_frame.rows,gpu_depth_frame.cols,CV_8UC1);
//    //true threshold=depth_gap_threshold/4?
//    Cuda::hostMaskImage(sobel_dx,sobel_dy,mask_gpu,config_.depth_gap_threshold);

//    int dilate_kernel=3;
//    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size( 2*dilate_kernel + 1,2*dilate_kernel+1),
//                                                cv::Point(dilate_kernel,dilate_kernel));
//    filter = cv::cuda::createMorphologyFilter(cv::MORPH_DILATE,mask_gpu.type(),element);
//    filter->apply(mask_gpu,mask_gpu);
//    cv::Mat mask;
//    mask_gpu.download(mask);
//    if(debug)
//    {
//        std::ostringstream ss;        ss<<current_cam;
//        cv::imwrite("discontinuity-frame-"+ss.str()+".png",mask);
//    }
//    mask_gpu.release();
//    gpu_depth_nan.release();
//    gpu_depth_frame.release();
//    gpu_depth.release();
//    return std::move(mask);
//}


float bilinearInterpolationGrayIntensity(const cv::Mat& img, float u, float v)
{
    int x = (int)u;
    int y = (int)v;
    int x0 = cv::borderInterpolate(x,   img.cols, cv::BORDER_REFLECT_101);
    int x1 = cv::borderInterpolate(x+1, img.cols, cv::BORDER_REFLECT_101);
    int y0 = cv::borderInterpolate(y,   img.rows, cv::BORDER_REFLECT_101);
    int y1 = cv::borderInterpolate(y+1, img.rows, cv::BORDER_REFLECT_101);
    float a = u - (float)x;
    float b = v - (float)y;
    float value = (img.at<float>(y0,x0)* (1.f-a) + img.at<float>(y0,x1)* a) * (1.f-b)
            +(img.at<float>(y1,x0)*(1.f-a) + img.at<float>(y1,x1)* a) *b;
    return value;
}

bool TextureMapping::getIntensity(float& intesity, const Eigen::Vector3f& pt, int cam_id)
{
    const cv::Mat& gray_image = images_gray[cam_id];
    float u,v;
    Eigen::Vector2f pix=cameras[cam_id].getPixelCoords4GlobalPt(pt);
    u=pix(0); v=pix(1);
    if(u>MARGIN && u<gray_image.cols-MARGIN && v>MARGIN && v<gray_image.rows-MARGIN)
    {
        //int u_round = int(round(u));
        //int v_round = int(round(v));
        //return std::make_tuple(true, gray_image.at<T>(v_round,u_round));
        intesity=bilinearInterpolationGrayIntensity(gray_image,u,v);
        return true;
    }
    else{
        return false;
    }
}

void TextureMapping::setProxyIntensity4Vertex(std::vector<float>& proxy_intensity)
{
    proxy_intensity.resize(cloud->points.size());

#pragma omp parallel for schedule(dynamic)
    for(int i=0;i<cloud->points.size();i++)
    {
        proxy_intensity[i] =0.0;
        float sum =0.0;
        Eigen::Vector3f pt(cloud->points[i].x,cloud->points[i].y,cloud->points[i].z);
        for(int j =0; j<visiblity_vertex_to_image[i].size();j++)
        {
            int cam_id = visiblity_vertex_to_image[i][j];
            float gray;
            bool valid = getIntensity(gray,pt,cam_id);
            if(valid){
                sum +=1.0;
                proxy_intensity[i] +=gray;
            }
        }
        if(sum > 0) proxy_intensity[i] /=sum;
    }
}

void TextureMapping::optimization()
{
    checkPose();
    cout<<"begin optimization "<<endl;
    optimised=true;
    std::vector<float> proxy_intensity;
    int total_num_ = 0;
    setProxyIntensity4Vertex(proxy_intensity);
    for(int itr=0;itr<config_.maximum_iteration_;itr++)
    {
        float residual = 0.0;
        total_num_ = 0;
#pragma omp parallel for schedule(dynamic)
        for(int i=0;i<cameras.size();i++)
        {
            Eigen::MatrixXf JJ = Eigen::MatrixXf::Zero(6, 6);
            Eigen::VectorXf Jr = Eigen::VectorXf::Zero(6);
            float rr = 0.0;
            int this_num = 0;
            Eigen::Matrix4f pose = cameras[i].world_to_cam;
            for (auto iter = 0; iter < visiblity_image_to_vertex[i].size();iter++)
            {
                //vertex index
                int vert_idx = visiblity_image_to_vertex[i][iter];
                Eigen::Vector3f V(cloud->points[vert_idx].x,cloud->points[vert_idx].y,cloud->points[vert_idx].z);
                //Convert to camera coordinate
                Eigen::Vector4f g = pose * Eigen::Vector4f(V(0), V(1), V(2), 1);
                //Reproject image plane
                float u,v;
                Eigen::Vector2f pix=cameras[i].getPixelCoords4GlobalPt(V);
                u=pix(0); v=pix(1);
                if(u>images_gray[i].cols-MARGIN || u<MARGIN || v >images_gray[i].rows-MARGIN || v<MARGIN )
                    continue;
                //int u_round = int(round(u));
                //int v_round = int(round(v));
                //float gray  = images_gray[i].at<float>(v_round,u_round);
                //float dIdx = grays_gx[i].at<float>(v_round,u_round);
                //float dIdy = grays_gy[i].at<float>(v_round,u_round);
                float intensity = bilinearInterpolationGrayIntensity(images_gray[i],u,v);
                float dIdx = bilinearInterpolationGrayIntensity(grays_gx[i],u,v);
                float dIdy = bilinearInterpolationGrayIntensity(grays_gy[i],u,v);

                float invz = 1. /g(2);
                float v0 = dIdx * cameras[i].focal_length_w * invz;
                float v1 = dIdy * cameras[i].focal_length_h * invz;
                float v2 = -(v0 * g(0) + v1 * g(1)) * invz;
                float J[6];
                J[0] = (-g(2) * v1 + g(1) * v2);
                J[1] = (g(2) * v0 - g(0) * v2);
                J[2] = (-g(1) * v0 + g(0) * v1);
                J[3] = v0;
                J[4] = v1;
                J[5] = v2;

                for (int x = 0; x < 6; x++) {
                    for (int y = 0; y < 6; y++) {
                        JJ(x, y) += J[x] * J[y];
                    }
                }

                float r = (proxy_intensity[vert_idx] - intensity);
                for (int x = 0; x < 6; x++) {
                    Jr(x) += J[x] * r;
                }
                rr += r * r;
                this_num++;
            }//end all vertices for one camera

            Eigen::VectorXf increment(6);
            increment = /*-*/JJ.inverse() * Jr;
            //increment = -JJ.fullPivLu().solve(Jr).cast<float>();

            //            Eigen::Affine3f aff_mat;
            //            aff_mat.linear() = (Eigen::Matrix3f)
            //                    Eigen::AngleAxisf(increment(2), Eigen::Vector3f::UnitZ())
            //                    * Eigen::AngleAxisf(increment(1), Eigen::Vector3f::UnitY())
            //                    * Eigen::AngleAxisf(increment(0), Eigen::Vector3f::UnitX());
            //            aff_mat.translation() =
            //                    Eigen::Vector3f(increment(3), increment(4), increment(5));
            //            pose = aff_mat.matrix() * pose;
            Eigen::Matrix4f mat;
            mat<<1,-increment[2],increment[1],increment[3],
                    increment[2],1,-increment[0],increment[4],
                    -increment[1],increment[0],1,increment[5],
                    0,0,0,1;
            pose=mat*pose;
            cameras[i].world_to_cam = pose;
#pragma omp critical
            {
                residual += rr;
                total_num_ += this_num;
            }
        }//end all cameras
        if(itr==0 || itr == (config_.maximum_iteration_-1))
        {
            printf("Average squared residuals : %f\n", residual / total_num_);
        }
        setProxyIntensity4Vertex(proxy_intensity);
    }//end maximum_iteration_
    checkPose();
}

void TextureMapping::poseOptimization()
{
    checkPose();
    cout<<"begin optimization "<<endl;
    optimised=true;
    std::vector<float> proxy_intensity;
    setProxyIntensity4Vertex(proxy_intensity);
    for(int itr=0;itr<config_.maximum_iteration_;itr++)
    {
//      cout<<"iteration "<<itr<<endl;
      int total_num_ = 0;
      Eigen::MatrixXf residual = Eigen::MatrixXf::Zero(1,1);
#pragma omp parallel for schedule(dynamic)
        for(int i=0;i<cameras.size();i++)
        {
            Eigen::MatrixXf JJ = Eigen::MatrixXf::Zero(6, 6);
            Eigen::VectorXf Jr = Eigen::VectorXf::Zero(6);
            Eigen::Matrix4f pose = cameras[i].world_to_cam;
            Eigen::MatrixXf J(visiblity_image_to_vertex[i].size(),6);
            Eigen::MatrixXf r(visiblity_image_to_vertex[i].size(),1);

            for (auto vt = 0; vt < visiblity_image_to_vertex[i].size();vt++)
            {
                //vertex index
                int vert_idx = visiblity_image_to_vertex[i][vt];
                Eigen::Vector3f V(cloud->points[vert_idx].x,cloud->points[vert_idx].y,cloud->points[vert_idx].z);
                //Convert to camera coordinate
                Eigen::Vector4f g = pose * Eigen::Vector4f(V(0), V(1), V(2), 1);
                //Reproject image plane
                float u,v;
                Eigen::Vector2f pix=cameras[i].getPixelCoords4GlobalPt(V);
                u=pix(0); v=pix(1);
//                if(u>images_gray[i].cols-MARGIN || u<MARGIN || v >images_gray[i].rows-MARGIN || v<MARGIN )
//                    continue;
                float intensity = bilinearInterpolationGrayIntensity(images_gray[i],u,v);
                float dIdx = bilinearInterpolationGrayIntensity(grays_gx[i],u,v);
                float dIdy = bilinearInterpolationGrayIntensity(grays_gy[i],u,v);

                float invz = 1. /g(2);
                float v0 = dIdx * cameras[i].focal_length_w * invz;
                float v1 = dIdy * cameras[i].focal_length_h * invz;
                float v2 = -(v0 * g(0) + v1 * g(1)) * invz;
//                float J[6];
                J(vt,0) = (-g(2) * v1 + g(1) * v2);
                J(vt,1) = (g(2) * v0 - g(0) * v2);
                J(vt,2) = (-g(1) * v0 + g(0) * v1);
                J(vt,3) = v0;
                J(vt,4) = v1;
                J(vt,5) = v2;
                r(vt,0) = (proxy_intensity[vert_idx] - intensity);
//                this_num++;
            }//end all vertices for one camera

            JJ = J.transpose() * J;
            Jr = J.transpose() * r;
            Eigen::VectorXf increment(6);
            increment = /*-*/JJ.inverse() * Jr;
            //increment = -JJ.fullPivLu().solve(Jr).cast<float>();

            Eigen::Matrix4f mat;
            mat<<1,-increment[2],increment[1],increment[3],
                    increment[2],1,-increment[0],increment[4],
                    -increment[1],increment[0],1,increment[5],
                    0,0,0,1;
            pose=mat*pose;
            cameras[i].world_to_cam = pose;
#pragma omp critical
            {
                residual+= r.transpose()*r;
                total_num_ += r.size();
            }
        }//end all cameras
        if(itr==0 || itr == (config_.maximum_iteration_-1))
        {
            printf("Average squared residuals : %f\n", residual(0,0) / total_num_);
        }
        setProxyIntensity4Vertex(proxy_intensity);
    }//end maximum_iteration_
    checkPose();
}

void TextureMapping::checkPose()
{
    //check for the first vertex
    //    int vt_id=10;
    //#pragma omp parallel for schedule(dynamic)
    //    for(int j =0; j<visiblity_vertex_to_image[vt_id].size();j++)
    //    {
    //        int cam_id = visiblity_vertex_to_image[vt_id][j];
    //        std::ostringstream ss;        ss<<cam_id;
    //        Eigen::Vector2f pix=cameras[cam_id].getPixelCoords4GlobalPt(cloud->points[vt_id]);
    //        cv::Mat img=cameras[cam_id].image.clone();
    //        cv::circle(img,cv::Point2f(pix(0),pix(1)),8,cv::Scalar(255,0,0),CV_FILLED, 8,0);
    //        if (optimised) cv::imwrite("opt-"+ss.str()+".png",img);
    //        else cv::imwrite("raw-"+ss.str()+".png",img);
    //    }

    //check for the first image
    int cam_id = 10;
    //    cout<<visiblity_image_to_vertex[cam_id]<<endl;
    cv::Mat img=cameras[cam_id].image.clone();
    for(int j =0; j<visiblity_image_to_vertex[cam_id].size();j++)
    {
        int pt_id = visiblity_image_to_vertex[cam_id][j];
        Eigen::Vector2f pix=cameras[cam_id].getPixelCoords4GlobalPt(cloud->points[pt_id]);
        cv::circle(img,cv::Point2f(pix(0),pix(1)),4,cv::Scalar(255,0,0),CV_FILLED, 8,0);
    }
    if (optimised) cv::imwrite("opt.png",img);
    else cv::imwrite("raw.png",img);

}
