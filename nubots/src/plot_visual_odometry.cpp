#include <filesystem>
#include <string>
#include <print>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>
#include "BufferedVideo.h"
#include "Camera.h"
#include "DJIVideoCaption.h"
#include "GaussianInfo.hpp"
#include "SystemVisualNav.h"
#include "plot_visual_odometry.h"

// Forward declarations
// static void plotGroundPlane(cv::Mat & img, const Eigen::Vector6d & etak, const Camera & camera, const int & divisor);
// static void plotHorizon(cv::Mat & img, const Eigen::Vector6d & etak, const Camera & camera, const int & divisor);
// static void plotCompass(cv::Mat & img, const Eigen::Vector6d & etak, const Camera & camera, const int & divisor);
// static void plotEpipole(cv::Mat & img, const Eigen::Vector6d & etak, const Eigen::Vector6d & etakm1, const Camera & camera, const int & divisor);

void plotGroundPlane(cv::Mat & img, const Eigen::Vector6d & etak, const Camera & camera, const int & divisor)
{
    // Extract pose information
    Eigen::Vector3d rN_Bn = etak.head<3>();
    Eigen::Vector3d rpy = etak.tail<3>();
    Eigen::Matrix3d Rnb = rpy2rot(rpy);
    
    // Camera pose in NED frame
    Eigen::Matrix3d Rnc = Rnb * camera.Tbc.rotationMatrix;
    Eigen::Vector3d rN_Cn = rN_Bn + Rnb * camera.Tbc.translationVector;
    
    // Camera intrinsics
    double fx = camera.cameraMatrix.at<double>(0, 0) / divisor;
    double fy = camera.cameraMatrix.at<double>(1, 1) / divisor;
    double cx = camera.cameraMatrix.at<double>(0, 2) / divisor;
    double cy = camera.cameraMatrix.at<double>(1, 2) / divisor;
    
    // Create ground grid: 2 km × 2 km area with 100 m spacing
    const double gridSize = 2000.0;  // 2 km
    const double gridStep = 100.0;   // 100 m spacing        
    
    std::vector<cv::Point2f> gridPoints;
    
    // North-South lines (constant East coordinate)
    for (double east = -gridSize/2; east <= gridSize/2; east += gridStep)
    {
        for (double north = -gridSize/2; north <= gridSize/2; north += gridStep/10)
        {
            Eigen::Vector3d rN_Pn(north, east, 0.0);  // Point on ground plane (z=0 in NED)
            
            // Transform point to camera frame
            Eigen::Vector3d rC_Pc = Rnc.transpose() * (rN_Pn - rN_Cn);
            
            // Project to image plane (only if in front of camera)
            if (rC_Pc(2) > 0)
            {
                double u = fx * rC_Pc(0) / rC_Pc(2) + cx;
                double v = fy * rC_Pc(1) / rC_Pc(2) + cy;
                
                // Check if within image bounds
                if (u >= 0 && u < img.cols && v >= 0 && v < img.rows)
                {
                    gridPoints.push_back(cv::Point2f(u, v));
                }
            }
        }
        
        // Draw the line segments
        if (gridPoints.size() > 1)
        {
            for (size_t i = 0; i < gridPoints.size() - 1; ++i)
            {
                cv::line(img, gridPoints[i], gridPoints[i+1], cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
            }
        }
        gridPoints.clear();
    }
    
    // East-West lines (constant North coordinate)
    for (double north = -gridSize/2; north <= gridSize/2; north += gridStep)
    {
        for (double east = -gridSize/2; east <= gridSize/2; east += gridStep/10)
        {
            Eigen::Vector3d rN_Pn(north, east, 0.0);
            
            Eigen::Vector3d rC_Pc = Rnc.transpose() * (rN_Pn - rN_Cn);
            
            if (rC_Pc(2) > 0)
            {
                double u = fx * rC_Pc(0) / rC_Pc(2) + cx;
                double v = fy * rC_Pc(1) / rC_Pc(2) + cy;
                
                if (u >= 0 && u < img.cols && v >= 0 && v < img.rows)
                {
                    gridPoints.push_back(cv::Point2f(u, v));
                }
            }
        }
        
        if (gridPoints.size() > 1)
        {
            for (size_t i = 0; i < gridPoints.size() - 1; ++i)
            {
                cv::line(img, gridPoints[i], gridPoints[i+1], cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
            }
        }
        gridPoints.clear();
    }
}

void plotHorizon(cv::Mat & img, const Eigen::Vector6d & etak, const Camera & camera, const int & divisor)
{
    // Extract pose information
    Eigen::Vector3d rpy = etak.tail<3>();
    Eigen::Matrix3d Rnb = rpy2rot(rpy);
    
    // Camera orientation in NED frame
    Eigen::Matrix3d Rnc = Rnb * camera.Tbc.rotationMatrix;
    
    // Camera intrinsics (scaled consistently)
    double fx = camera.cameraMatrix.at<double>(0, 0) / divisor;
    double fy = camera.cameraMatrix.at<double>(1, 1) / divisor;
    double cx = camera.cameraMatrix.at<double>(0, 2) / divisor;
    double cy = camera.cameraMatrix.at<double>(1, 2) / divisor;
    
    std::vector<cv::Point2f> horizonPoints;
    horizonPoints.reserve(img.cols);

    for (int u = 0; u < img.cols; ++u)
    {
        double bestV = -1.0;
        bool found = false;

        // Evaluate z-component of rayN for v=0
        double xn0 = (static_cast<double>(u) - cx) / fx;
        double yn0 = (0.0 - cy) / fy;
        Eigen::Vector3d rayC0(xn0, yn0, 1.0);
        double zPrev = (Rnc * rayC0)(2);
        int vPrev = 0;

        // Search for sign change in z-component over v
        for (int v = 1; v < img.rows; ++v)
        {
            double xn = (static_cast<double>(u) - cx) / fx;
            double yn = (static_cast<double>(v) - cy) / fy;
            Eigen::Vector3d rayC(xn, yn, 1.0);
            double z = (Rnc * rayC)(2);

            if (zPrev == 0.0)
            {
                bestV = vPrev;
                found = true;
                break;
            }

            // sign change => zero crossing between vPrev and v
            if ((zPrev > 0.0 && z < 0.0) || (zPrev < 0.0 && z > 0.0))
            {
                // linear interpolation to approximate zero crossing
                double t = zPrev / (zPrev - z);
                bestV = vPrev + t * (v - vPrev);
                found = true;
                break;
            }

            zPrev = z;
            vPrev = v;
        }

        if (!found)
        {
            // fallback: pick v with smallest abs(z) (robust fallback)
            double minAbs = std::numeric_limits<double>::infinity();
            for (int v = 0; v < img.rows; ++v)
            {
                double xn = (static_cast<double>(u) - cx) / fx;
                double yn = (static_cast<double>(v) - cy) / fy;
                Eigen::Vector3d rayC(xn, yn, 1.0);
                double z = (Rnc * rayC)(2);
                if (std::abs(z) < minAbs)
                {
                    minAbs = std::abs(z);
                    bestV = v;
                }
            }
        }

        if (bestV >= 0 && bestV < img.rows)
        {
            horizonPoints.push_back(cv::Point2f(static_cast<float>(u), static_cast<float>(bestV)));
        }
    }
    
    // Draw horizon as red curve
    if (horizonPoints.size() > 1)
    {
        for (size_t i = 0; i < horizonPoints.size() - 1; ++i)
        {
            cv::line(img, horizonPoints[i], horizonPoints[i+1], cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
        }
    }
}


void plotCompass(cv::Mat & img, const Eigen::Vector6d & etak, const Camera & camera, const int & divisor)
{
    // Extract pose information
    Eigen::Vector3d rpy = etak.tail<3>();
    Eigen::Matrix3d Rnb = rpy2rot(rpy);
    
    // Camera orientation in NED frame
    Eigen::Matrix3d Rnc = Rnb * camera.Tbc.rotationMatrix;
    
    // Camera intrinsics
    double fx = camera.cameraMatrix.at<double>(0, 0) / divisor;
    double fy = camera.cameraMatrix.at<double>(1, 1) / divisor;
    double cx = camera.cameraMatrix.at<double>(0, 2) / divisor;
    double cy = camera.cameraMatrix.at<double>(1, 2) / divisor;
    
    // Cardinal directions in NED frame (horizontal rays at infinity)
    std::vector<std::pair<std::string, Eigen::Vector3d>> directions = {
        {"N",  Eigen::Vector3d(1, 0, 0)},
        {"NE", Eigen::Vector3d(1, 1, 0).normalized()},
        {"E",  Eigen::Vector3d(0, 1, 0)},
        {"SE", Eigen::Vector3d(-1, 1, 0).normalized()},
        {"S",  Eigen::Vector3d(-1, 0, 0)},
        {"SW", Eigen::Vector3d(-1, -1, 0).normalized()},
        {"W",  Eigen::Vector3d(0, -1, 0)},
        {"NW", Eigen::Vector3d(1, -1, 0).normalized()}
    };
    
    for (const auto & [label, dirN] : directions)
    {
        // Transform direction to camera frame
        Eigen::Vector3d dirC = Rnc.transpose() * dirN;
        
        // Check if direction is visible (in front of camera)
        if (dirC(2) > 0)
        {
            // Project to image plane
            double u = fx * dirC(0) / dirC(2) + cx;
            double v = fy * dirC(1) / dirC(2) + cy;
            
            // Check if within image bounds
            if (u >= 0 && u < img.cols && v >= 0 && v < img.rows)
            {
                cv::putText(img, label, cv::Point(u, v), 
                           cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
            }
        }
    }
}

void plotEpipole(cv::Mat & img, const Eigen::Vector6d & etak, const Eigen::Vector6d & etakm1, const Camera & camera, const int & divisor)
{
    // Extract poses
    Eigen::Vector3d rN_Bnk = etak.head<3>();
    Eigen::Vector3d rpyk = etak.tail<3>();
    Eigen::Matrix3d Rnbk = rpy2rot(rpyk);
    
    Eigen::Vector3d rN_Bnkm1 = etakm1.head<3>();
    Eigen::Vector3d rpykm1 = etakm1.tail<3>();
    Eigen::Matrix3d Rnbkm1 = rpy2rot(rpykm1);
    
    // Camera poses
    Eigen::Matrix3d Rnck = Rnbk * camera.Tbc.rotationMatrix;
    Eigen::Vector3d rN_Cnk = rN_Bnk + Rnbk * camera.Tbc.translationVector;
    
    Eigen::Matrix3d Rnckm1 = Rnbkm1 * camera.Tbc.rotationMatrix;
    Eigen::Vector3d rN_Cnkm1 = rN_Bnkm1 + Rnbkm1 * camera.Tbc.translationVector;
    
    // Translational velocity direction (secant approximation)
    Eigen::Vector3d delta_rN = rN_Cnk - rN_Cnkm1;
    
    // Transform to current camera frame
    Eigen::Vector3d velocityC = Rnck.transpose() * delta_rN;
    
    // Camera intrinsics
    double fx = camera.cameraMatrix.at<double>(0, 0) / divisor;
    double fy = camera.cameraMatrix.at<double>(1, 1) / divisor;
    double cx = camera.cameraMatrix.at<double>(0, 2) / divisor;
    double cy = camera.cameraMatrix.at<double>(1, 2) / divisor;
    
    // Check if epipole is visible (velocity in front of camera)
    if (velocityC(2) > 0)
    {
        // Project epipole to image plane
        double u = fx * velocityC(0) / velocityC(2) + cx;
        double v = fy * velocityC(1) / velocityC(2) + cy;
        
        // Check if within image bounds
        if (u >= 0 && u < img.cols && v >= 0 && v < img.rows)
        {
            // Draw orange circle at epipole
            cv::circle(img, cv::Point(u, v), 8, cv::Scalar(0, 165, 255), -1, cv::LINE_AA);
            cv::circle(img, cv::Point(u, v), 10, cv::Scalar(0, 0, 0), 2, cv::LINE_AA);
        }
    }
}
