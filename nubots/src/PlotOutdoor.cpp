#include "PlotOutdoor.h"
#include "rotation.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <cmath>
#include <iostream>
#include <algorithm>

PlotOutdoor::PlotOutdoor()
    : measuredAlt_(0.0)
    , estimatedAlt_(0.0)
{
    eta_.setZero();
    etakm1_.setZero();
}
// void PlotOutdoor::setData(const SystemVisualNav& system, const MeasurementOutdoorFlowBundle& measurement)
// {
//     pSystem_.reset(system.clone());
//     img_ = system.getImage().clone();
    
//     // Extract flow data from measurement
//     rQOikm1_ = measurement.trackedPreviousFeatures();
//     rQOik_ = measurement.trackedCurrentFeatures();
//     inlierMask_ = measurement.inlierMask();
    
//     // Compute predicted features
//     Eigen::VectorXd x = system.getState();
//     rQOik_hat_ = measurement.predictedFeatures(x, system);
// }


void PlotOutdoor::setFlowData(
    const Eigen::Matrix<double, 2, Eigen::Dynamic>& rQOikm1,
    const Eigen::Matrix<double, 2, Eigen::Dynamic>& rQOik,
    const Eigen::Matrix<double, 2, Eigen::Dynamic>& rQOik_hat,
    const std::vector<unsigned char>& inlierMask)
{
    rQOikm1_ = rQOikm1;
    rQOik_ = rQOik;
    rQOik_hat_ = rQOik_hat;
    inlierMask_ = inlierMask;
}

void PlotOutdoor::setPose(const Eigen::Vector6d& eta, const Eigen::Vector6d& etakm1)
{
    eta_ = eta;
    etakm1_ = etakm1;
}

void PlotOutdoor::setAltitude(double measured, double estimated)
{
    measuredAlt_ = measured;
    estimatedAlt_ = estimated;
}

void PlotOutdoor::setNorthEast(double north, double east)
{
    north_ = north;
    east_ = east;
}

// void PlotOutdoor::render(cv::Mat& img, int divisor)
// {
//     // Render in order specified in assignment (tip box):
//     // 1. Camera image (already provided)
//     // 2. Predicted flow vectors (blue)
//     // 3. Measured flow vectors (green/red)
//     // 4. Horizon (red curve)
//     // 5. Ground grid (black)
//     // 6. Cardinal directions (text)
//     // 7. Epipole (orange marker)
//     // 8. Text overlays (altitude, position)
    
//     plotGroundPlane(img, divisor);
//     plotHorizon(img, divisor);
//     plotCompass(img, divisor);
//     plotEpipole(img, divisor);
//     plotFlowVectors(img, divisor);
//     plotText(img);
// }

void PlotOutdoor::plotFlowVectors(cv::Mat& img, int divisor)
{
    if (rQOikm1_.cols() == 0) return;
    
    int np = rQOik_.cols();
    
    // Draw predicted flow vectors in BLUE first (so measured can overwrite)
    for (int j = 0; j < np; ++j)
    {
        cv::Point2f p0(rQOikm1_(0, j) / divisor, rQOikm1_(1, j) / divisor);
        cv::Point2f p1_hat(rQOik_hat_(0, j) / divisor, rQOik_hat_(1, j) / divisor);
        cv::arrowedLine(img, p0, p1_hat, cv::Scalar(255, 0, 0), 1.5, cv::LINE_AA, 0, 0.2);
    }
    
    // Draw measured flow vectors in GREEN (inliers) or RED (outliers)
    for (int j = 0; j < np; ++j)
    {
        cv::Point2f p0(rQOikm1_(0, j) / divisor, rQOikm1_(1, j) / divisor);
        cv::Point2f p1(rQOik_(0, j) / divisor, rQOik_(1, j) / divisor);
        
        cv::Scalar color = inlierMask_[j] ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
        cv::arrowedLine(img, p0, p1, color, 1.5, cv::LINE_AA, 0, 0.2);
    }
}

void PlotOutdoor::plotGroundPlane(cv::Mat & img, const Eigen::Vector6d & etak, const Camera & camera, const int & divisor)
{
    // Extract body pose
    Eigen::Vector3d rN_Bn = etak.head<3>();
    Eigen::Vector3d rpy = etak.tail<3>();
    Eigen::Matrix3d Rnb = rpy2rot(rpy);
    
    // Get camera pose
    Eigen::Matrix3d Rnc = Rnb * camera.Tbc.rotationMatrix;
    Eigen::Vector3d rN_Cn = rN_Bn /* + Rnb * camera.Tbc.translationVector */ ;
    
    // Camera intrinsics (scaled)
    double fx = camera.cameraMatrix.at<double>(0, 0) / divisor;
    double fy = camera.cameraMatrix.at<double>(1, 1) / divisor;
    double cx = camera.cameraMatrix.at<double>(0, 2) / divisor;
    double cy = camera.cameraMatrix.at<double>(1, 2) / divisor;
    
    // Ground grid: 2 km × 2 km, 100 m spacing
    const double gridSize = 2000.0;
    const double gridStep = 100.0;
    
    std::vector<cv::Point2f> gridPoints;
    
    // North-South lines (constant East)
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

void PlotOutdoor::plotHorizon(cv::Mat & img, const Eigen::Vector6d & etak, const Camera & camera, const int & divisor)
{
    // Extract pose information
    Eigen::Vector3d rpy = etak.tail<3>();
    Eigen::Matrix3d Rnb = rpy2rot(rpy);
    
    // Camera orientation in NED frame
    Eigen::Matrix3d Rnc = Rnb * camera.Tbc.rotationMatrix;
    
    std::vector<cv::Point2f> horizonPoints;
    horizonPoints.reserve(img.cols);

    // Sample horizon directions around the camera
    for (int u = 0; u < img.cols; u += 2) // Sample every 2 pixels for performance
    {
        // Create horizontal rays (z=0 in NED frame)
        double angle = 2.0 * M_PI * u / img.cols; // 0 to 2π
        Eigen::Vector3d horizonDirN(cos(angle), sin(angle), 0.0); // Horizontal direction in NED
        
        // Transform to camera frame
        Eigen::Vector3d horizonDirC = Rnc.transpose() * horizonDirN;
        
        // Only project if ray points forward from camera
        if (horizonDirC(2) > 0)
        {
            cv::Vec3d rayVec(horizonDirC(0), horizonDirC(1), horizonDirC(2));
            cv::Vec2d pixel = camera.vectorToPixel(rayVec);
            
            // Scale for plotting image
            double u_scaled = pixel[0] / divisor;
            double v_scaled = pixel[1] / divisor;
            
            // Check if within image bounds
            if (u_scaled >= 0 && u_scaled < img.cols && v_scaled >= 0 && v_scaled < img.rows)
            {
                horizonPoints.push_back(cv::Point2f(u_scaled, v_scaled));
            }
        }
    }
    
    // Draw horizon as red curve
    if (horizonPoints.size() > 1)
    {
        // Sort points by x-coordinate for smooth line drawing
        std::sort(horizonPoints.begin(), horizonPoints.end(), 
                  [](const cv::Point2f& a, const cv::Point2f& b) { return a.x < b.x; });
        
        for (size_t i = 0; i < horizonPoints.size() - 1; ++i)
        {
            cv::line(img, horizonPoints[i], horizonPoints[i+1], cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
        }
    }
}

void PlotOutdoor::plotCompass(cv::Mat & img, const Eigen::Vector6d & etak, const Camera & camera, const int & divisor)
{
    Eigen::Vector3d rpy = etak.tail<3>();
    Eigen::Matrix3d Rnb = rpy2rot(rpy);
    Eigen::Matrix3d Rnc = Rnb * camera.Tbc.rotationMatrix;

    // Cardinal directions in NED
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
    
    for (const auto& [label, dirN] : directions)
    {
        Eigen::Vector3d dirC = Rnc.transpose() * dirN;

        if (dirC(2) > 0)
        {
            // Use Camera::vectorToPixel to get pixel coordinates (includes intrinsics/distortion)
            cv::Vec3d rPCc(dirC(0), dirC(1), dirC(2));
            cv::Vec2d q = camera.vectorToPixel(rPCc);

            // Scale for the plotting image (img has been resized by 1/divisor)
            double u = q[0] / double(divisor);
            double v = q[1] / double(divisor);
            
            // Check if within image bounds
            if (u >= 0 && u < img.cols && v >= 0 && v < img.rows)
            {
                cv::putText(img, label, cv::Point(u, v), 
                           cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
            }
        }
    }
}

void PlotOutdoor::plotEpipole(cv::Mat & img, const Eigen::Vector6d & nuk, const Eigen::Vector6d & etak, const Eigen::Vector6d & etakm1, const Camera & camera, const int & divisor)
{
    // Current and previous body poses
    Eigen::Vector3d rNBnk = etak.head<3>();
    Eigen::Matrix3d Rnbk = rpy2rot(etak.tail<3>());

    Eigen::Vector3d rNBnkm1 = etakm1.head<3>();
    Eigen::Matrix3d Rnbkm1 = rpy2rot(etakm1.tail<3>());
    
    // Body-to-camera 
    Eigen::Matrix3d Rbc = camera.Tbc.rotationMatrix;
    Eigen::Vector3d rBCb = camera.Tbc.translationVector;

    // Camera poses
    Eigen::Matrix3d Rnck = Rnbk * Rbc;
    Eigen::Vector3d rNCnk = rNBnk + Rnbk * rBCb;

    Eigen::Matrix3d Rnckm1 = Rnbkm1 * Rbc;
    Eigen::Vector3d rNCnkm1 = rNBnkm1 + Rnbkm1 * rBCb;

    // Translational velocity (secant approximation)
    Eigen::Vector3d delta_rN = rNCnk - rNCnkm1;
    Eigen::Vector3d velocityC = Rnck.transpose() * delta_rN;

    // Translational velocity (direct from state)
    Eigen::Vector3d v_b = nuk.segment<3>(0);     // Body-frame velocity
    Eigen::Vector3d v_c = Rbc.transpose() * v_b; // Camera-frame velocity

    // Plot the direct method with yellow dot
    if (v_c(2) > 0)
    {
        cv::Vec3d velocityVec(v_c(0), v_c(1), v_c(2));
        cv::Vec2d epipole_direct = camera.vectorToPixel(velocityVec);
        
        // Scale for plotting image
        double u = epipole_direct[0] / divisor;
        double v = epipole_direct[1] / divisor;
        
        if (u >= 0 && u < img.cols && v >= 0 && v < img.rows)
        {
            // Draw YELLOW circle
            cv::circle(img, cv::Point(u, v), 8, cv::Scalar(0, 255, 255), -1, cv::LINE_AA);
            cv::circle(img, cv::Point(u, v), 10, cv::Scalar(0, 0, 0), 2, cv::LINE_AA);
        }
    }
    // Plot secant approximation with orange dot (this will appear over the direct method)
    if (velocityC(2) > 0)
    {
        cv::Vec3d velocityVec(velocityC(0), velocityC(1), velocityC(2));
        cv::Vec2d epipole_secant = camera.vectorToPixel(velocityVec);
        
        // Scale for plotting image
        double u = epipole_secant[0] / divisor;
        double v = epipole_secant[1] / divisor;
        
        if (u >= 0 && u < img.cols && v >= 0 && v < img.rows)
        {
            // Draw ORANGE circle
            cv::circle(img, cv::Point(u, v), 8, cv::Scalar(0, 165, 255), -1, cv::LINE_AA);
            cv::circle(img, cv::Point(u, v), 10, cv::Scalar(0, 0, 0), 2, cv::LINE_AA);
        }
    }

}

void PlotOutdoor::plotAltitude(cv::Mat & img, const Eigen::Vector6d & etak, const Camera & camera, const int & divisor)
{
    // Plot estimated altitude in blue text and measured altitude in green text
    // Moved into plotText

}

void PlotOutdoor::plotText(cv::Mat& img)
{
    int fontFace = cv::FONT_HERSHEY_SIMPLEX;
    double fontScale = 0.6;
    int thickness = 2;
    
    // Estimated altitude in BLUE
    std::string altEstText = "Alt (est): " + std::to_string(static_cast<double>(estimatedAlt_)) + " m";
    cv::putText(img, altEstText, cv::Point(10, 30),
               fontFace, fontScale, cv::Scalar(255, 0, 0), thickness, cv::LINE_AA);
    
    // Measured altitude in GREEN
    std::string altMeasText = "Alt (meas): " + std::to_string(static_cast<double>(measuredAlt_)) + " m";
    cv::putText(img, altMeasText, cv::Point(10, 60),
               fontFace, fontScale, cv::Scalar(0, 255, 0), thickness, cv::LINE_AA);
    
    // North position in BLUE
    std::string northText = "North: " + std::to_string(static_cast<double>(north_)) + " m";
    cv::putText(img, northText, cv::Point(10, 90),
               fontFace, fontScale, cv::Scalar(255, 0, 0), thickness, cv::LINE_AA);
    
    // East position in BLUE
    std::string eastText = "East: " + std::to_string(static_cast<double>(east_)) + " m";
    cv::putText(img, eastText, cv::Point(10, 120),
               fontFace, fontScale, cv::Scalar(255, 0, 0), thickness, cv::LINE_AA);

}

// Trajectory stuff
void PlotOutdoor::addTrajectoryPoint(double timestamp, double slam_north, double slam_east,
                                     double gps_north, double gps_east)
{
    timestamps_.push_back(timestamp);
    slam_north_.push_back(slam_north);
    slam_east_.push_back(slam_east);
    gps_north_.push_back(gps_north);
    gps_east_.push_back(gps_east);
}

void PlotOutdoor::exportTrajectoryPlot(const std::filesystem::path& outputPath)
{
    if (slam_north_.empty()) {
        std::cerr << "Warning: No trajectory data to plot" << std::endl;
        return;
    }
    
    // Create plot image (800x800 pixels)
    cv::Mat plotImg(800, 800, CV_8UC3, cv::Scalar(255, 255, 255));
    
    // Find data bounds
    auto [min_n_gps, max_n_gps] = std::minmax_element(gps_north_.begin(), gps_north_.end());
    auto [min_e_gps, max_e_gps] = std::minmax_element(gps_east_.begin(), gps_east_.end());
    auto [min_n_slam, max_n_slam] = std::minmax_element(slam_north_.begin(), slam_north_.end());
    auto [min_e_slam, max_e_slam] = std::minmax_element(slam_east_.begin(), slam_east_.end());
    
    double min_north = std::min(*min_n_gps, *min_n_slam);
    double max_north = std::max(*max_n_gps, *max_n_slam);
    double min_east = std::min(*min_e_gps, *min_e_slam);
    double max_east = std::max(*max_e_gps, *max_e_slam);
    
    // Add margin (10%)
    double range_north = max_north - min_north;
    double range_east = max_east - min_east;
    double margin = 0.1;
    min_north -= range_north * margin;
    max_north += range_north * margin;
    min_east -= range_east * margin;
    max_east += range_east * margin;
    
    // Scale to fit in image (leave 50px border)
    int border = 50;
    int plot_width = plotImg.cols - 2 * border;
    int plot_height = plotImg.rows - 2 * border;
    
    auto to_pixel = [&](double north, double east) -> cv::Point {
        int x = border + static_cast<int>((east - min_east) / (max_east - min_east) * plot_width);
        int y = plotImg.rows - border - static_cast<int>((north - min_north) / (max_north - min_north) * plot_height);
        return cv::Point(x, y);
    };
    
    // Draw grid and axes
    cv::line(plotImg, to_pixel(0, min_east), to_pixel(0, max_east), cv::Scalar(200, 200, 200), 1);
    cv::line(plotImg, to_pixel(min_north, 0), to_pixel(max_north, 0), cv::Scalar(200, 200, 200), 1);
    
    // Plot GPS trajectory (blue)
    for (size_t i = 1; i < gps_north_.size(); ++i) {
        cv::line(plotImg, 
                to_pixel(gps_north_[i-1], gps_east_[i-1]),
                to_pixel(gps_north_[i], gps_east_[i]),
                cv::Scalar(255, 0, 0), 2);
    }
    
    // Plot SLAM trajectory (green)
    for (size_t i = 1; i < slam_north_.size(); ++i) {
        cv::line(plotImg,
                to_pixel(slam_north_[i-1], slam_east_[i-1]),
                to_pixel(slam_north_[i], slam_east_[i]),
                cv::Scalar(0, 255, 0), 2);
    }
    
    // Mark start position (red circle)
    cv::circle(plotImg, to_pixel(0, 0), 5, cv::Scalar(0, 0, 255), -1);
    
    // Add legend
    cv::putText(plotImg, "GPS", cv::Point(border, 30), 
               cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 0, 0), 2);
    cv::putText(plotImg, "SLAM", cv::Point(border + 80, 30),
               cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
    cv::putText(plotImg, "Start", cv::Point(border + 160, 30),
               cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2);
    
    // Add axis labels
    cv::putText(plotImg, "East (m)", cv::Point(plotImg.cols/2 - 40, plotImg.rows - 10),
               cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    cv::putText(plotImg, "North (m)", cv::Point(10, plotImg.rows/2),
               cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    
    // Save plot
    cv::imwrite(outputPath.string(), plotImg);
    std::cout << "Trajectory plot saved to: " << outputPath.string() << std::endl;
}
