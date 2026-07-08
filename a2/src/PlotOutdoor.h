#ifndef PLOTOUTDOOR_H
#define PLOTOUTDOOR_H

#include <opencv2/core.hpp>
#include <Eigen/Core>
#include <vector>
#include <filesystem>
#include "Camera.h"
#include "GaussianInfo.hpp"
#include "SystemVisualNav.h"
#include "MeasurementOutdoorFlowBundle.h"

class PlotOutdoor
{
public:
    explicit PlotOutdoor();
    
    // Set data for current frame
    void setFlowData(
        const Eigen::Matrix<double, 2, Eigen::Dynamic>& rQOikm1,
        const Eigen::Matrix<double, 2, Eigen::Dynamic>& rQOik,
        const Eigen::Matrix<double, 2, Eigen::Dynamic>& rQOik_hat,
        const std::vector<unsigned char>& inlierMask
    );
    
    void setPose(const Eigen::Vector6d& eta, const Eigen::Vector6d& etakm1);
    void setAltitude(double measured, double estimated);
    void setNorthEast(double north, double east);

    // Trajectory collection and export
    void addTrajectoryPoint(double timestamp, double slam_north, double slam_east, 
                           double gps_north, double gps_east);
    void exportTrajectoryPlot(const std::filesystem::path& outputPath);
    
    // Render visualization on the image
    void render(cv::Mat& img, int divisor = 1);

    void setData(const SystemVisualNav& system, const MeasurementOutdoorFlowBundle& measurement);

    void plotGroundPlane(cv::Mat & img, const Eigen::Vector6d & etak, const Camera & camera, const int & divisor);
    void plotHorizon(cv::Mat & img, const Eigen::Vector6d & etak, const Camera & camera, const int & divisor);
    void plotCompass(cv::Mat & img, const Eigen::Vector6d & etak, const Camera & camera, const int & divisor);
    void plotEpipole(cv::Mat & img, const Eigen::Vector6d & nuk, const Eigen::Vector6d & etak, const Eigen::Vector6d & etakm1, const Camera & camera, const int & divisor);
    void plotAltitude(cv::Mat & img, const Eigen::Vector6d & etak, const Camera & camera, const int & divisor);
    void plotText(cv::Mat& img);
    
private:    
    // Flow data
    Eigen::Matrix<double, 2, Eigen::Dynamic> rQOikm1_;
    Eigen::Matrix<double, 2, Eigen::Dynamic> rQOik_;
    Eigen::Matrix<double, 2, Eigen::Dynamic> rQOik_hat_;
    std::vector<unsigned char> inlierMask_;
    
    // Pose data
    Eigen::Vector6d eta_;      // Current pose
    Eigen::Vector6d etakm1_;   // Previous pose
    
    // Altitude data
    double measuredAlt_;
    double estimatedAlt_;

    // North and East data
    double north_;
    double east_;

    // Trajectory data
    std::vector<double> timestamps_;
    std::vector<double> slam_north_;
    std::vector<double> slam_east_;
    std::vector<double> gps_north_;
    std::vector<double> gps_east_;
    
    // Helper functions (like they don't all help something)
    void plotFlowVectors(cv::Mat& img, int divisor);
};

#endif // PLOTOUTDOOR_H
