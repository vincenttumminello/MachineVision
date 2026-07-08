#ifndef PLOT_VISUAL_ODOMETRY_H
#define PLOT_VISUAL_ODOMETRY_H

#include <opencv2/core/mat.hpp>
#include <Eigen/Core>

struct Camera;

void plotGroundPlane(cv::Mat & img, const Eigen::Vector6d & etak, const Camera & camera, int divisor);
void plotHorizon(cv::Mat & img, const Eigen::Vector6d & etak, const Camera & camera, int divisor);
void plotCompass(cv::Mat & img, const Eigen::Vector6d & etak, const Camera & camera, int divisor);
void plotEpipole(cv::Mat & img, const Eigen::Vector6d & etak, const Eigen::Vector6d & etakm1, const Camera & camera, int divisor);

#endif // PLOT_VISUAL_ODOMETRY_H
