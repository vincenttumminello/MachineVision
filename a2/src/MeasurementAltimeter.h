#ifndef MEASUREMENTALTIMETER_H
#define MEASUREMENTALTIMETER_H

#include <vector>
#include <Eigen/Core>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include "SystemEstimator.h"
#include "Pose.hpp"
#include "Camera.h"
#include "Measurement.h"
#include "rotation.hpp"
#include "SystemVisualNav.h"

class MeasurementAltimeter : public Measurement
{
public:
    MeasurementAltimeter(double time, const Camera & camera, SystemVisualNav & system, const double currentAltitude, const double lastUsedAltitude, const Eigen::VectorXd & etakm1);
    virtual Eigen::VectorXd simulate(const Eigen::VectorXd & x, const SystemEstimator & system) const override;
    virtual double logLikelihood(const Eigen::VectorXd & x, const SystemEstimator & system) const override;
    virtual double logLikelihood(const Eigen::VectorXd & x, const SystemEstimator & system, Eigen::VectorXd & g) const override;
    virtual double logLikelihood(const Eigen::VectorXd & x, const SystemEstimator & system, Eigen::VectorXd & g, Eigen::MatrixXd & H) const override;

    // Override update method
    void update(SystemBase & system_) override;


    
  
protected:
    const Camera & camera_;

    Eigen::Matrix<double, 2, Eigen::Dynamic> rQOikm1_;      // Measured features for previous frame
    Eigen::Matrix<double, 2, Eigen::Dynamic> rQOik_;        // Measured features for current frame

    Eigen::Matrix<double, 2, Eigen::Dynamic> rQbarOikm1_;   // Undistorted features for previous frame
    Eigen::Matrix<double, 2, Eigen::Dynamic> rQbarOik_;     // Undistorted features for current frame

    std::vector<unsigned char> mask_;                       // Inlier mask

    Eigen::Matrix<double, 3, Eigen::Dynamic> pkm1_;         // Inlier undistorted homogeneous points in previous frame
    Eigen::Matrix<double, 3, Eigen::Dynamic> pk_;           // Inlier undistorted homogeneous points in current frame

    double sigma_;                                          // Feature error standard deviation (in pixels)
    double measurement_;                                    // Altimeter measurement
    Eigen::VectorXd etakm1_;                                // Previous state estimate

    enum class UpdateMethod {BFGSTRUSTSQRT, BFGSLMSQRT, SR1TRUSTEIG, NEWTONTRUSTEIG, AFFINE, GAUSSNEWTON, LEVENBERGMARQUARDT};

    UpdateMethod updateMethod_;  ///< The method used for updating the system.
};



#endif
