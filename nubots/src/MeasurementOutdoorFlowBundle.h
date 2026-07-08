#ifndef MEASUREMENTOUTDOORFLOWBUNDLE_H
#define MEASUREMENTOUTDOORFLOWBUNDLE_H

#include <vector>
#include <Eigen/Core>
#include <iostream>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include "SystemEstimator.h"
#include "Pose.hpp"
#include "Camera.h"
#include "Measurement.h"
#include "rotation.hpp"
#include "SystemVisualNav.h"

class MeasurementOutdoorFlowBundle : public Measurement
{
public:
    MeasurementOutdoorFlowBundle(double time, const Camera & camera, SystemVisualNav & system, const cv::Mat & imgk_raw, const cv::Mat & imgkm1_raw, const Eigen::Matrix<double, 2, Eigen::Dynamic> & rQOikm1, const Eigen::VectorXd & etakm1);
    virtual Eigen::VectorXd simulate(const Eigen::VectorXd & x, const SystemEstimator & system) const override;
    virtual double logLikelihood(const Eigen::VectorXd & x, const SystemEstimator & system) const override;
    virtual double logLikelihood(const Eigen::VectorXd & x, const SystemEstimator & system, Eigen::VectorXd & g) const override;
    virtual double logLikelihood(const Eigen::VectorXd & x, const SystemEstimator & system, Eigen::VectorXd & g, Eigen::MatrixXd & H) const override;

    // Helper functions for log likelihood and visualisation
    template <typename Scalar> Eigen::Matrix<Scalar, 3, Eigen::Dynamic> predictFlowImpl(const Eigen::VectorX<Scalar> & x, const Eigen::Matrix<Scalar, 3, Eigen::Dynamic> & pkm1, const Eigen::Matrix<Scalar, 3, Eigen::Dynamic> & pk) const;
    template <typename Scalar> Scalar logLikelihoodImpl(const Eigen::VectorX<Scalar> & x) const;
    Eigen::Matrix<double, 2, Eigen::Dynamic> predictedFeatures(const Eigen::VectorXd & x, const SystemEstimator & system) const;

    const Eigen::Matrix<double, 2, Eigen::Dynamic> & trackedPreviousFeatures() const;
    const Eigen::Matrix<double, 2, Eigen::Dynamic> & trackedCurrentFeatures() const;
    const std::vector<unsigned char> & inlierMask() const;

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

    Eigen::VectorXd etakm1_;
    
    enum class UpdateMethod {BFGSTRUSTSQRT, BFGSLMSQRT, SR1TRUSTEIG, NEWTONTRUSTEIG, AFFINE, GAUSSNEWTON, LEVENBERGMARQUARDT};

    UpdateMethod updateMethod_;  ///< The method used for updating the system.
};

template <typename Scalar>
Eigen::Matrix<Scalar, 3, Eigen::Dynamic> MeasurementOutdoorFlowBundle::predictFlowImpl(const Eigen::VectorX<Scalar> & x, const Eigen::Matrix<Scalar, 3, Eigen::Dynamic> & pkm1, const Eigen::Matrix<Scalar, 3, Eigen::Dynamic> & pk) const
{
    assert(x.rows() >= 18);
    assert(x.cols() == 1);
    assert(pkm1.cols() == pk.cols());
    
    // Extract poses from state vector
    // x = [dummy[0-5], eta_k[6-11], eta_k-1[12-17]]
    // eta = [rN_B/N, theta_nb] where theta_nb are roll-pitch-yaw Euler angles for body
    
    // Extract body poses
    Eigen::Vector3<Scalar> rNBnk = x.template segment<3>(6);
    Eigen::Vector3<Scalar> thetaNbk = x.template segment<3>(9);
    Eigen::Vector3<Scalar> rNBnkm1 = x.template segment<3>(12);
    Eigen::Vector3<Scalar> thetaNbkm1 = x.template segment<3>(15);

    // Compute rotation matrices (body to NED)
    Eigen::Matrix3<Scalar> Rnbk = rpy2rot(thetaNbk);
    Eigen::Matrix3<Scalar> Rnbkm1 = rpy2rot(thetaNbkm1);

    // Create Pose objects for body
    Pose<Scalar> Tnbk;
    Tnbk.rotationMatrix = Rnbk;
    Tnbk.translationVector = rNBnk;
    
    Pose<Scalar> Tnbkm1;
    Tnbkm1.rotationMatrix = Rnbkm1;
    Tnbkm1.translationVector = rNBnkm1;
    
    // Get camera poses using bodyToCamera transformation
    Pose<Scalar> Tnck = camera_.bodyToCamera(Tnbk);
    Pose<Scalar> Tnckm1 = camera_.bodyToCamera(Tnbkm1);
    
    Eigen::Matrix3<Scalar> Rnck = Tnck.rotationMatrix;
    Eigen::Vector3<Scalar> rNCnk = Tnck.translationVector;
    Eigen::Matrix3<Scalar> Rnckm1 = Tnckm1.rotationMatrix;
    Eigen::Vector3<Scalar> rNCnkm1 = Tnckm1.translationVector;
    
    // Camera calibration matrix
    double fx = camera_.cameraMatrix.at<double>(0, 0);
    double fy = camera_.cameraMatrix.at<double>(1, 1);
    double cx = camera_.cameraMatrix.at<double>(0, 2);
    double cy = camera_.cameraMatrix.at<double>(1, 2);
    
    Eigen::Matrix3<Scalar> K;
    K << Scalar(fx), Scalar(0),  Scalar(cx),
         Scalar(0),  Scalar(fy), Scalar(cy),
         Scalar(0),  Scalar(0),  Scalar(1);
    
    Eigen::Matrix3<Scalar> Kinv = K.inverse();
    
    // e3 vector (pointing down in NED frame)
    Eigen::Vector3<Scalar> e3(Scalar(0), Scalar(0), Scalar(1));
    
    // Ground plane homography Hz (equation 3)
    // Hz = K * Rnc[k]^T * (I - (rN_C/N[k-1] - rN_C/N[k]) * e3^T / (e3^T * rN_C/N[k-1])) * Rnc[k-1] * K^-1
    Eigen::Vector3<Scalar> delta_r = rNCnkm1 - rNCnk;
    Scalar altitude_km1 = e3.dot(rNCnkm1);  // altitude at k-1 (negative in NED since z is down)
    
    // Avoid division by zero
    const Scalar eps = Scalar(1e-4);
    if (Eigen::numext::abs(altitude_km1) < eps) {
        if (altitude_km1 >= Scalar(0)) {
            altitude_km1 = eps;
        } else {
            altitude_km1 = -eps;
        }
    }
    
    Eigen::Matrix3<Scalar> projectionTerm = Eigen::Matrix3<Scalar>::Identity() - 
                                             (delta_r * e3.transpose()) / altitude_km1;
    
    Eigen::Matrix3<Scalar> Hz = K * Rnck.transpose() * projectionTerm * Rnckm1 * Kinv;
    
    // Sky dome homography Hinf (equation 4)
    // Hinf = K * Rnc[k]^T * Rnc[k-1] * K^-1
    Eigen::Matrix3<Scalar> Hinf = K * Rnck.transpose() * Rnckm1 * Kinv;
    
    // Predict features using appropriate homography
    Eigen::Matrix<Scalar, 3, Eigen::Dynamic> pk_hat(3, pkm1.cols());
    
    for (int j = 0; j < pkm1.cols(); ++j)
    {
        // Check if feature is above or below horizon (equation 2)
        // Below horizon if e3^T * Rnc[k] * K^-1 * p[k] > 0
        // Use the MEASURED point in frame k to determine which homography to use
        Eigen::Vector3<Scalar> p_hom = pk.col(j).template cast<Scalar>();
        Eigen::Vector3<Scalar> ray_camera = Kinv * p_hom;
        Eigen::Vector3<Scalar> ray_ned = Rnck * ray_camera;
        Scalar horizon_check = e3.dot(ray_ned);
        
        // Cast pkm1 column to Scalar type
        Eigen::Vector3<Scalar> pkm1_j = pkm1.col(j).template cast<Scalar>();        

        if (horizon_check > Scalar(0))
        {
            // Below horizon - use ground plane homography
            pk_hat.col(j) = Hz * pkm1_j;
        }
        else
        {
            // Above horizon - use sky dome homography
            pk_hat.col(j) = Hinf * pkm1_j;
        }
    }
    
    return pk_hat;
}

template <typename Scalar>
Scalar MeasurementOutdoorFlowBundle::logLikelihoodImpl(const Eigen::VectorX<Scalar> & x) const
{
    // TODO: Assignment
    assert(pkm1_.cols() == pk_.cols());

    // Cast measured features to Scalar type for autodiff
    Eigen::Matrix<Scalar, 3, Eigen::Dynamic> pkm1_scalar = pkm1_.template cast<Scalar>();
    Eigen::Matrix<Scalar, 3, Eigen::Dynamic> pk_scalar = pk_.template cast<Scalar>();

    Eigen::Matrix<Scalar, 3, Eigen::Dynamic> pk_hat = predictFlowImpl(x, pkm1_scalar, pk_scalar);
    
    // Convert to Cartesian coordinates with division-by-zero protection
    const Scalar eps = Scalar(1e-8);
    Eigen::Matrix<Scalar, 2, Eigen::Dynamic> rQbarOik_hat(2, pk_hat.cols());
    for (int j = 0; j < pk_hat.cols(); ++j)
    {
        Scalar w = pk_hat(2, j);
        if (Eigen::numext::abs(w) < eps) {
            w = (w >= Scalar(0)) ? eps : -eps;
        }
        rQbarOik_hat(0, j) = pk_hat(0, j) / w;
        rQbarOik_hat(1, j) = pk_hat(1, j) / w;
    }

    // ALSO normalize the measured homogeneous points in the same way
    Eigen::Matrix<Scalar, 3, Eigen::Dynamic> pk_meas = pk_.template cast<Scalar>();
    Eigen::Matrix<Scalar, 2, Eigen::Dynamic> rQbarOik_measured(2, pk_meas.cols());
    for (int j = 0; j < pk_meas.cols(); ++j)
    {
        Scalar wm = pk_meas(2, j);
        if (Eigen::numext::abs(wm) < eps) {
            wm = (wm >= Scalar(0)) ? eps : -eps;
        }
        rQbarOik_measured(0, j) = pk_meas(0, j) / wm;
        rQbarOik_measured(1, j) = pk_meas(1, j) / wm;
    }


    // Compute log likelihood using equation (6a) 
    Scalar logLik = Scalar(0);
    double sigma2 = sigma_ * sigma_;
    // double log_norm_const = -0.5 * std::log(2.0 * M_PI * sigma2);
    double log_norm_const = -std::log(2.0 * M_PI) - std::log(sigma2);
    
    for (int j = 0; j < pk_.cols(); ++j)
    {
        // Use the normalized measured feature
        Eigen::Vector2<Scalar> rQbarOik_meas = rQbarOik_measured.col(j);
        
        // Predicted undistorted feature
        Eigen::Vector2<Scalar> rQbarOik_pred = rQbarOik_hat.col(j);
        
        // Innovation (residual)
        Eigen::Vector2<Scalar> innovation = rQbarOik_meas - rQbarOik_pred;
        
        // Log likelihood contribution from this feature
        Scalar mahalanobis = innovation.squaredNorm() / Scalar(sigma2);
        logLik += Scalar(log_norm_const) - Scalar(0.5) * mahalanobis;
    }
    
    return logLik;
}


#endif
