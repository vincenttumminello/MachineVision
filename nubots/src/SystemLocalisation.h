/**
 * @file SystemLocalisation.h
 * @brief Defines the SystemLocalisation class for humanoid robot field localisation.
 */
#ifndef SYSTEMLOCALISATION_H
#define SYSTEMLOCALISATION_H

#include <vector>
#include <Eigen/Core>
#include "GaussianInfo.hpp"
#include "Pose.hpp"
#include "SystemEstimator.h"
#include "SensorLog.h"

/**
 * @brief A body-fixed twist sample derived from odometry.
 */
struct BodyTwistSample
{
    double t;                   ///< Sample time [s] (relative to log start)
    Eigen::Vector3d vBb;        ///< Body-fixed translational velocity [m/s]
    Eigen::Vector3d omegaBb;    ///< Body-fixed angular velocity [rad/s]
};

/*
 * State contains the 6-DOF torso pose in the field frame {f} plus a 2-DOF
 * camera-mount attitude bias:
 *
 *     [ rBFf    ] Torso position in field frame (3)
 * x = [ Thetafb ] Torso orientation RPY angles (3), Rfb = rpy2rot(Thetafb)
 *     [ deltaC  ] Camera mount attitude bias (roll, pitch) about the camera axes (2)
 *
 * Field frame {f}: origin at centre of field on the ground plane, z up,
 * consistent with the NUbots Hfw convention.
 *
 * The camera bias models a constant error in the kinematic torso-to-camera
 * chain (evident in the recorded data as ground-projection errors growing
 * with range^2, ~1 m at 4-5 m range, consistent with a 1.5-2 deg pitch
 * bias). It is a random-walk state with very small process noise, applied
 * on the camera side of the extrinsic transform by vision measurements:
 *   Tfc = Tfb(x) * Tbc * R(deltaC)
 *
 * Prediction is driven by a buffer of body-fixed twist samples obtained by
 * finite-differencing the odometry stream (Htw), treated as a known input
 * with additive process noise:
 *
 *   deta/dt = JK(eta) * nu(t) + dw,   ddeltaC/dt = dw_c
 *
 * where JK(eta) = blkdiag(Rfb, TK(Theta)) transports the body twist to
 * field-frame pose rates.
 */
class SystemLocalisation : public SystemEstimator
{
public:
    /**
     * @brief Process noise and input handling parameters.
     */
    struct Parameters
    {
        double sigmaPosXY  = 0.05;  ///< Position process noise PSD, horizontal [m/sqrt(s)]
        double sigmaPosZ   = 0.02;  ///< Position process noise PSD, vertical [m/sqrt(s)]
        double sigmaAtt    = 0.05;  ///< Roll/pitch process noise PSD [rad/sqrt(s)]
        double sigmaYaw    = 0.05;  ///< Yaw process noise PSD [rad/sqrt(s)]
        double sigmaCamBias = 1e-3; ///< Camera mount bias process noise PSD [rad/sqrt(s)]
    };

    static constexpr Eigen::Index nx = 8;   ///< State dimension

    SystemLocalisation(const GaussianInfo<double> & density, const std::vector<BodyTwistSample> & twistBuffer);
    virtual SystemLocalisation * clone() const;

    virtual Eigen::VectorXd dynamics(double t, const Eigen::VectorXd & x, const Eigen::VectorXd & u) const override;
    virtual Eigen::VectorXd dynamics(double t, const Eigen::VectorXd & x, const Eigen::VectorXd & u, Eigen::MatrixXd & J) const override;
    virtual Eigen::VectorXd input(double t, const Eigen::VectorXd & x) const override;
    virtual GaussianInfo<double> processNoiseDensity(double dt) const override;
    virtual std::vector<Eigen::Index> processNoiseIndex() const override;

    /**
     * @brief Torso pose in field frame from a state vector.
     * @param x State vector (6)
     * @return Tfb with rotationMatrix Rfb and translationVector rBFf
     */
    template <typename Scalar>
    static Pose<Scalar> fieldPose(const Eigen::VectorX<Scalar> & x)
    {
        Pose<Scalar> Tfb;
        Tfb.rotationMatrix = rpy2rot(Eigen::Vector3<Scalar>(x.template segment<3>(3)));
        Tfb.translationVector = x.template segment<3>(0);
        return Tfb;
    }

    /**
     * @brief Camera-mount attitude bias correction from a state vector.
     * @param x State vector (8)
     * @return Rotation applied on the camera side of the extrinsic: R(deltaC)
     */
    template <typename Scalar>
    static Eigen::Matrix3<Scalar> cameraBiasRotation(const Eigen::VectorX<Scalar> & x)
    {
        Eigen::Vector3<Scalar> rpy;
        rpy << x(6), x(7), Scalar(0);
        return rpy2rot(rpy);
    }

    /**
     * @brief Build a body-twist buffer by finite-differencing the odometry stream.
     *
     * For consecutive odometry samples, the relative pose
     * DeltaT = Twt(t1)^{-1} * Twt(t2) with Twt = Htw^{-1} yields
     * vBb = Delta r / dt and omegaBb = log(Delta R) / dt, stamped at the
     * interval midpoint. Samples spanning gaps larger than maxGap are skipped.
     *
     * @param sensors Time-ordered odometry samples (absolute time [s])
     * @param t0 Time origin subtracted from sample times [s]
     * @param maxGap Maximum sample spacing to difference across [s]
     */
    static std::vector<BodyTwistSample> twistFromOdometry(const std::vector<SensorsSample> & sensors, double t0, double maxGap = 0.1);

    GaussianInfo<double> positionDensity() const;       ///< Marginal density of rBFf
    GaussianInfo<double> orientationDensity() const;    ///< Marginal density of Thetafb

    /**
     * @brief Reset the state density and system clock (initialisation / relocalisation).
     * @param density New state density
     * @param time New system time [s]
     */
    void resetTo(const GaussianInfo<double> & density, double time);

    Parameters params;

protected:
    const std::vector<BodyTwistSample> * twistBuffer_;  ///< Non-owning; ZOH input lookup
};

#endif
