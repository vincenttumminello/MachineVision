/**
 * @file MeasurementFlowRotation.h
 * @brief Optical flow of far-field features as a bias-free measurement of the body angular rate.
 *
 * The rotational part of the optical flow field does not depend on depth. For a
 * feature far enough away that the camera's translation over one frame subtends
 * nothing, the bearing transforms by the inter-frame camera rotation alone:
 *
 *     u_c(k) = R_c(k)c(k-1) * u_c(k-1)
 *
 * and that rotation is the body rotation composed with the (known, kinematic)
 * motion of the head:
 *
 *     R_c(k)c(k-1) = Rbc(k)' * expSO3(phi) * Rbc(k-1),   phi = -integral of omegaBb
 *
 * using dRfb/dt = Rfb*hatSO3(omegaBb), the same kinematics the process model
 * integrates.
 *
 * ## Why this is worth having
 *
 * MeasurementGyroscope sees `omegaBb + bGyro` -- the sum -- and can never split
 * it on its own. The bias is currently made observable indirectly, by the
 * landmarks pinning attitude over many frames while the gyroscope drifts against
 * them. This model sees the rotation *directly, with no bias of its own*, so the
 * difference between the two instruments is the bias. On `data2` the very first
 * stationary frames already give gyro - flow = [+0.005, +0.042, -0.015] rad/s,
 * which is the bias the filter currently takes tens of seconds of landmark
 * evidence to converge to ([+0.31, +2.40, -0.78] deg/s).
 *
 * The component that matters is yaw rate: the upstream Mahony filter's bias
 * integrator is driven by a cross product of two near-vertical vectors and so has
 * no component about the vertical, which is exactly the axis that turns into
 * heading drift. Flow is also the one signal here that keeps measuring rotation
 * when no landmark is in view -- falls, getups, motion blur.
 *
 * ## Why the measurement is a fitted rotation rather than raw ray residuals
 *
 * Every other vision model here puts its residuals straight into the likelihood
 * and lets the MAP update fit them. That is the wrong shape for this one, and
 * the reason is the interval.
 *
 * The flow measures the rotation *integrated over* the frame interval,
 * phi = -integral(omega dt), while the state carries omega at the instant the
 * frame ends. Under the process model's own random walk those differ by a
 * zero-mean error of covariance sigmaOmega^2*dt^3/3 -- 6.9 mrad at
 * sigmaOmega = 1.5 rad/s/sqrt(s) and 25 fps, against a median inter-frame
 * rotation of 15.5 mrad. It is not a small term.
 *
 * Critically it is *common to every ray in the frame*, not independent per ray.
 * Expressed as per-ray noise, ~190 rays would average it down by sqrt(190) and
 * the update would believe the rotation about thirteen times more precisely than
 * it is known. That is not a subtle mis-tuning: it is what makes the measurement
 * overrule the gyroscope and drag the estimate off the field. Measured on
 * `data2`, the raw-residual form takes the run from 0.093 m / 5.41 deg to
 * 1.28 m / 51.4 deg and pushes the gyroscope bias from +2.40 to +5.96 deg/s.
 *
 * So the rays are reduced first, by a robust iteratively-reweighted fit, to the
 * rotation they determine and the information they determine it with. That is a
 * sufficient statistic -- the rays enter the state only through this rotation --
 * so nothing is double-counted, and the interval term is then added where it
 * belongs, as a common-mode covariance on the rotation itself:
 *
 *     omega_meas = -phi/dt,   Sigma = (Lambda^-1 + I*sigmaOmega^2*dt^3/3)/dt^2
 *
 * The result is a plain 3-dof Gaussian measurement of omegaBb, structurally the
 * same as MeasurementGyroscope but with no bias state, which is precisely the
 * property that makes the bias observable. Being linear in the state it also
 * costs one exact Newton step rather than an autodiff Hessian over hundreds of
 * residuals.
 *
 * ## What is deliberately left out
 *
 * The camera mount-bias states (deltaC) rotate both frames' extrinsics, so they
 * enter as R(deltaC)' * R_c(k)c(k-1) * R(deltaC) -- a similarity transform, not
 * a cancellation. It is dropped: at |deltaC| ~ 1 deg and an inter-frame rotation
 * of ~15 mrad the difference is ~3e-4 rad, an order below the interval term that
 * dominates the error budget.
 *
 * Translation is dropped too, but that one is enforced rather than assumed:
 * FlowTracker only returns features it classifies as beyond the field carpet or
 * above the horizon. A feature at range Z shifts by d/Z for a camera baseline d,
 * which at walking pace is ~0.7 mrad at 6 m and ~3 mrad at 1.5 m.
 *
 * The extrinsic comes from the head servos (NeckKinematics), not from
 * Htw*Hcw^-1. See that file for why: the differenced form reports up to
 * 0.5 rad/s of head motion on a recording whose head never moves, which is
 * larger than the signal.
 */
#ifndef MEASUREMENTFLOWROTATION_H
#define MEASUREMENTFLOWROTATION_H

#include <cmath>
#include <Eigen/Core>
#include "FlowTracker.h"
#include "Measurement.h"
#include "rotation.hpp"
#include "SystemEstimator.h"
#include "SystemLocalisation.h"

/**
 * @brief Least-squares rotation between two sets of unit rays (Kabsch/Procrustes).
 *
 * Minimises sum ||u1_j - R*u0_j||^2 over SO(3). Used to seed the robust fit, and
 * by the unit tests as an independent answer.
 *
 * @param u0 3xN unit rays in the first frame
 * @param u1 3xN unit rays in the second frame
 * @return The rotation taking u0 to u1
 */
Eigen::Matrix3d fitRotation(const Eigen::Matrix<double, 3, Eigen::Dynamic> & u0,
                            const Eigen::Matrix<double, 3, Eigen::Dynamic> & u1);

/**
 * @class MeasurementFlowRotation
 * @brief Far-field optical flow as a Gaussian measurement of omegaBb.
 *
 *   y = omegaBb + v,   v ~ N(0, Sigma)
 *
 * where y and Sigma come from a robust fit of the inter-frame rotation to the
 * ray correspondences, widened by the interval term described in the file
 * comment. No bias state appears, which is the whole point.
 */
class MeasurementFlowRotation : public Measurement
{
public:
    /**
     * @brief Noise, fitting and gating options.
     */
    struct Options
    {
        // Tracking places a corner to a few tenths of a pixel, which at this
        // lens (444 px/rad) is under 1 mrad. The sigma is set above that for the
        // residual parallax of a "far" feature that is only moderately far. It
        // sets the weight of the rays against each other and the scale of the
        // robust cutoff; the dominant term in the final covariance is the
        // interval one, which is computed rather than tuned.
        double sigmaAngular      = 0.003;   ///< Inlier ray noise std dev [rad]

        double inlierProbability = 0.85;    ///< Robust mixture inlier weight

        // Pre-gate on the residual at the prior mean, widened by the prior rate
        // uncertainty for the same reason the landmark gate widens with yaw
        // uncertainty: a fixed gate silently caps what the filter can recover
        // from. The robust weights do the real work; this only catches a frame
        // that has gone badly wrong.
        double gateAngle         = 0.05;    ///< Minimum pre-gate residual angle [rad]
        double gateRateScale     = 3.0;     ///< Pre-gate widens to this many rate std devs
        double gateAngleMax      = 0.30;    ///< Ceiling on the widened pre-gate [rad]

        std::size_t minMatches   = 20;      ///< Below this the frame is not a measurement
        int maxIterations        = 8;       ///< Iteratively-reweighted fit iterations
        double convergence       = 1e-7;    ///< Stop when the update falls below this [rad]

        // Fast head motion is where the model's known term stops being known:
        // the inter-frame camera rotation is then mostly neck, and a few
        // milliseconds of skew between the servo readings and the image lands in
        // the residual as pure error.
        double maxHeadRate       = 2.0;     ///< Max kinematic camera rate to accept [rad/s]

        // The random-walk PSD the process model puts on omegaBb, which is what
        // sizes the interval term. Kept here rather than read from the system so
        // the measurement can be constructed and tested standalone; the caller
        // passes the disturbed value while the robot is not upright.
        double sigmaOmegaPsd     = 1.50;    ///< omegaBb process noise PSD [rad/s/sqrt(s)]

        // An ill-conditioned fit means the rays did not pin all three rotation
        // axes -- too few, or all bunched into one direction. Rather than emit a
        // confident measurement about the axes that were seen and a meaningless
        // one about the third, reject the frame.
        double minInformation    = 1e4;     ///< Min eigenvalue of the fit information [1/rad^2]
    };

    /**
     * @brief Construct a flow-rotation measurement: pre-gate, robust fit, covariance.
     * @param time Event time [s]
     * @param frame Far-field correspondences from FlowTracker
     * @param system System whose prior seeds the fit and sets the pre-gate
     * @param options Noise, fitting and gating options
     */
    MeasurementFlowRotation(double time, const FlowFrame & frame, const SystemLocalisation & system,
                            const Options & options);

    /// @brief Construct with default options.
    MeasurementFlowRotation(double time, const FlowFrame & frame, const SystemLocalisation & system);

    virtual Eigen::VectorXd simulate(const Eigen::VectorXd & x, const SystemEstimator & system) const override;
    virtual double logLikelihood(const Eigen::VectorXd & x, const SystemEstimator & system) const override;
    virtual double logLikelihood(const Eigen::VectorXd & x, const SystemEstimator & system, Eigen::VectorXd & g) const override;
    virtual double logLikelihood(const Eigen::VectorXd & x, const SystemEstimator & system, Eigen::VectorXd & g, Eigen::MatrixXd & H) const override;

    /**
     * @brief True if the frame produced a usable measurement.
     *
     * False when the frame was invalid, the head was slewing, too few matches
     * survived the pre-gate, or the fit did not pin all three rotation axes.
     * Check before calling system.process(): an unusable measurement has an
     * identically zero likelihood, so applying it only costs an optimiser run.
     */
    bool usable() const { return usable_; }

    /// @brief The measured body angular velocity [rad/s].
    const Eigen::Vector3d & bodyRate() const { return y_; }

    /// @brief Covariance of the measured body angular velocity [(rad/s)^2].
    const Eigen::Matrix3d & covariance() const { return S_; }

    /// @brief Per-axis standard deviation of the measurement [rad/s].
    Eigen::Vector3d sigma() const { return S_.diagonal().cwiseSqrt(); }

    /**
     * @brief The share of the measurement variance that comes from the interval term.
     *
     * 1.0 means the rays are effectively exact and the whole uncertainty is "we
     * do not know how omega varied within the frame". Reported by the run
     * summary because it says which way to spend effort: below ~0.5 better
     * tracking helps, near 1.0 only a faster camera does.
     */
    double intervalVarianceShare() const { return intervalShare_; }

    std::size_t numMatches() const { return static_cast<std::size_t>(uCurr_.cols()); }
    std::size_t numGated() const { return nGated_; }
    double dt() const { return dt_; }

    /// @brief RMS chordal residual of the robust fit [rad]. Near the tracking noise if the flow is sound.
    double fitResidual() const;

    /// @brief The fitted camera rotation, before the head motion is stripped (diagnostic).
    Eigen::Matrix3d fittedCameraRotation() const;

protected:
    /// @brief Predicted rays in {c(k)} for a body rotation vector phi over the interval.
    Eigen::Matrix<double, 3, Eigen::Dynamic> predictRays(const Eigen::Vector3d & phi) const
    {
        return RbcCurr_.transpose()*expSO3(phi)*RbcPrev_*uPrev_;
    }

    Eigen::Matrix<double, 3, Eigen::Dynamic> uPrev_;    ///< Surviving rays in {c(k-1)}
    Eigen::Matrix<double, 3, Eigen::Dynamic> uCurr_;    ///< Surviving rays in {c(k)}
    Eigen::Matrix3d RbcPrev_;                           ///< Body-from-camera rotation at k-1
    Eigen::Matrix3d RbcCurr_;                           ///< Body-from-camera rotation at k
    double dt_ = 0.0;                                   ///< Frame interval [s]

    Eigen::Vector3d phi_ = Eigen::Vector3d::Zero();     ///< Fitted body rotation over the interval [rad]
    Eigen::Vector3d y_ = Eigen::Vector3d::Zero();       ///< Measured body angular velocity [rad/s]
    Eigen::Matrix3d S_ = Eigen::Matrix3d::Identity();   ///< Measurement covariance [(rad/s)^2]
    Eigen::Matrix3d Sinv_ = Eigen::Matrix3d::Zero();    ///< Its inverse, precomputed for the update
    double logNormConst_ = 0.0;                         ///< -0.5*log((2 pi)^3 |S|)
    double intervalShare_ = 0.0;                        ///< Fraction of trace(S) from the interval term
    bool usable_ = false;
    std::size_t nGated_ = 0;                            ///< Matches dropped by the pre-gate
    Options options_;
};

#endif
