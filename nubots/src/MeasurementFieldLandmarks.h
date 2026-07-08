/**
 * @file MeasurementFieldLandmarks.h
 * @brief Vision measurement of known field landmarks from YOLO detections.
 */
#ifndef MEASUREMENTFIELDLANDMARKS_H
#define MEASUREMENTFIELDLANDMARKS_H

#include <string>
#include <vector>
#include <Eigen/Core>
#include "FieldMap.h"
#include "Measurement.h"
#include "Pose.hpp"
#include "SensorLog.h"
#include "SystemEstimator.h"
#include "SystemLocalisation.h"

/**
 * @class MeasurementFieldLandmarks
 * @brief Measurement event for YOLO field landmark detections (goal posts, L/T/X intersections).
 *
 * Each detection provides a unit ray in the camera frame {c}:
 *  - intersections: ray to the bounding box centre,
 *  - goal posts: ray to the bottom-centre of the bounding box (post base on the ground).
 *
 * Detections are associated to mapped landmarks of the same class by greedy
 * nearest-angle assignment against rays predicted at the prior mean, gated by
 * a maximum angular residual.
 *
 * The likelihood for each associated pair is an isotropic Gaussian on the
 * chordal residual e = u_meas - u_pred, which agrees with an angular error
 * Gaussian to second order:
 *   log p = sum_j [ -log(2 pi sigma^2) - ||e_j||^2 / (2 sigma^2) ]
 *
 * The MAP update through Measurement::update yields the Laplace-approximation
 * posterior (mean and sqrt information) in the usual way.
 */
class MeasurementFieldLandmarks : public Measurement
{
public:
    /**
     * @brief Association and noise options.
     *
     * Defaults calibrated against the recorded NUbots data: ground-projected
     * intersection detections scatter 0.3-1.0 m at 3-5 m range (5-10 deg angular)
     * with occasional gross outliers, so the likelihood is an inlier Gaussian
     * mixed with a uniform clutter component over the unit sphere.
     */
    struct Options
    {
        double sigmaAngular         = 0.10;             ///< Inlier ray angular noise std dev [rad]
        double gateAngle            = 0.35;             ///< Max association residual angle [rad] (~20 deg)
        double minConfidence        = 0.5;              ///< Minimum detection confidence
        double inlierProbability    = 0.7;              ///< Mixture weight of the inlier component
    };

    /**
     * @brief Construct and associate a landmark measurement.
     * @param time Event time [s]
     * @param sample Vision sample containing YOLO detections (rays in {c})
     * @param Tbc Camera pose w.r.t. torso (from kinematics; Tbc = Htw * Hcw^{-1} at capture time)
     * @param map Field landmark map
     * @param system System whose prior mean is used for data association
     * @param options Association and noise options
     */
    MeasurementFieldLandmarks(double time, const VisionSample & sample, const Pose<double> & Tbc,
                              const FieldMap & map, const SystemLocalisation & system,
                              const Options & options);

    /**
     * @brief Construct with default options.
     */
    MeasurementFieldLandmarks(double time, const VisionSample & sample, const Pose<double> & Tbc,
                              const FieldMap & map, const SystemLocalisation & system);

    virtual Eigen::VectorXd simulate(const Eigen::VectorXd & x, const SystemEstimator & system) const override;
    virtual double logLikelihood(const Eigen::VectorXd & x, const SystemEstimator & system) const override;
    virtual double logLikelihood(const Eigen::VectorXd & x, const SystemEstimator & system, Eigen::VectorXd & g) const override;
    virtual double logLikelihood(const Eigen::VectorXd & x, const SystemEstimator & system, Eigen::VectorXd & g, Eigen::MatrixXd & H) const override;

    /**
     * @brief Templated log-likelihood for autodiff.
     */
    template <typename Scalar> Scalar logLikelihoodImpl(const Eigen::VectorX<Scalar> & x) const;

    /**
     * @brief Predicted unit rays in {c} for the associated landmarks.
     */
    template <typename Scalar> Eigen::Matrix<Scalar, 3, Eigen::Dynamic> predictRays(const Eigen::VectorX<Scalar> & x) const;

    std::size_t numAssociated() const { return static_cast<std::size_t>(uMeas_.cols()); }
    std::size_t numCandidates() const { return candidates_.size(); }
    const Eigen::Matrix<double, 3, Eigen::Dynamic> & measuredRays() const { return uMeas_; }
    const Eigen::Matrix<double, 3, Eigen::Dynamic> & associatedLandmarks() const { return rLFf_; }

protected:
    /**
     * @brief MAP update with iterated re-association (cf. iterative landmark matching).
     *
     * After the MAP optimisation, detections are re-associated at the posterior
     * mean; if the association set changed, the prior is restored and the
     * optimisation re-run, for at most maxAssociationIterations_ passes.
     */
    virtual void update(SystemBase & system) override;

    /**
     * @brief Extract the measurement ray and landmark class for a detection.
     * @return true if the detection is a usable landmark class
     */
    static bool detectionRay(const Detection & det, Eigen::Vector3d & ray, LandmarkType & type);

    /**
     * @brief A usable detection before association.
     */
    struct CandidateDetection
    {
        Eigen::Vector3d ray;
        LandmarkType type;
    };

    /**
     * @brief (Re)build the associated pairs from the stored candidates at the given state.
     * @return Association keys (candidate index, landmark index) for change detection
     */
    std::vector<std::pair<std::size_t, std::size_t>> associate(const Eigen::VectorXd & x);

    const FieldMap & map_;                              ///< Field landmark map
    std::vector<CandidateDetection> candidates_;        ///< Usable detections (rays in {c})
    Pose<double> Tbc_;                                  ///< Camera pose w.r.t. torso at capture time
    Eigen::Matrix<double, 3, Eigen::Dynamic> uMeas_;    ///< Measured unit rays in {c} (associated only)
    Eigen::Matrix<double, 3, Eigen::Dynamic> rLFf_;     ///< Associated landmark positions in {f}
    std::vector<std::pair<std::size_t, std::size_t>> assocKeys_;  ///< Last association (candidate, landmark)
    Options options_;
    int maxAssociationIterations_ = 1;                  ///< Maximum association/optimisation passes
};

#include "rotation.hpp"

template <typename Scalar>
Eigen::Matrix<Scalar, 3, Eigen::Dynamic> MeasurementFieldLandmarks::predictRays(const Eigen::VectorX<Scalar> & x) const
{
    // Camera pose in field frame with mount-bias correction:
    // Tfc = Tfb(x) * Tbc * R(deltaC)
    Pose<Scalar> Tfb = SystemLocalisation::fieldPose(x);
    Pose<Scalar> Tbias(SystemLocalisation::cameraBiasRotation(x), Eigen::Vector3<Scalar>::Zero());
    Pose<Scalar> Tfc = Tfb*Pose<Scalar>(Tbc_)*Tbias;

    const Eigen::Matrix3<Scalar> Rcf = Tfc.rotationMatrix.transpose();
    const Eigen::Vector3<Scalar> rCFf = Tfc.translationVector;

    Eigen::Matrix<Scalar, 3, Eigen::Dynamic> uPred(3, rLFf_.cols());
    for (Eigen::Index j = 0; j < rLFf_.cols(); ++j)
    {
        Eigen::Vector3<Scalar> rLCc = Rcf*(rLFf_.col(j).cast<Scalar>() - rCFf);
        uPred.col(j) = rLCc/rLCc.norm();
    }
    return uPred;
}

template <typename Scalar>
Scalar MeasurementFieldLandmarks::logLikelihoodImpl(const Eigen::VectorX<Scalar> & x) const
{
    const Eigen::Index n = uMeas_.cols();
    if (n == 0)
    {
        return Scalar(0);
    }

    Eigen::Matrix<Scalar, 3, Eigen::Dynamic> uPred = predictRays<Scalar>(x);

    const double sigma2 = options_.sigmaAngular*options_.sigmaAngular;
    const double logNormConst = -std::log(2.0*M_PI*sigma2);   // 2 effective DOF per ray

    // Robust mixture: inlier Gaussian on the chordal residual + uniform clutter
    // over the unit sphere (density 1/(4 pi) per steradian).
    const double logInlierWeight = std::log(options_.inlierProbability);
    const double logClutter = std::log(1.0 - options_.inlierProbability) - std::log(4.0*M_PI);

    using std::exp, std::log;

    Scalar logLik = Scalar(0);
    for (Eigen::Index j = 0; j < n; ++j)
    {
        Eigen::Vector3<Scalar> e = uMeas_.col(j).cast<Scalar>() - uPred.col(j);
        Scalar a = Scalar(logInlierWeight + logNormConst) - Scalar(0.5)*e.squaredNorm()/Scalar(sigma2);
        Scalar b = Scalar(logClutter);
        // Stable log-sum-exp of the two mixture components
        if (a > b)
        {
            logLik += a + log(Scalar(1) + exp(b - a));
        }
        else
        {
            logLik += b + log(Scalar(1) + exp(a - b));
        }
    }
    return logLik;
}

#endif
