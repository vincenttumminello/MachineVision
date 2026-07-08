#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <vector>
#include <Eigen/Core>
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>
#include "FieldMap.h"
#include "Measurement.h"
#include "MeasurementFieldLandmarks.h"
#include "SystemLocalisation.h"

MeasurementFieldLandmarks::MeasurementFieldLandmarks(double time, const VisionSample & sample, const Pose<double> & Tbc,
                                                     const FieldMap & map, const SystemLocalisation & system,
                                                     const Options & options)
    : Measurement(time, 0)
    , map_(map)
    , Tbc_(Tbc)
    , options_(options)
{
    // Exact Hessian is cheap for a 6-dim state and yields the exact
    // Laplace-approximation posterior sqrt information
    updateMethod_ = UpdateMethod::NEWTONTRUSTEIG;

    // Gather usable detections
    for (const Detection & det : sample.detections)
    {
        if (det.confidence < options_.minConfidence)
        {
            continue;
        }
        Eigen::Vector3d ray;
        LandmarkType type;
        if (detectionRay(det, ray, type))
        {
            candidates_.push_back({ray, type});
        }
    }

    assocKeys_ = associate(system.density.mean());
}

MeasurementFieldLandmarks::MeasurementFieldLandmarks(double time, const VisionSample & sample, const Pose<double> & Tbc,
                                                     const FieldMap & map, const SystemLocalisation & system)
    : MeasurementFieldLandmarks(time, sample, Tbc, map, system, Options{})
{}

bool MeasurementFieldLandmarks::detectionRay(const Detection & det, Eigen::Vector3d & ray, LandmarkType & type)
{
    if (det.name == "L-intersection")
    {
        type = LandmarkType::L_INTERSECTION;
        ray = det.corners.rowwise().sum();          // Bounding box centre
    }
    else if (det.name == "T-intersection")
    {
        type = LandmarkType::T_INTERSECTION;
        ray = det.corners.rowwise().sum();
    }
    else if (det.name == "X-intersection")
    {
        type = LandmarkType::X_INTERSECTION;
        ray = det.corners.rowwise().sum();
    }
    else if (det.name == "goal post")
    {
        type = LandmarkType::GOAL_POST;
        ray = det.corners.col(2) + det.corners.col(3);  // Bottom-centre (post base): BR + BL
    }
    else
    {
        return false;   // ball, robot, etc. are not mapped landmarks
    }

    if (!ray.allFinite() || ray.norm() < 1e-12)
    {
        return false;
    }
    ray.normalize();
    return true;
}

std::vector<std::pair<std::size_t, std::size_t>> MeasurementFieldLandmarks::associate(const Eigen::VectorXd & x)
{
    std::vector<std::pair<std::size_t, std::size_t>> keys;
    if (candidates_.empty())
    {
        uMeas_.resize(3, 0);
        rLFf_.resize(3, 0);
        return keys;
    }

    // Predicted ray for every mapped landmark at the given state (incl. camera mount bias)
    Pose<double> Tbias(SystemLocalisation::cameraBiasRotation<double>(x), Eigen::Vector3d::Zero());
    Pose<double> Tfc = SystemLocalisation::fieldPose<double>(x)*Tbc_*Tbias;
    const Eigen::Matrix3d Rcf = Tfc.rotationMatrix.transpose();
    const Eigen::Vector3d rCFf = Tfc.translationVector;

    // Greedy nearest-angle assignment per landmark type with gating:
    // enumerate all (detection, landmark) pairs of matching type, sort by angle,
    // then assign smallest-angle pairs first, each detection/landmark at most once.
    struct CandidatePair
    {
        double angle;
        std::size_t det;
        LandmarkType type;
        std::size_t lm;
    };
    std::vector<CandidatePair> pairs;
    for (std::size_t i = 0; i < candidates_.size(); ++i)
    {
        const std::vector<Eigen::Vector3d> & lms = map_.landmarks(candidates_[i].type);
        for (std::size_t j = 0; j < lms.size(); ++j)
        {
            Eigen::Vector3d uPred = Rcf*(lms[j] - rCFf);
            double n = uPred.norm();
            if (n < 1e-9)
            {
                continue;
            }
            double angle = std::acos(std::clamp(candidates_[i].ray.dot(uPred)/n, -1.0, 1.0));
            if (angle <= options_.gateAngle)
            {
                pairs.push_back({angle, i, candidates_[i].type, j});
            }
        }
    }
    std::sort(pairs.begin(), pairs.end(), [](const CandidatePair & a, const CandidatePair & b) { return a.angle < b.angle; });

    std::vector<bool> detUsed(candidates_.size(), false);
    // Landmark usage tracked per type via flat key: type-major index
    auto lmKey = [](LandmarkType type, std::size_t j)
    {
        return static_cast<std::size_t>(type)*1000 + j;
    };
    std::vector<std::size_t> usedLandmarks;
    std::vector<const CandidatePair *> chosen;
    for (const CandidatePair & p : pairs)
    {
        if (detUsed[p.det])
        {
            continue;
        }
        std::size_t key = lmKey(p.type, p.lm);
        if (std::find(usedLandmarks.begin(), usedLandmarks.end(), key) != usedLandmarks.end())
        {
            continue;
        }
        detUsed[p.det] = true;
        usedLandmarks.push_back(key);
        chosen.push_back(&p);
    }

    uMeas_.resize(3, static_cast<Eigen::Index>(chosen.size()));
    rLFf_.resize(3, static_cast<Eigen::Index>(chosen.size()));
    for (std::size_t k = 0; k < chosen.size(); ++k)
    {
        const CandidatePair & p = *chosen[k];
        uMeas_.col(static_cast<Eigen::Index>(k)) = candidates_[p.det].ray;
        rLFf_.col(static_cast<Eigen::Index>(k)) = map_.landmarks(p.type)[p.lm];
        keys.emplace_back(p.det, lmKey(p.type, p.lm));
    }
    std::sort(keys.begin(), keys.end());
    return keys;
}

Eigen::VectorXd MeasurementFieldLandmarks::simulate(const Eigen::VectorXd & x, const SystemEstimator & system) const
{
    // Stack predicted rays (noise-free measurement model)
    Eigen::Matrix<double, 3, Eigen::Dynamic> uPred = predictRays<double>(x);
    return Eigen::Map<Eigen::VectorXd>(uPred.data(), uPred.size());
}

double MeasurementFieldLandmarks::logLikelihood(const Eigen::VectorXd & x, const SystemEstimator & system) const
{
    return logLikelihoodImpl<double>(x);
}

double MeasurementFieldLandmarks::logLikelihood(const Eigen::VectorXd & x, const SystemEstimator & system, Eigen::VectorXd & g) const
{
    using autodiff::dual;
    using autodiff::gradient;
    using autodiff::wrt;
    using autodiff::at;

    Eigen::VectorX<dual> xdual = x.cast<dual>();
    dual fdual;
    auto func = [this](const Eigen::VectorX<dual> & xd) -> dual
    {
        return this->template logLikelihoodImpl<dual>(xd);
    };
    g = gradient(func, wrt(xdual), at(xdual), fdual);
    return static_cast<double>(fdual);
}

double MeasurementFieldLandmarks::logLikelihood(const Eigen::VectorXd & x, const SystemEstimator & system, Eigen::VectorXd & g, Eigen::MatrixXd & H) const
{
    using autodiff::dual2nd;
    using autodiff::hessian;
    using autodiff::wrt;
    using autodiff::at;

    g.resize(x.size());
    H.resize(x.size(), x.size());

    Eigen::VectorX<dual2nd> xdual = x.cast<dual2nd>();
    dual2nd fdual;
    auto func = [this](const Eigen::VectorX<dual2nd> & xd) -> dual2nd
    {
        return this->template logLikelihoodImpl<dual2nd>(xd);
    };
    H = hessian(func, wrt(xdual), at(xdual), fdual, g);
    return static_cast<double>(fdual);
}

void MeasurementFieldLandmarks::update(SystemBase & system_)
{
    SystemEstimator & system = dynamic_cast<SystemEstimator &>(system_);
    const GaussianInfo<double> prior = system.density;

    // Iterated re-association (cf. iterative landmark matching): optimise with
    // the current association, then re-associate at the posterior mean; if the
    // association set changed, restore the prior and re-run.
    for (int iteration = 0; iteration < maxAssociationIterations_; ++iteration)
    {
        if (uMeas_.cols() == 0)
        {
            system.density = prior;     // Nothing associated; leave the prior untouched
            return;
        }

        Measurement::update(system);

        if (iteration == maxAssociationIterations_ - 1)
        {
            break;      // Iteration budget exhausted
        }

        std::vector<std::pair<std::size_t, std::size_t>> newKeys = associate(system.density.mean());
        if (newKeys == assocKeys_)
        {
            break;      // Association converged (associate() rebuilt the same set)
        }
        assocKeys_ = newKeys;
        system.density = prior;
    }
}
