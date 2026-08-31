#include <algorithm>
#include <cmath>
#include <vector>
#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <Eigen/SVD>
#include "MeasurementFlowRotation.h"
#include "rotation.hpp"

Eigen::Matrix3d fitRotation(const Eigen::Matrix<double, 3, Eigen::Dynamic> & u0,
                            const Eigen::Matrix<double, 3, Eigen::Dynamic> & u1)
{
    if (u0.cols() == 0 || u0.cols() != u1.cols())
    {
        return Eigen::Matrix3d::Identity();
    }
    const Eigen::Matrix3d H = u1*u0.transpose();
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    // Reflect the least-significant singular direction if the raw product came
    // out with a negative determinant, which is the standard Kabsch correction
    // for noise pushing the fit off SO(3) and onto O(3).
    Eigen::Matrix3d D = Eigen::Matrix3d::Identity();
    D(2, 2) = (svd.matrixU()*svd.matrixV().transpose()).determinant() < 0.0 ? -1.0 : 1.0;
    return svd.matrixU()*D*svd.matrixV().transpose();
}

MeasurementFlowRotation::MeasurementFlowRotation(double time, const FlowFrame & frame,
                                                 const SystemLocalisation & system,
                                                 const Options & options)
    : Measurement(time)
    , RbcPrev_(frame.RbcPrev)
    , RbcCurr_(frame.RbcCurr)
    , dt_(frame.dt)
    , options_(options)
{
    // Linear in the state, so the exact Hessian is constant and the trust-region
    // Newton update converges in one step.
    updateMethod_ = UpdateMethod::NEWTONTRUSTEIG;

    if (!frame.valid || frame.matches.empty() || !(dt_ > 0.0))
    {
        return;
    }

    // The known part of the inter-frame camera rotation is the head's own
    // motion. When that dominates, the measurement is mostly reporting the neck
    // kinematics back to itself.
    if (logSO3(Eigen::Matrix3d(RbcCurr_.transpose()*RbcPrev_)).norm()/dt_ > options_.maxHeadRate)
    {
        return;
    }

    // ---- pre-gate at the prior mean, widened by the prior rate uncertainty ----
    const Eigen::VectorXd xPrior = system.density.mean();
    const Eigen::Vector3d omegaPrior = xPrior.segment<3>(SystemLocalisation::iOmega);
    const Eigen::Matrix3d Pomega = system.density.cov()
        .block<3, 3>(SystemLocalisation::iOmega, SystemLocalisation::iOmega);
    const double sigmaOmega = std::sqrt(std::max(Pomega.diagonal().maxCoeff(), 0.0));
    const double gate = std::min(options_.gateAngleMax,
                                 std::max(options_.gateAngle,
                                          options_.gateRateScale*sigmaOmega*dt_));

    const Eigen::Vector3d phiPrior = -dt_*omegaPrior;
    const Eigen::Matrix3d Rpred = RbcCurr_.transpose()*expSO3(phiPrior)*RbcPrev_;

    std::vector<Eigen::Index> keep;
    keep.reserve(frame.matches.size());
    for (std::size_t j = 0; j < frame.matches.size(); ++j)
    {
        if ((frame.matches[j].uCurr - Rpred*frame.matches[j].uPrev).squaredNorm() > gate*gate)
        {
            nGated_++;
            continue;
        }
        keep.push_back(static_cast<Eigen::Index>(j));
    }
    if (keep.size() < options_.minMatches)
    {
        return;
    }

    uPrev_.resize(3, static_cast<Eigen::Index>(keep.size()));
    uCurr_.resize(3, static_cast<Eigen::Index>(keep.size()));
    for (Eigen::Index j = 0; j < static_cast<Eigen::Index>(keep.size()); ++j)
    {
        uPrev_.col(j) = frame.matches[static_cast<std::size_t>(keep[j])].uPrev;
        uCurr_.col(j) = frame.matches[static_cast<std::size_t>(keep[j])].uCurr;
    }

    // ---- robust fit of the body rotation over the interval ----
    //
    // phi is the body rotation vector, so that
    //   u_c(k) = Rbc(k)' * expSO3(phi) * Rbc(k-1) * u_c(k-1),   phi = -omega*dt.
    //
    // Iteratively reweighted Gauss-Newton against the same inlier-Gaussian +
    // uniform-clutter mixture the landmark model uses: the weight of a
    // correspondence is its posterior probability of being an inlier, so a
    // mistrack or a moving spectator fades out of both the update and the
    // information rather than being fought.
    const double sigma2 = options_.sigmaAngular*options_.sigmaAngular;
    const double w = options_.inlierProbability;
    const double logInlierWeight = std::log(w) - std::log(2.0*M_PI*sigma2);
    const double logClutter = std::log(1.0 - w) - std::log(4.0*M_PI);

    const Eigen::Index n = uCurr_.cols();
    const Eigen::Matrix<double, 3, Eigen::Dynamic> v = RbcPrev_*uPrev_;   // rays in {b(k-1)}

    // Seed from the rays themselves rather than the prior, so a badly wrong
    // prior rate cannot pull the fit into a local basin of the robust cost.
    phi_ = logSO3(Eigen::Matrix3d(RbcCurr_*fitRotation(uPrev_, uCurr_)*RbcPrev_.transpose()));

    Eigen::Matrix3d Lambda = Eigen::Matrix3d::Zero();
    for (int iter = 0; iter < options_.maxIterations; ++iter)
    {
        const Eigen::Matrix3d E = expSO3(phi_);
        Lambda.setZero();
        Eigen::Vector3d rhs = Eigen::Vector3d::Zero();

        for (Eigen::Index j = 0; j < n; ++j)
        {
            const Eigen::Vector3d p = E*v.col(j);                       // in {b(k)}
            const Eigen::Vector3d r = uCurr_.col(j) - RbcCurr_.transpose()*p;

            // Posterior inlier weight from the two-component mixture.
            const double a = logInlierWeight - 0.5*r.squaredNorm()/sigma2;
            const double wj = 1.0/(1.0 + std::exp(logClutter - a));

            // d(predicted ray)/d(left perturbation of phi) = -Rbc(k)' * [p]x.
            // Exact to first order in the perturbation; the left-Jacobian
            // correction is O(|phi|) and |phi| is ~0.015 rad at video rate.
            const Eigen::Matrix3d J = -RbcCurr_.transpose()*hatSO3(p);
            Lambda.noalias() += (wj/sigma2)*(J.transpose()*J);
            rhs.noalias() += (wj/sigma2)*(J.transpose()*r);
        }

        const Eigen::Vector3d delta = Lambda.ldlt().solve(rhs);
        if (!delta.allFinite())
        {
            return;
        }
        // Compose on the left, which is the perturbation the Jacobian is for.
        phi_ = logSO3(Eigen::Matrix3d(expSO3(delta)*E));
        if (delta.norm() < options_.convergence)
        {
            break;
        }
    }

    // Did the rays actually pin all three axes? A single ray constrains only the
    // two rotations that move it, so a bunched set leaves one direction free.
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(Lambda);
    if (es.info() != Eigen::Success || es.eigenvalues()(0) < options_.minInformation)
    {
        return;
    }

    // ---- covariance: the fit, plus the interval term ----
    //
    // The rays determine phi, the rotation integrated over the frame. The state
    // carries omega at the end of the frame. Under the process model's random
    // walk on omegaBb those differ by a zero-mean error of covariance
    // sigmaOmega^2*dt^3/3 -- and that error is common to every ray, so it cannot
    // be expressed as per-ray noise without the ray count averaging it away.
    // Here is where it belongs.
    const double intervalVar = options_.sigmaOmegaPsd*options_.sigmaOmegaPsd*dt_*dt_*dt_/3.0;
    const Eigen::Matrix3d Pphi = Lambda.inverse();
    const Eigen::Matrix3d Ptot = Pphi + intervalVar*Eigen::Matrix3d::Identity();

    y_ = -phi_/dt_;
    S_ = Ptot/(dt_*dt_);
    intervalShare_ = 3.0*intervalVar/Ptot.trace();

    const Eigen::LLT<Eigen::Matrix3d> llt(S_);
    if (llt.info() != Eigen::Success)
    {
        return;
    }
    Sinv_ = S_.inverse();
    logNormConst_ = -0.5*(3.0*std::log(2.0*M_PI) + std::log(S_.determinant()));
    usable_ = true;
}

MeasurementFlowRotation::MeasurementFlowRotation(double time, const FlowFrame & frame,
                                                 const SystemLocalisation & system)
    : MeasurementFlowRotation(time, frame, system, Options{})
{}

Eigen::Matrix3d MeasurementFlowRotation::fittedCameraRotation() const
{
    return RbcCurr_.transpose()*expSO3(phi_)*RbcPrev_;
}

double MeasurementFlowRotation::fitResidual() const
{
    if (uCurr_.cols() == 0)
    {
        return 0.0;
    }
    return std::sqrt((uCurr_ - predictRays(phi_)).squaredNorm()/uCurr_.cols());
}

Eigen::VectorXd MeasurementFlowRotation::simulate(const Eigen::VectorXd & x, const SystemEstimator &) const
{
    return x.segment<3>(SystemLocalisation::iOmega);
}

// y = omegaBb + v, v ~ N(0, S). Linear in the state, so the gradient and Hessian
// are written out rather than handed to autodiff: forward-mode over even three
// active variables would cost six evaluations to reproduce a constant matrix.

double MeasurementFlowRotation::logLikelihood(const Eigen::VectorXd & x, const SystemEstimator &) const
{
    if (!usable_)
    {
        return 0.0;
    }
    const Eigen::Vector3d e = y_ - x.segment<3>(SystemLocalisation::iOmega);
    return logNormConst_ - 0.5*e.dot(Sinv_*e);
}

double MeasurementFlowRotation::logLikelihood(const Eigen::VectorXd & x, const SystemEstimator &,
                                              Eigen::VectorXd & g) const
{
    g.setZero(x.size());
    if (!usable_)
    {
        return 0.0;
    }
    const Eigen::Vector3d e = y_ - x.segment<3>(SystemLocalisation::iOmega);
    g.segment<3>(SystemLocalisation::iOmega) = Sinv_*e;
    return logNormConst_ - 0.5*e.dot(Sinv_*e);
}

double MeasurementFlowRotation::logLikelihood(const Eigen::VectorXd & x, const SystemEstimator &,
                                              Eigen::VectorXd & g, Eigen::MatrixXd & H) const
{
    g.setZero(x.size());
    H.setZero(x.size(), x.size());
    if (!usable_)
    {
        return 0.0;
    }
    const Eigen::Vector3d e = y_ - x.segment<3>(SystemLocalisation::iOmega);
    g.segment<3>(SystemLocalisation::iOmega) = Sinv_*e;
    H.block<3, 3>(SystemLocalisation::iOmega, SystemLocalisation::iOmega) = -Sinv_;
    return logNormConst_ - 0.5*e.dot(Sinv_*e);
}
