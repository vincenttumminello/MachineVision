#include <cstddef>
#include <print>
#include <numeric>
#include <vector>
#include <algorithm>
#include <Eigen/Core>
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>
#include <autodiff/forward/utils/derivative.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include "GaussianInfo.hpp"
#include "rotation.hpp"
#include "SystemEstimator.h"
#include "MeasurementAltimeter.h"
#include "funcmin.hpp"

MeasurementAltimeter::MeasurementAltimeter(double time, const Camera & camera, SystemVisualNav & system, const double currentAltitude, const double lastUsedAltitude, const Eigen::VectorXd & etakm1)
    : Measurement(time)
    , camera_(camera)
    , sigma_(1.5)
    , measurement_(currentAltitude)
    , etakm1_(etakm1)
{
    // updateMethod_ = UpdateMethod::NEWTONTRUSTEIG;
    updateMethod_ = UpdateMethod::BFGSTRUSTSQRT;

    // Check if measurement has changed
    if (std::abs(currentAltitude - lastUsedAltitude) > 0.01) 
    {
        // If changed do an update
        system.clearFlowEvent();
        this->process(system);
    }
    
}

void MeasurementAltimeter::update(SystemBase & system_)
{
    // Downcast since we know that Measurement events only occur to SystemEstimator objects
    SystemEstimator & system = dynamic_cast<SystemEstimator &>(system_);

    const Eigen::Index & nx = system.density.dim();

    // Second-order iterated update
    Eigen::VectorXd g(nx);
    Eigen::VectorXd x = system.density.mean();
    Eigen::MatrixXd Xi = system.density.sqrtInfoMat();

    switch (updateMethod_)
    {
        case UpdateMethod::BFGSTRUSTSQRT: 
        {
            // Create cost function with prototype V = costFunc(x, g)
            auto costFunc = [&](const Eigen::VectorXd & x, Eigen::VectorXd & g){ return costJointDensity(x, system, g); };

            // Minimise cost
            int ret = funcmin::BFGSTrustSqrt(costFunc, x, g, Xi, verbosity_);
            assert(ret == 0);
            break;
        }
        case UpdateMethod::BFGSLMSQRT:
        {
            // Create cost function with prototype V = costFunc(x, g)
            auto costFunc = [&](const Eigen::VectorXd & x, Eigen::VectorXd & g){ return costJointDensity(x, system, g); };

            // Minimise cost
            int ret = funcmin::BFGSLMSqrt(costFunc, x, g, Xi, verbosity_);
            assert(ret == 0);
            break;
        }
        case UpdateMethod::SR1TRUSTEIG:
        {
            // Generate eigendecomposition of initial Hessian (prior information matrix)
            // via an SVD of Xi = U*D*V.', i.e., Xi.'*Xi = V*D*U.'*U*D*V.' = V*D^2*V.'
            // This avoids the loss of precision associated with directly computing the eigendecomposition of Xi.'*Xi
            Eigen::JacobiSVD<Eigen::MatrixXd> svd(Xi, Eigen::ComputeFullV);
            Eigen::MatrixXd Q = svd.matrixV();
            Eigen::VectorXd v = svd.singularValues().array().square();

            assert(Q.rows() == nx);
            assert(Q.cols() == nx);
            assert(v.size() == nx);

            // Foreshadowing: If we were doing landmark SLAM with a quasi-Newton method,
            //                we can purposely introduce negative eigenvalues for newly
            //                initialised landmarks to force the Hessian and hence
            //                posterior sqrt information matrix to be approximated correctly.

            // Create cost function with prototype V = costFunc(x, g)
            auto costFunc = [&](const Eigen::VectorXd & x, Eigen::VectorXd & g){ return costJointDensity(x, system, g); };

            // Minimise cost
            int ret = funcmin::SR1TrustEig(costFunc, x, g, Q, v, verbosity_);
            assert(ret == 0);

            // Post-calculate posterior square-root information matrix from Hessian eigendecomposition
            Xi = v.array().sqrt().matrix().asDiagonal()*Q.transpose();
            Eigen::HouseholderQR<Eigen::Ref<Eigen::MatrixXd>> qr(Xi);           // In-place QR decomposition
            Xi = Xi.triangularView<Eigen::Upper>();                             // Safe aliasing
            break;
        }
        case UpdateMethod::NEWTONTRUSTEIG:
        {
            // Create cost function with prototype V = costFunc(x, g, H)
            auto costFunc = [&](const Eigen::VectorXd & x, Eigen::VectorXd & g, Eigen::MatrixXd & H){ return costJointDensity(x, system, g, H); };

            // Minimise cost
            Eigen::MatrixXd Q(nx, nx);
            Eigen::VectorXd v(nx);
            int ret = funcmin::NewtonTrustEig(costFunc, x, g, Q, v, verbosity_);
            assert(ret == 0);

            // Post-calculate posterior square-root information matrix from Hessian eigendecomposition
            Xi = v.array().sqrt().matrix().asDiagonal()*Q.transpose();
            Eigen::HouseholderQR<Eigen::Ref<Eigen::MatrixXd>> qr(Xi);           // In-place QR decomposition
            Xi = Xi.triangularView<Eigen::Upper>();                             // Safe aliasing
            break;
        }
        default:
            throw std::invalid_argument("Invalid update method");
    }

    // Set posterior mean to maximum a posteriori (MAP) estimate
    Eigen::VectorXd mu = x;
    system.density = GaussianInfo<double>::fromSqrtInfo(Xi*mu, Xi);
}

Eigen::VectorXd MeasurementAltimeter::simulate(const Eigen::VectorXd & x, const SystemEstimator & system) const
{
    Eigen::VectorXd y;
    throw std::runtime_error("Not implemented");
    return y;
}

double MeasurementAltimeter::logLikelihood(const Eigen::VectorXd & x, const SystemEstimator & system) const
{
    const double y_alt = (this->measurement_);

    const double predicted_alt = -x(8);

    // Residual
    const double r = y_alt - predicted_alt;

    // Log likelihood
    const double loglik = -0.5 * std::log(2.0 * M_PI * sigma_ * sigma_)
                        - 0.5 * (r * r) / (sigma_ * sigma_);

    return loglik;
}

double MeasurementAltimeter::logLikelihood(const Eigen::VectorXd & x, const SystemEstimator & system, Eigen::VectorXd & g) const
{
    // Evaluate gradient for Newton and quasi-Newton methods
    g.resize(x.size());
    g.setZero();

    // Extract parameters
    const double y_alt = this->measurement_;               // y_alt[k]
    
    // Predicted measurement
    const double predicted_alt = -x(8);                    // -e3^T * r_C/N[k]
    const double r = y_alt - predicted_alt;                // y_alt + e3^T * r_C/N[k]
    
    // Gradient wrt state (only altitude component nonzero)
    // g(8) = -r / σ_² 
    g(8) = -(1.0 / (sigma_ * sigma_)) * r; 

    return logLikelihood(x, system);
}

double MeasurementAltimeter::logLikelihood(const Eigen::VectorXd & x, const SystemEstimator & system, Eigen::VectorXd & g, Eigen::MatrixXd & H) const
{
    // Evaluate Hessian for Newton method
    H.resize(x.size(), x.size());
    H.setZero();

    H(8, 8) = -1.0 / (sigma_ * sigma_);

    return logLikelihood(x, system, g);
}

