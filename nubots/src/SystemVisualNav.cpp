#include <cstddef>
#include <cmath>
#include <vector>
#include <print>
#include <Eigen/Core>
#include <print>
#include <opencv2/core/mat.hpp>
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>
#include "GaussianInfo.hpp"
#include "SystemEstimator.h"
#include "SystemVisualNav.h"
#include "kinematics_helper.h"

SystemVisualNav::SystemVisualNav(const GaussianInfo<double> & density)
    : SystemEstimator(density)
{

}

SystemVisualNav * SystemVisualNav::clone() const
{
    return new SystemVisualNav(*this);
}

void SystemVisualNav::predict(double time)
{
    double dt = time - time_;
    assert(dt >= 0);
    // if (dt == 0.0) return;

    // Augment state density with independent noise increment dw ~ N^{-1}(0, LambdaQ/dt)
    // [ x] ~ N^{-1}([ eta ], [ Lambda,          0 ])
    // [dw]         ([   0 ]  [      0, LambdaQ/dt ])

    auto pdw = processNoiseDensity(dt); // p(dw(idxQ)[k])
    auto pxdw = density*pdw;            // p(x[k], dw(idxQ)[k]) = p(x[k])*p(dw(idxQ)[k])

    // Phi maps [ x[k]; dw(idxQ)[k] ] to x[k+1]
    auto Phi = [&](const Eigen::VectorXd & xdw, Eigen::MatrixXd & J)
    {
        // xdw = [ x; dw(idxQ) ] where x is [current velocity, current pose, previous pose]
        const Eigen::Index nx = 18; // density.dim();
        const std::vector<Eigen::Index> idxQ = processNoiseIndex();
        const Eigen::Index nq = static_cast<Eigen::Index>(idxQ.size());
        assert(xdw.size() == nx + nq);

        // core state to integrate: first 12 elements (nu(6), eta(6))
        const Eigen::Index coreN = 12;

        // zeta (pose-at-last-flow) follows core, up to 6 elements
        Eigen::Index zetaDim = 6;

        // landmarks are any remaining elements after core and zeta
        // const Eigen::Index landStart = coreN + zetaDim;
        // const Eigen::Index landLen = std::max<Eigen::Index>(0, nx - landStart);

        // copy inputs (do not modify xdw)
        const Eigen::VectorXd x = xdw.head(nx);
        const Eigen::VectorXd dw = xdw.tail(nq);

        // build reduced zdw = [ x_core ; dw ]
        Eigen::VectorXd zdw(coreN + nq);
        zdw.setZero();
        zdw.head(coreN) = x.head(coreN); // build top 12 elements
        zdw.tail(nq) = dw; // add noise elements to end

        // integrate reduced system
        Eigen::MatrixXd J_temp;
        Eigen::VectorXd z_next;
        z_next = RK4SDEHelper(zdw, dt, J_temp);
        J_temp_ = J_temp; // store for potential zero-dt use next time
        
        // J_temp should be coreN x (coreN + nq)
        // J_temp will be 12x18. The top left 12x12 is dz_next/dz, top right 12x6 is dz_next/ddw

        // Reconstruct full state x_next
        Eigen::VectorXd x_next = x; // where x = [z; zeta]
        x_next.head(coreN) = z_next.head(coreN);

        if (isFlowEvent) {
            x_next.segment<6>(12) = x.segment<6>(6); // new zeta is old eta
        } else {
            x_next.segment<6>(12) = x.segment<6>(12); // new zeta is old zeta
        }

        // build full Jacobian: rows nx, cols nx + nq
        J.setZero(nx, nx + nq);

        // place core->core block
        if (coreN > 0) {
            J.block(0, 0, coreN, coreN) = J_temp.block(0, 0, coreN, coreN);
            // place core->dw block into columns [nx .. nx+nq-1]
            if (nq > 0) {
                J.block(0, nx, coreN, nq) = J_temp.block(0, coreN, coreN, nq);
            }
        }

        // landmarks: static, identity mapping from input to output
        // if (landLen > 0) {
        //     J.block(landStart, landStart, landLen, landLen).setIdentity();
        // }

        if (isFlowEvent) {
            J.block(coreN, 6, zetaDim, 6).setIdentity();            // dzeta_next/deta = I
            J.block(coreN, coreN, zetaDim, zetaDim).setZero();      // dzeta_next/dzeta = 0
        } else {
            J.block(coreN, 6, zetaDim, 6).setZero();                // dzeta_next/deta = 0
            J.block(coreN, coreN, zetaDim, zetaDim).setIdentity();  // dzeta_next/dzeta = I
        }
        return x_next;
    };

    auto Phi_zeroCase = [&](const Eigen::VectorXd & xdw, Eigen::MatrixXd & J)
    {
        // xdw = [ x; dw(idxQ) ] where x is [current velocity, current pose, previous pose]
        const Eigen::Index nx = 18; // density.dim();
        const std::vector<Eigen::Index> idxQ = processNoiseIndex();
        const Eigen::Index nq = static_cast<Eigen::Index>(idxQ.size());
        assert(xdw.size() == nx + nq);

        // core state to integrate: first 12 elements (nu(6), eta(6))
        const Eigen::Index coreN = 12;

        // zeta (pose-at-last-flow) follows core, up to 6 elements
        Eigen::Index zetaDim = 6;

        // copy inputs (do not modify xdw)
        const Eigen::VectorXd x = xdw.head(nx);
        const Eigen::VectorXd dw = xdw.tail(nq);

        // build reduced zdw = [ x_core ; dw ]
        Eigen::VectorXd zdw(coreN + nq);
        zdw.setZero();
        zdw.head(coreN) = x.head(coreN); // build top 12 elements
        zdw.tail(nq) = dw; // add noise elements to end

        // integrate reduced system
        Eigen::VectorXd z_next;
    
        // Reconstruct full state x_next
        Eigen::VectorXd x_next = x; // where x = [z; zeta]
        // x_next.head(coreN) = z_next.head(coreN);

        if (isFlowEvent) {
            x_next.segment<6>(12) = x.segment<6>(6); // new zeta is old eta
        } else {
            x_next.segment<6>(12) = x.segment<6>(12); // new zeta is old zeta
        }

        // build full Jacobian: rows nx, cols nx + nq
        J.setZero(nx, nx + nq);

        // place core->core block
        if (coreN > 0) {
            J.block(0, 0, coreN, coreN) = J_temp_.block(0, 0, coreN, coreN);
            // place core->dw block into columns [nx .. nx+nq-1]
            if (nq > 0) {
                J.block(0, nx, coreN, nq) = J_temp_.block(0, coreN, coreN, nq);
            }
        }

        if (isFlowEvent) {
            J.block(coreN, 6, zetaDim, 6).setIdentity();            // dzeta_next/deta = I
            J.block(coreN, coreN, zetaDim, zetaDim).setZero();      // dzeta_next/dzeta = 0
        } else {
            J.block(coreN, 6, zetaDim, 6).setZero();                // dzeta_next/deta = 0
            J.block(coreN, coreN, zetaDim, zetaDim).setIdentity();  // dzeta_next/dzeta = I
        }
        return x_next;
    };
    
    // Map p(x[k], dw(idxQ)[k]) to p(x[k+1])
    if (dt > 0.0) {
        density = pxdw.affineTransform(Phi);
    } else {
        density = pxdw.affineTransform(Phi_zeroCase);
    }

    // Reset flow event flag
    isFlowEvent = false;

    time_ = time;
}

// Templated version of dynamics function for autodiff
template <typename Scalar = double>
Eigen::VectorX<Scalar> dynamicsTemplated(double t, const Eigen::VectorX<Scalar> & x, const Eigen::VectorXd & u, const SystemVisualNav* system) 
{
    Eigen::VectorX<Scalar> f(x.size());
    f.setZero();

    using std::sin, std::cos, std::tan;

    // Extract state components
    const Eigen::VectorX<Scalar> nu = x.segment(0, 6);     // [v_b_B/N; omega_b_B/N]
    const Eigen::VectorX<Scalar> eta = x.segment(6, 6);    // [r_n_B/N; Theta_n_b]
    const Eigen::VectorX<Scalar> v_b = nu.head(3);         // Body-fixed velocities
    const Eigen::VectorX<Scalar> omega_b = nu.tail(3);     // Body-fixed angular velocities
    const Eigen::VectorX<Scalar> Theta_nb = eta.tail(3);   // RPY angles
    
    // Compute kinematic transformation matrices using existing templated functions
    Eigen::Matrix<Scalar, 3, 3> Rnb = rpy2rot(Theta_nb);        
    Eigen::Matrix<Scalar, 3, 3> TK = TKfromThetaTemplated<Scalar>(Theta_nb);     
    
    // Compute f(x) according to equation (4)
    // First 6 states (nu): dnu/dt = 0 (no dynamics, driven by process noise)
    f.segment(0, 6).setZero();
    
    // Next 6 states (eta): deta/dt = JK(eta) * nu
    // f.segment(6, 3) = Rnb * v_b;           // dr_n/dt = R_n_b * v_b
    // f.segment(9, 3) = TK * omega_b;        // dTheta/dt = TK * omega_b
    
    // Using block assignment for templated types
    f.segment(6, 3) = Rnb * v_b;
    f.segment(9, 3) = TK * omega_b;
    
    // Remaining states (landmarks): dm/dt = 0 (static environment assumption)
    // Already set to zero by f.setZero()

    return f;
}

// Evaluate f(x) from the SDE dx = f(x)*dt + dw
Eigen::VectorXd SystemVisualNav::dynamics(double t, const Eigen::VectorXd & x, const Eigen::VectorXd & u) const
{
    return dynamicsTemplated<double>(t, x, u, this);
}

// Evaluate f(x) and its Jacobian J = df/fx from the SDE dx = f(x)*dt + dw
Eigen::VectorXd SystemVisualNav::dynamics(double t, const Eigen::VectorXd & x, const Eigen::VectorXd & u, Eigen::MatrixXd & J) const
{
    Eigen::VectorXd f = dynamics(t, x, u);

    // Jacobian J = df/dx
    //    
    //     [  0                  0 0 ]
    // J = [ JK d(JK(eta)*nu)/deta 0 ]
    //     [  0                  0 0 ]
    //
    J.resize(f.size(), x.size());
    J.setZero();
    // TODO: Implement in Assignment(s)
    
    // Convert input to dual numbers for autodiff
    Eigen::VectorX<autodiff::dual> x_dual = x.cast<autodiff::dual>();
    
    // Create lambda function that captures the necessary parameters
    auto dynamics_lambda = [&](const Eigen::VectorX<autodiff::dual> & x_ad) -> Eigen::VectorX<autodiff::dual> {
        return dynamicsTemplated<autodiff::dual>(t, x_ad, u, this);
    };
    
    // Compute function value and Jacobian using autodiff
    Eigen::VectorX<autodiff::dual> f_dual;
    J = jacobian(dynamics_lambda, wrt(x_dual), at(x_dual), f_dual);
    
    // Extract the function value (convert from dual to double)
    // Eigen::VectorXd f(x.size());
    for (int i = 0; i < x.size(); ++i) {
        f(i) = val(f_dual(i));
    }
    
    return f;
}


Eigen::VectorXd SystemVisualNav::input(double t, const Eigen::VectorXd & x) const
{
    return Eigen::VectorXd(0);
}

GaussianInfo<double> SystemVisualNav::processNoiseDensity(double dt) const
{
    const auto& idxQ = processNoiseIndex();                 // expected: {0,1,2,3,4,5}
    const Eigen::Index nq = static_cast<Eigen::Index>(idxQ.size());
    assert(nq > 0);

    // White-noise on [v, ω] with variance scaling in time
    const double sv = 0.015*1.75; // m/s per sqrt(s)
    const double sw = 0.012*1.5/4; // rad/s per sqrt(s)

    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(nq, nq);
    if (nq >= 3) Q.block<3,3>(0,0).diagonal().setConstant((sv*sv) * dt);
    if (nq >= 6) Q.block<3,3>(3,3).diagonal().setConstant((sw*sw) * dt);

    // sqrt-cov (upper-triangular) for GaussianInfo
    Eigen::LLT<Eigen::MatrixXd> llt(Q);
    if (llt.info() != Eigen::Success) {
        Q += 1e-12 * Eigen::MatrixXd::Identity(nq, nq);
        llt.compute(Q);
    }
    const Eigen::MatrixXd S = llt.matrixU();

    return GaussianInfo<double>::fromSqrtMoment(Eigen::VectorXd::Zero(nq), S);
}

std::vector<Eigen::Index> SystemVisualNav::processNoiseIndex() const
{
    // Indices of process model equations where process noise is injected
    std::vector<Eigen::Index> idxQ;
    idxQ = {0, 1, 2, 3, 4, 5};
    return idxQ;
}

cv::Mat & SystemVisualNav::view()
{
    return view_;
};

const cv::Mat & SystemVisualNav::view() const
{
    return view_;
};

std::size_t SystemVisualNav::numberLandmarks() const
{
    return (density.dim() - 18)/3;
}

std::size_t SystemVisualNav::landmarkPositionIndex(std::size_t idxLandmark) const
{
    assert(idxLandmark < numberLandmarks());
    return 18 + 3*idxLandmark;    
}

GaussianInfo<double> SystemVisualNav::bodyPositionDensity() const
{
    return density.marginal(Eigen::seqN(6, 3));
}

GaussianInfo<double> SystemVisualNav::bodyOrientationDensity() const
{
    return density.marginal(Eigen::seqN(9, 3));
}

GaussianInfo<double> SystemVisualNav::bodyTranslationalVelocityDensity() const
{
    return density.marginal(Eigen::seqN(0, 3));
}

GaussianInfo<double> SystemVisualNav::bodyAngularVelocityDensity() const
{
    return density.marginal(Eigen::seqN(3, 3));
}

#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>

Eigen::Vector3d SystemVisualNav::cameraPosition(const Camera & camera, const Eigen::VectorXd & x, Eigen::MatrixXd & J)
{
    Eigen::Vector3<autodiff::dual> rCNn_dual;
    Eigen::VectorX<autodiff::dual> x_dual = x.cast<autodiff::dual>();
    J = jacobian(cameraPosition<autodiff::dual>, wrt(x_dual), at(camera, x_dual), rCNn_dual);
    return rCNn_dual.cast<double>();
};

GaussianInfo<double> SystemVisualNav::cameraPositionDensity(const Camera & camera) const
{
    auto f = [&](const Eigen::VectorXd & x, Eigen::MatrixXd & J) { return cameraPosition(camera, x, J); };
    return density.affineTransform(f);
}

Eigen::Vector3d SystemVisualNav::cameraOrientationEuler(const Camera & camera, const Eigen::VectorXd & x, Eigen::MatrixXd & J)
{
    Eigen::Vector3<autodiff::dual> Thetanc_dual;
    Eigen::VectorX<autodiff::dual> x_dual = x.cast<autodiff::dual>();
    J = jacobian(cameraOrientationEuler<autodiff::dual>, wrt(x_dual), at(camera, x_dual), Thetanc_dual);
    return Thetanc_dual.cast<double>();
};

GaussianInfo<double> SystemVisualNav::cameraOrientationEulerDensity(const Camera & camera) const
{
    auto f = [&](const Eigen::VectorXd & x, Eigen::MatrixXd & J) { return cameraOrientationEuler(camera, x, J); };
    return density.affineTransform(f);    
}

GaussianInfo<double> SystemVisualNav::landmarkPositionDensity(std::size_t idxLandmark) const
{
    assert(idxLandmark < numberLandmarks());
    std::size_t idx = landmarkPositionIndex(idxLandmark);
    return density.marginal(Eigen::seqN(idx, 3));
}
