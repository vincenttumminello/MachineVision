#include <algorithm>
#include <cassert>
#include <cmath>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>
#include "GaussianInfo.hpp"
#include "kinematics_helper.h"
#include "rotation.hpp"
#include "SystemEstimator.h"
#include "SystemLocalisation.h"

SystemLocalisation::SystemLocalisation(const GaussianInfo<double> & density, const std::vector<BodyTwistSample> & twistBuffer)
    : SystemEstimator(density)
    , twistBuffer_(&twistBuffer)
{
    assert(density.dim() == nx);
}

SystemLocalisation * SystemLocalisation::clone() const
{
    return new SystemLocalisation(*this);
}

// Templated dynamics for autodiff:
// deta/dt = JK(eta) * nu(t), JK(eta) = blkdiag(Rfb, TK(Theta))
template <typename Scalar>
static Eigen::VectorX<Scalar> dynamicsLocalisationTemplated(const Eigen::VectorX<Scalar> & x, const Eigen::VectorXd & u)
{
    assert(x.size() == SystemLocalisation::nx);
    assert(u.size() == 6);

    const Eigen::VectorX<Scalar> Theta = x.segment(3, 3);
    const Eigen::Matrix3<Scalar> Rfb = rpy2rot(Theta);
    const Eigen::Matrix3<Scalar> TK = TKfromThetaTemplated<Scalar>(Theta);

    const Eigen::Vector3d vBb = u.head<3>();
    const Eigen::Vector3d omegaBb = u.tail<3>();

    Eigen::VectorX<Scalar> f(SystemLocalisation::nx);
    f.setZero();
    f.segment(0, 3) = Rfb * vBb.cast<Scalar>();
    f.segment(3, 3) = TK * omegaBb.cast<Scalar>();
    // Camera mount bias states are a random walk: zero drift
    return f;
}

Eigen::VectorXd SystemLocalisation::dynamics(double t, const Eigen::VectorXd & x, const Eigen::VectorXd & u) const
{
    return dynamicsLocalisationTemplated<double>(x, u);
}

Eigen::VectorXd SystemLocalisation::dynamics(double t, const Eigen::VectorXd & x, const Eigen::VectorXd & u, Eigen::MatrixXd & J) const
{
    using autodiff::dual;
    using autodiff::jacobian;
    using autodiff::wrt;
    using autodiff::at;

    Eigen::VectorX<dual> xdual = x.cast<dual>();
    auto func = [&](const Eigen::VectorX<dual> & xd) -> Eigen::VectorX<dual>
    {
        return dynamicsLocalisationTemplated<dual>(xd, u);
    };

    Eigen::VectorX<dual> fdual;
    J = jacobian(func, wrt(xdual), at(xdual), fdual);

    Eigen::VectorXd f(x.size());
    for (Eigen::Index i = 0; i < f.size(); ++i)
    {
        f(i) = val(fdual(i));
    }
    return f;
}

Eigen::VectorXd SystemLocalisation::input(double t, const Eigen::VectorXd & x) const
{
    // Zero-order-hold lookup of the body twist at time t (zero outside buffer)
    Eigen::VectorXd u = Eigen::VectorXd::Zero(6);
    if (twistBuffer_ == nullptr || twistBuffer_->empty())
    {
        return u;
    }

    const auto & buf = *twistBuffer_;
    auto it = std::upper_bound(buf.begin(), buf.end(), t,
        [](double time, const BodyTwistSample & s) { return time < s.t; });
    if (it == buf.begin())
    {
        return u; // Before first sample
    }
    const BodyTwistSample & s = *std::prev(it);
    u.head<3>() = s.vBb;
    u.tail<3>() = s.omegaBb;
    return u;
}

GaussianInfo<double> SystemLocalisation::processNoiseDensity(double dt) const
{
    // dw ~ N^{-1}(0, LambdaQ/dt), i.e., cov(dw) = Q*dt
    Eigen::VectorXd sigma(nx);
    sigma << params.sigmaPosXY, params.sigmaPosXY, params.sigmaPosZ,
             params.sigmaAtt, params.sigmaAtt, params.sigmaYaw,
             params.sigmaCamBias, params.sigmaCamBias;

    Eigen::MatrixXd XiQ = Eigen::MatrixXd::Zero(nx, nx);
    XiQ.diagonal() = (sigma*std::sqrt(dt)).cwiseInverse();
    return GaussianInfo<double>::fromSqrtInfo(XiQ);
}

std::vector<Eigen::Index> SystemLocalisation::processNoiseIndex() const
{
    return {0, 1, 2, 3, 4, 5, 6, 7};
}

std::vector<BodyTwistSample> SystemLocalisation::twistFromOdometry(const std::vector<SensorsSample> & sensors, double t0, double maxGap)
{
    std::vector<BodyTwistSample> twists;
    if (sensors.size() < 2)
    {
        return twists;
    }
    twists.reserve(sensors.size() - 1);

    for (std::size_t k = 1; k < sensors.size(); ++k)
    {
        const SensorsSample & a = sensors[k-1];
        const SensorsSample & b = sensors[k];
        const double dt = b.t - a.t;
        if (dt <= 0 || dt > maxGap)
        {
            continue;
        }

        // Torso pose in world: Twt = Htw^{-1}
        Pose<double> Twta = Pose<double>(a.Htw.rotationMatrix, a.Htw.translationVector).inverse();
        Pose<double> Twtb = Pose<double>(b.Htw.rotationMatrix, b.Htw.translationVector).inverse();

        if (!Twta.translationVector.allFinite() || !Twtb.translationVector.allFinite() ||
            !Twta.rotationMatrix.allFinite() || !Twtb.rotationMatrix.allFinite())
        {
            continue;
        }

        // Relative pose of torso(b) w.r.t. torso(a): body-frame increment
        Pose<double> dT = Twta.inverse()*Twtb;

        Eigen::AngleAxisd aa(dT.rotationMatrix);

        BodyTwistSample s;
        s.t = 0.5*(a.t + b.t) - t0;
        s.vBb = dT.translationVector/dt;
        s.omegaBb = aa.angle()*aa.axis()/dt;
        if (!s.vBb.allFinite() || !s.omegaBb.allFinite())
        {
            continue;
        }
        twists.push_back(s);
    }
    return twists;
}

void SystemLocalisation::resetTo(const GaussianInfo<double> & newDensity, double time)
{
    assert(newDensity.dim() == nx);
    density = newDensity;
    time_ = time;
}

GaussianInfo<double> SystemLocalisation::positionDensity() const
{
    return density.marginal(Eigen::seqN(0, 3));
}

GaussianInfo<double> SystemLocalisation::orientationDensity() const
{
    return density.marginal(Eigen::seqN(3, 3));
}
