#include <doctest/doctest.h>
#include <cmath>
#include <vector>
#include <Eigen/Core>
#include "../../src/GaussianInfo.hpp"
#include "../../src/kinematics_helper.h"
#include "../../src/SystemLocalisation.h"

namespace
{

GaussianInfo<double> tightBelief(const Eigen::VectorXd & mu, double std = 0.05)
{
    Eigen::MatrixXd S = Eigen::MatrixXd::Identity(SystemLocalisation::nx, SystemLocalisation::nx)*std;
    return GaussianInfo<double>::fromSqrtMoment(mu, S);
}

Eigen::VectorXd nominalState()
{
    Eigen::VectorXd x(SystemLocalisation::nx);
    x << 1.5, -0.8, 0.44, 0.0, 0.0, 0.3, 0.0, 0.0;
    return x;
}

/// A constant body twist over a long span, so predict() always finds an input.
std::vector<BodyTwistSample> constantTwist(const Eigen::Vector3d & v, const Eigen::Vector3d & w)
{
    std::vector<BodyTwistSample> buf;
    for (double t = -1.0; t < 100.0; t += 0.01)
    {
        buf.push_back({t, v, w});
    }
    return buf;
}

} // namespace

SCENARIO("The Euler-rate transform stays finite through a fall")
{
    GIVEN("Pitch at the singularity a forward fall passes through")
    {
        THEN("TK is finite at exactly 90 deg")
        {
            // Unguarded this is tan(pi/2) and 1/cos(pi/2): the dynamics Jacobian
            // becomes infinite, RK4SDEHelper propagates that into the predicted
            // covariance, and the Newton update is handed a NaN prior -- which
            // would poison the filter for the rest of the run, not just the fall.
            Eigen::VectorXd theta(3);
            theta << 0.1, 0.5*M_PI, -0.4;
            Eigen::Matrix3d TK = TKfromTheta(theta);
            CHECK(TK.allFinite());
        }
        THEN("TK is finite just past the singularity, on both sides")
        {
            for (double pitch : {0.5*M_PI - 1e-12, 0.5*M_PI + 1e-9, -0.5*M_PI, -0.5*M_PI - 1e-9})
            {
                Eigen::VectorXd theta(3);
                theta << 0.1, pitch, -0.4;
                CHECK(TKfromTheta(theta).allFinite());
            }
        }
    }

    GIVEN("Pitch anywhere in the range an upright robot reaches")
    {
        THEN("the clamp never binds, so upright behaviour is unchanged")
        {
            // The guard saturates |cos(pitch)| at 1e-3, i.e. beyond 89.94 deg.
            for (double pitchDeg = -80.0; pitchDeg <= 80.0; pitchDeg += 5.0)
            {
                const double pitch = pitchDeg*M_PI/180.0;
                Eigen::VectorXd theta(3);
                theta << 0.2, pitch, 0.5;
                Eigen::Matrix3d TK = TKfromTheta(theta);
                CHECK(TK(0, 1) == doctest::Approx(std::sin(0.2)*std::tan(pitch)));
                CHECK(TK(2, 2) == doctest::Approx(std::cos(0.2)/std::cos(pitch)));
            }
        }
    }
}

SCENARIO("Disturbed mode changes what prediction believes")
{
    GIVEN("A filter walking forward at 0.3 m/s with a tight belief")
    {
        std::vector<BodyTwistSample> twists = constantTwist(Eigen::Vector3d(0.3, 0.0, 0.0),
                                                            Eigen::Vector3d::Zero());
        const Eigen::VectorXd x0 = nominalState();

        WHEN("two seconds are predicted while upright")
        {
            SystemLocalisation sys(tightBelief(x0), twists);
            sys.resetTo(tightBelief(x0), 0.0);
            sys.predictAll(2.0);
            const Eigen::VectorXd mu = sys.density.mean();
            const Eigen::MatrixXd P = sys.density.cov();

            THEN("the odometry twist carries the estimate forward")
            {
                // 0.3 m/s for 2 s along the body x axis, rotated into {f} by yaw 0.3.
                CHECK(mu(0) == doctest::Approx(x0(0) + 0.6*std::cos(0.3)).epsilon(0.02));
                CHECK(mu(1) == doctest::Approx(x0(1) + 0.6*std::sin(0.3)).epsilon(0.02));
            }
            THEN("uncertainty grows at the walking process noise")
            {
                CHECK(std::sqrt(P(0, 0)) < 0.2);
            }
        }

        WHEN("the same two seconds are predicted while not upright")
        {
            SystemLocalisation sys(tightBelief(x0), twists);
            sys.resetTo(tightBelief(x0), 0.0);
            sys.setDisturbed(true);
            sys.predictAll(2.0);
            const Eigen::VectorXd mu = sys.density.mean();
            const Eigen::MatrixXd P = sys.density.cov();

            THEN("the walk-engine velocity is discarded rather than integrated")
            {
                // On the ground the odometry describes a gait that is not happening.
                CHECK(mu(0) == doctest::Approx(x0(0)).epsilon(1e-6));
                CHECK(mu(1) == doctest::Approx(x0(1)).epsilon(1e-6));
            }
            THEN("the belief decays to honest uncertainty instead of coasting")
            {
                CHECK(std::sqrt(P(0, 0)) > 0.4);
                CHECK(std::sqrt(P(5, 5)) > 0.4);
            }
        }

        WHEN("the robot is disturbed but rotating")
        {
            // The gyroscope measures the topple for real, so unlike the odometry
            // velocity it is still integrated.
            std::vector<BodyTwistSample> spin = constantTwist(Eigen::Vector3d(0.3, 0.0, 0.0),
                                                              Eigen::Vector3d(0.0, 0.0, 0.5));
            SystemLocalisation sys(tightBelief(x0), spin);
            sys.resetTo(tightBelief(x0), 0.0);
            sys.setDisturbed(true);
            sys.predictAll(1.0);

            THEN("yaw follows the gyroscope")
            {
                CHECK(sys.density.mean()(5) == doctest::Approx(x0(5) + 0.5).epsilon(0.05));
            }
        }
    }
}

SCENARIO("Recovery from a fall widens the belief without moving it")
{
    GIVEN("A confident single-hypothesis belief")
    {
        const Eigen::VectorXd x0 = nominalState();
        SystemLocalisation sys(tightBelief(x0), {});
        sys.resetTo(tightBelief(x0), 0.0);
        const Eigen::MatrixXd P0 = sys.density.cov();

        WHEN("the recovery inflation is applied")
        {
            Eigen::VectorXd extra = Eigen::VectorXd::Zero(SystemLocalisation::nx);
            extra(0) = extra(1) = 0.25;         // 0.5 m std
            extra(5) = 1.0;                     // 1 rad std
            sys.inflateCovariance(extra);

            THEN("the mean is untouched")
            {
                // The pre-fall position is still the best estimate available; a
                // fall and getup move the torso well under a metre.
                CHECK((sys.density.mean() - x0).norm() == doctest::Approx(0.0).epsilon(1e-9));
            }
            THEN("exactly the requested variance is added")
            {
                const Eigen::MatrixXd P = sys.density.cov();
                CHECK(P(0, 0) - P0(0, 0) == doctest::Approx(0.25));
                CHECK(P(5, 5) - P0(5, 5) == doctest::Approx(1.0));
            }
            THEN("untouched states keep their confidence")
            {
                // Roll and pitch are re-fixed by the very next gravity update, and
                // the camera mount bias is a property of the kinematic chain, not
                // of the posture.
                const Eigen::MatrixXd P = sys.density.cov();
                CHECK(P(3, 3) == doctest::Approx(P0(3, 3)));
                CHECK(P(6, 6) == doctest::Approx(P0(6, 6)));
            }
            THEN("the position std exceeds the disambiguator's map-building gate")
            {
                // SideDisambiguator::Options::maxPosStd is 0.5 m, so recovering
                // also freezes background-map building until the filter
                // reconverges -- no landmarks get triangulated from a bad pose.
                CHECK(std::sqrt(sys.density.cov()(0, 0)) > 0.5);
            }
        }
    }

    GIVEN("An active hypothesis bank")
    {
        const Eigen::VectorXd x0 = nominalState();
        SystemLocalisation sys(tightBelief(x0), {});
        sys.resetTo(tightBelief(x0), 0.0);
        sys.initialiseHypotheses();
        REQUIRE(sys.numHypotheses() == 2);

        WHEN("the recovery inflation is applied")
        {
            Eigen::VectorXd extra = Eigen::VectorXd::Zero(SystemLocalisation::nx);
            extra(0) = extra(1) = 0.25;
            sys.inflateCovariance(extra);

            THEN("every hypothesis is widened, not just the representative")
            {
                REQUIRE(sys.numHypotheses() == 2);
                for (const GaussianInfo<double> & c : sys.hypotheses())
                {
                    CHECK(std::sqrt(c.cov()(0, 0)) > 0.5);
                }
            }
            THEN("the representative still matches the leading component")
            {
                CHECK((sys.density.mean() - sys.hypotheses().front().mean()).norm()
                      == doctest::Approx(0.0).epsilon(1e-9));
            }
        }
    }
}

SCENARIO("predictAll advances every hypothesis over the same interval")
{
    GIVEN("A bank of two hypotheses and a stationary robot")
    {
        const Eigen::VectorXd x0 = nominalState();
        std::vector<BodyTwistSample> twists = constantTwist(Eigen::Vector3d::Zero(),
                                                            Eigen::Vector3d::Zero());
        SystemLocalisation sys(tightBelief(x0), twists);
        sys.resetTo(tightBelief(x0), 0.0);
        sys.initialiseHypotheses();
        const Eigen::VectorXd mirror0 = sys.hypotheses()[1].mean();

        WHEN("a measurement-free interval is predicted")
        {
            sys.predictAll(3.0);

            THEN("both components keep their means and both lose confidence")
            {
                REQUIRE(sys.numHypotheses() == 2);
                CHECK((sys.hypotheses()[0].mean() - x0).norm() == doctest::Approx(0.0).epsilon(1e-6));
                CHECK((sys.hypotheses()[1].mean() - mirror0).norm() == doctest::Approx(0.0).epsilon(1e-6));
                for (const GaussianInfo<double> & c : sys.hypotheses())
                {
                    CHECK(std::sqrt(c.cov()(0, 0)) > std::sqrt(0.05*0.05));
                }
            }
        }
    }
}
