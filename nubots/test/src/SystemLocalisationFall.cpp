#include <doctest/doctest.h>
#include <cmath>
#include <vector>
#include <Eigen/Core>
#include "../../src/GaussianInfo.hpp"
#include "../../src/kinematics_helper.h"
#include "../../src/MeasurementBodyRates.h"
#include "../../src/SystemLocalisation.h"
#include "stateHelpers.hpp"

namespace
{

GaussianInfo<double> tightBelief(const Eigen::VectorXd & mu, double std = 0.05)
{
    Eigen::MatrixXd S = Eigen::MatrixXd::Identity(SystemLocalisation::nx, SystemLocalisation::nx)*std;
    return GaussianInfo<double>::fromSqrtMoment(mu, S);
}

Eigen::VectorXd nominalState()
{
    return makeState(1.5, -0.8, 0.44, 0.0, 0.0, 0.3);
}

/// A state moving at a given body-fixed twist. The twist is part of the state now,
/// not an input buffer -- that is the whole point of the velocity states.
Eigen::VectorXd movingState(const Eigen::Vector3d & v, const Eigen::Vector3d & w = Eigen::Vector3d::Zero())
{
    Eigen::VectorXd x = nominalState();
    x.segment<3>(SystemLocalisation::iVel) = v;
    x.segment<3>(SystemLocalisation::iOmega) = w;
    return x;
}

} // namespace

// The localisation filter no longer uses the roll-pitch-yaw rate transform --
// its attitude is a quaternion, whose kinematics have no singularity. TKfromTheta
// is still used by SystemVisualNav, so the guard is still worth holding to; what
// these cases no longer cover is the fall path, which is exercised by the
// quaternion cases below instead.
SCENARIO("The Euler-rate transform stays finite at gimbal lock")
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

SCENARIO("The velocity states carry the estimate, and measurements steer them")
{
    GIVEN("A belief moving at 0.3 m/s along the body x axis")
    {
        const Eigen::VectorXd x0 = movingState(Eigen::Vector3d(0.3, 0.0, 0.0));

        WHEN("two seconds are predicted")
        {
            SystemLocalisation sys(tightBelief(x0));
            sys.resetTo(tightBelief(x0), 0.0);
            sys.predictAll(2.0);
            const Eigen::VectorXd mu = sys.density.mean();

            THEN("the velocity state carries the position forward")
            {
                // 0.3 m/s for 2 s along body x, rotated into {f} by yaw 0.3. Nothing
                // was fed in: the motion comes from the state, which is what makes
                // the odometry a measurement rather than an assertion.
                CHECK(mu(0) == doctest::Approx(x0(0) + 0.6*std::cos(0.3)).epsilon(0.05));
                CHECK(mu(1) == doctest::Approx(x0(1) + 0.6*std::sin(0.3)).epsilon(0.05));
            }
        }

        WHEN("a zero-velocity update is applied")
        {
            SystemLocalisation sys(tightBelief(x0));
            sys.resetTo(tightBelief(x0), 0.0);
            MeasurementBodyVelocity zupt = MeasurementBodyVelocity::stationary(0.0, 0.01);
            sys.process(zupt);

            THEN("the velocity state is pulled to a stop")
            {
                CHECK(SystemLocalisation::bodyVelocity(sys.density.mean()).norm() < 0.05);
            }
            THEN("and the estimate then stays put under prediction")
            {
                // This is the property that keeps a fallen robot on the spot. Without
                // it the pre-fall gait integrates for the whole fall.
                const Eigen::VectorXd before = sys.density.mean();
                sys.predictAll(4.0);
                CHECK((sys.density.mean().head<2>() - before.head<2>()).norm() < 0.05);
            }
        }
    }

    GIVEN("A belief with a spin about the body z axis")
    {
        const Eigen::VectorXd x0 = movingState(Eigen::Vector3d::Zero(), Eigen::Vector3d(0.0, 0.0, 0.5));

        WHEN("one second is predicted")
        {
            SystemLocalisation sys(tightBelief(x0));
            sys.resetTo(tightBelief(x0), 0.0);
            sys.predictAll(1.0);

            THEN("the heading follows the angular velocity state")
            {
                CHECK(stateYaw(sys.density.mean()) == doctest::Approx(stateYaw(x0) + 0.5).epsilon(0.05));
            }
        }
    }
}

SCENARIO("The gyroscope measures the rate, and its bias is observable")
{
    GIVEN("A stationary belief and a gyroscope reading a pure bias")
    {
        // A robot held still while the gyro reads 0.02 rad/s about z. The reading is
        // omegaBb + bGyro, so on its own it cannot say which of the two it is --
        // that is exactly why the bias needs something else to pin the rate.
        const Eigen::VectorXd x0 = nominalState();
        SystemLocalisation sys(tightBelief(x0, 0.02));
        sys.resetTo(tightBelief(x0, 0.02), 0.0);

        WHEN("the gyroscope reading is applied with the rate held at zero")
        {
            // Standing in for what the landmark measurements do: they pin the
            // attitude, and hence the rate, leaving the difference to the bias.
            for (int i = 1; i <= 200; ++i)
            {
                const double t = 0.01*i;
                MeasurementGyroscope mg(t, Eigen::Vector3d(0.0, 0.0, 0.02), 0.02);
                sys.process(mg);
                MeasurementBodyVelocity zupt = MeasurementBodyVelocity::stationary(t, 0.01);
                sys.process(zupt);
            }

            THEN("the reading is explained by the rate and bias together")
            {
                const Eigen::VectorXd mu = sys.density.mean();
                const double predicted = SystemLocalisation::bodyRate(mu).z()
                                       + SystemLocalisation::gyroBias(mu).z();
                CHECK(predicted == doctest::Approx(0.02).epsilon(0.2));
            }
        }
    }
}

SCENARIO("Recovery from a fall widens the belief without moving it")
{
    GIVEN("A confident single-hypothesis belief")
    {
        const Eigen::VectorXd x0 = nominalState();
        SystemLocalisation sys(tightBelief(x0));
        sys.resetTo(tightBelief(x0), 0.0);
        const Eigen::MatrixXd P0 = sys.density.cov();

        WHEN("the recovery inflation is applied")
        {
            // Built exactly as runFieldLocalisation builds it: position on the
            // diagonal, yaw as a rank-one block about the field z axis.
            const double yawStd = 1.0;
            Eigen::MatrixXd extra = Eigen::MatrixXd::Zero(SystemLocalisation::nx, SystemLocalisation::nx);
            extra(0, 0) = extra(1, 1) = 0.25;   // 0.5 m std
            const Eigen::Vector4d jYaw = SystemLocalisation::attitudeTangentField(x0).col(2);
            extra.block<4, 4>(SystemLocalisation::iQuat, SystemLocalisation::iQuat)
                = yawStd*yawStd*jYaw*jYaw.transpose();
            const double yawVar0 = SystemLocalisation::yawVariance(x0, P0);
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
                // The whole point of the rank-one form: the yaw uncertainty the
                // filter reports grows by exactly what was asked for, even though
                // it is spread over four components and no element is "yaw".
                CHECK(SystemLocalisation::yawVariance(sys.density.mean(), P) - yawVar0
                      == doctest::Approx(yawStd*yawStd).epsilon(1e-6));
            }
            THEN("roll and pitch are left alone")
            {
                // A yaw inflation must not smear into the other two attitude axes:
                // they are re-fixed by the very next gravity update and inflating
                // them would only make that update fight a wider prior.
                const Eigen::Matrix3d A0 = SystemLocalisation::attitudeCovariance(x0, P0);
                const Eigen::Matrix3d A = SystemLocalisation::attitudeCovariance(sys.density.mean(),
                                                                                 sys.density.cov());
                CHECK(A(0, 0) == doctest::Approx(A0(0, 0)).epsilon(1e-6));
                CHECK(A(1, 1) == doctest::Approx(A0(1, 1)).epsilon(1e-6));
            }
            THEN("the camera mount bias keeps its confidence")
            {
                // A property of the kinematic chain, not of the posture.
                const Eigen::MatrixXd P = sys.density.cov();
                CHECK(P(SystemLocalisation::iBias, SystemLocalisation::iBias)
                      == doctest::Approx(P0(SystemLocalisation::iBias, SystemLocalisation::iBias)));
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
        SystemLocalisation sys(tightBelief(x0));
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
        SystemLocalisation sys(tightBelief(x0));
        sys.resetTo(tightBelief(x0), 0.0);
        sys.initialiseHypotheses();
        const Eigen::VectorXd mirror0 = sys.hypotheses()[1].mean();

        WHEN("a measurement-free interval is predicted")
        {
            sys.predictAll(3.0);

            THEN("both components keep their means and both lose confidence")
            {
                REQUIRE(sys.numHypotheses() == 2);
                // Compare poses, not raw components: predict renormalises the
                // quaternion into w >= 0, and q and -q are the same rotation, so
                // a component-wise comparison can fail on a sign that means nothing.
                CHECK((sys.hypotheses()[0].mean().head<3>() - x0.head<3>()).norm()
                      == doctest::Approx(0.0).epsilon(1e-6));
                CHECK((sys.hypotheses()[1].mean().head<3>() - mirror0.head<3>()).norm()
                      == doctest::Approx(0.0).epsilon(1e-6));
                CHECK((stateRpy(sys.hypotheses()[0].mean()) - stateRpy(x0)).norm() < 1e-6);
                CHECK((stateRpy(sys.hypotheses()[1].mean()) - stateRpy(mirror0)).norm() < 1e-6);
                for (const GaussianInfo<double> & c : sys.hypotheses())
                {
                    CHECK(std::sqrt(c.cov()(0, 0)) > std::sqrt(0.05*0.05));
                }
            }
        }
    }
}
