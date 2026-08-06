#include <doctest/doctest.h>
#include <cmath>
#include <Eigen/Core>
#include "../../src/GaussianInfo.hpp"
#include "../../src/SensorLog.h"
#include "../../src/SystemLocalisation.h"
#include "stateHelpers.hpp"

SCENARIO("SystemLocalisation dynamics and prediction")
{
    GIVEN("A system at the field origin with an identity-orientation state")
    {
        Eigen::VectorXd eta0 = makeState(0, 0, 0, 0, 0, 0);
        Eigen::MatrixXd S0 = Eigen::MatrixXd::Identity(SystemLocalisation::nx, SystemLocalisation::nx)*0.01;
        auto p0 = GaussianInfo<double>::fromSqrtMoment(eta0, S0);

        WHEN("the state carries a constant forward velocity")
        {
            // The velocity is part of the state now, not an input buffer, so this is
            // set on the mean rather than fed in.
            Eigen::VectorXd etaV = eta0;
            etaV.segment<3>(SystemLocalisation::iVel) << 0.5, 0.0, 0.0;
            etaV.segment<3>(SystemLocalisation::iOmega) << 0.0, 0.0, 0.1;
            auto pV = GaussianInfo<double>::fromSqrtMoment(etaV, S0);
            SystemLocalisation system(pV);

            THEN("dynamics maps the body-fixed velocity states to field rates")
            {
                Eigen::VectorXd f = system.dynamics(0.0, etaV, Eigen::VectorXd());
                REQUIRE(f.size() == SystemLocalisation::nx);
                CHECK(f(0) == doctest::Approx(0.5));
                CHECK(f(1) == doctest::Approx(0.0));
                // qdot = 0.5*Xi(q)*omega. At the identity quaternion (1,0,0,0) a
                // yaw rate of 0.1 shows up as 0.05 on the z component and nowhere
                // else -- there is no "the yaw rate element" any more.
                CHECK(f(SystemLocalisation::iQuat + 3) == doctest::Approx(0.05));
                CHECK(f(SystemLocalisation::iQuat) == doctest::Approx(0.0));
            }

            THEN("the velocity, bias and camera-bias states are pure random walks")
            {
                Eigen::VectorXd f = system.dynamics(0.0, etaV, Eigen::VectorXd());
                CHECK(f.segment<3>(SystemLocalisation::iVel).norm() == doctest::Approx(0.0));
                CHECK(f.segment<3>(SystemLocalisation::iOmega).norm() == doctest::Approx(0.0));
                CHECK(f.segment<3>(SystemLocalisation::iGyroBias).norm() == doctest::Approx(0.0));
                CHECK(f.segment<2>(SystemLocalisation::iBias).norm() == doctest::Approx(0.0));
            }

            THEN("the dynamics Jacobian matches finite differences")
            {
                const int n = SystemLocalisation::nx;
                Eigen::VectorXd x = makeState(0.3, -0.2, 0.0, 0.05, -0.1, 0.7);
                x.segment<3>(SystemLocalisation::iVel) << 0.4, 0.1, -0.02;
                x.segment<3>(SystemLocalisation::iOmega) << 0.01, 0.03, 0.2;

                Eigen::MatrixXd J;
                Eigen::VectorXd f = system.dynamics(0.0, x, Eigen::VectorXd(), J);
                REQUIRE(J.rows() == n);
                REQUIRE(J.cols() == n);

                const double h = 1e-6;
                for (int j = 0; j < n; ++j)
                {
                    Eigen::VectorXd xp = x, xm = x;
                    xp(j) += h;
                    xm(j) -= h;
                    Eigen::VectorXd dfd = (system.dynamics(0.0, xp, Eigen::VectorXd())
                                         - system.dynamics(0.0, xm, Eigen::VectorXd()))/(2*h);
                    for (int i = 0; i < n; ++i)
                    {
                        CHECK(J(i, j) == doctest::Approx(dfd(i)).epsilon(1e-4));
                    }
                }
            }

            THEN("predicting forward moves the mean along field x and grows the covariance")
            {
                double sigma0 = std::sqrt(system.density.cov()(0, 0));
                system.predict(1.0);
                Eigen::VectorXd mu = system.density.mean();
                CHECK(mu(0) == doctest::Approx(0.5).epsilon(0.05));
                CHECK(mu.segment<4>(SystemLocalisation::iQuat).norm() == doctest::Approx(1.0).epsilon(1e-9));
                CHECK(std::sqrt(system.density.cov()(0, 0)) > sigma0);
            }
        }
    }
}

SCENARIO("SystemLocalisation twist from odometry")
{
    GIVEN("Odometry samples of a robot walking straight in world x")
    {
        std::vector<SensorsSample> sensors;
        for (int i = 0; i <= 100; ++i)
        {
            SensorsSample s;
            s.t = 0.01*i;
            // Twt: torso at (0.5t, 0, 0.4), identity orientation. Htw = Twt^{-1}
            Pose<double> Twt;
            Twt.translationVector = Eigen::Vector3d(0.5*s.t, 0, 0.4);
            s.Htw = Twt.inverse();
            s.accelerometer = Eigen::Vector3d(0, 0, 9.8);
            s.gyroscope = Eigen::Vector3d::Zero();
            sensors.push_back(s);
        }

        WHEN("the twist buffer is derived")
        {
            auto twists = SystemLocalisation::twistFromOdometry(sensors, 0.0);

            THEN("body velocity is recovered")
            {
                REQUIRE(twists.size() == 100);
                for (const auto & s : twists)
                {
                    CHECK(s.vBb.x() == doctest::Approx(0.5).epsilon(1e-6));
                    CHECK(s.vBb.y() == doctest::Approx(0.0));
                    CHECK(s.omegaBb.norm() == doctest::Approx(0.0));
                }
            }
        }
    }
}

SCENARIO("Out-of-sequence events are rejected rather than integrated backwards")
{
    GIVEN("A system predicted forward to t = 1 s")
    {
        Eigen::VectorXd eta0 = makeState(0, 0, 0, 0, 0, 0);
        eta0.segment<3>(SystemLocalisation::iVel) << 0.5, 0.0, 0.0;
        Eigen::MatrixXd S0 = Eigen::MatrixXd::Identity(SystemLocalisation::nx, SystemLocalisation::nx)*0.01;
        SystemLocalisation system(GaussianInfo<double>::fromSqrtMoment(eta0, S0));
        system.resetTo(GaussianInfo<double>::fromSqrtMoment(eta0, S0), 0.0);
        system.predict(1.0);

        REQUIRE(system.backwardPredicts() == 0);
        const Eigen::VectorXd muBefore = system.density.mean();
        REQUIRE(muBefore.allFinite());

        WHEN("an event stamped earlier than the filter's clock arrives")
        {
            system.predict(0.75);

            THEN("it is counted, and the belief and clock are left untouched")
            {
                // The whole point: a negative dt puts the process-noise square root at
                // 1/(sigma*sqrt(dt)) and runs RK4 backwards, so an unguarded predict
                // returns NaN. assert() cannot catch it because Release defines NDEBUG.
                CHECK(system.backwardPredicts() == 1);
                CHECK(system.maxBackwardDt() == doctest::Approx(0.25));

                const Eigen::VectorXd muAfter = system.density.mean();
                REQUIRE(muAfter.allFinite());
                CHECK((muAfter - muBefore).norm() == doctest::Approx(0.0));
                CHECK(system.time() == doctest::Approx(1.0));
            }
        }

        WHEN("a correctly ordered event follows")
        {
            system.predict(1.5);

            THEN("prediction proceeds normally and nothing is counted")
            {
                CHECK(system.backwardPredicts() == 0);
                CHECK(system.density.mean().allFinite());
                CHECK(system.time() == doctest::Approx(1.5));
            }
        }
    }
}
