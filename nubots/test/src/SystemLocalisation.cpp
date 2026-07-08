#include <doctest/doctest.h>
#include <cmath>
#include <Eigen/Core>
#include "../../src/GaussianInfo.hpp"
#include "../../src/SensorLog.h"
#include "../../src/SystemLocalisation.h"

SCENARIO("SystemLocalisation dynamics and prediction")
{
    GIVEN("A system at the field origin with an identity-orientation state")
    {
        Eigen::VectorXd eta0 = Eigen::VectorXd::Zero(6);
        Eigen::MatrixXd S0 = Eigen::MatrixXd::Identity(6, 6)*0.01;
        auto p0 = GaussianInfo<double>::fromSqrtMoment(eta0, S0);

        WHEN("the twist buffer holds a constant forward velocity")
        {
            std::vector<BodyTwistSample> twists;
            for (int i = 0; i < 200; ++i)
            {
                BodyTwistSample s;
                s.t = 0.01*i;
                s.vBb = Eigen::Vector3d(0.5, 0, 0);
                s.omegaBb = Eigen::Vector3d::Zero();
                twists.push_back(s);
            }
            SystemLocalisation system(p0, twists);

            THEN("dynamics at zero attitude maps body twist to field rates directly")
            {
                Eigen::VectorXd u(6);
                u << 0.5, 0, 0, 0, 0, 0.1;
                Eigen::VectorXd f = system.dynamics(0.0, eta0, u);
                CHECK(f(0) == doctest::Approx(0.5));
                CHECK(f(1) == doctest::Approx(0.0));
                CHECK(f(5) == doctest::Approx(0.1));
            }

            THEN("the dynamics Jacobian matches finite differences")
            {
                Eigen::VectorXd x(6);
                x << 0.3, -0.2, 0.0, 0.05, -0.1, 0.7;
                Eigen::VectorXd u(6);
                u << 0.4, 0.1, -0.02, 0.01, 0.03, 0.2;

                Eigen::MatrixXd J;
                Eigen::VectorXd f = system.dynamics(0.0, x, u, J);
                REQUIRE(J.rows() == 6);
                REQUIRE(J.cols() == 6);

                const double h = 1e-6;
                for (int j = 0; j < 6; ++j)
                {
                    Eigen::VectorXd xp = x, xm = x;
                    xp(j) += h;
                    xm(j) -= h;
                    Eigen::VectorXd dfd = (system.dynamics(0.0, xp, u) - system.dynamics(0.0, xm, u))/(2*h);
                    for (int i = 0; i < 6; ++i)
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
                CHECK(mu(0) == doctest::Approx(0.5).epsilon(0.02));
                CHECK(std::abs(mu(1)) < 0.01);
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
