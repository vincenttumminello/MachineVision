#include <doctest/doctest.h>
#include <cmath>
#include <Eigen/Core>
#include "../../src/FisheyeLens.h"

SCENARIO("FisheyeLens project/unproject round-trip")
{
    // A grid of rays within the ~150 deg FOV (theta up to ~70 deg)
    auto forEachRay = [](auto && fn) {
        for (double thetaDeg = 0.0; thetaDeg <= 70.0; thetaDeg += 10.0)
            for (double phiDeg = 0.0; phiDeg < 360.0; phiDeg += 45.0)
            {
                const double theta = thetaDeg*M_PI/180.0;
                const double phi = phiDeg*M_PI/180.0;
                // x = optical axis, (y,z) span the perpendicular plane
                Eigen::Vector3d ray(std::cos(theta),
                                    std::sin(theta)*std::cos(phi),
                                    std::sin(theta)*std::sin(phi));
                ray.normalize();
                fn(ray);
            }
    };

    GIVEN("A distortion-free lens")
    {
        FisheyeLens lens;
        lens.k = Eigen::Vector2d::Zero();   // Pure equidistant: project/unproject are exact inverses

        THEN("project then unproject recovers each ray exactly")
        {
            forEachRay([&](const Eigen::Vector3d & ray) {
                Eigen::Vector3d back = lens.unproject(lens.project(ray));
                CHECK(back.x() == doctest::Approx(ray.x()).epsilon(1e-9));
                CHECK(back.y() == doctest::Approx(ray.y()).epsilon(1e-9));
                CHECK(back.z() == doctest::Approx(ray.z()).epsilon(1e-9));
            });
        }
    }

    GIVEN("The default distorted lens")
    {
        FisheyeLens lens;

        THEN("project then unproject recovers each ray to sub-degree accuracy")
        {
            // NUbots' distort/undistort are polynomial approximate inverses
            // (~0.2 px), so the recovered ray is close but not exact.
            forEachRay([&](const Eigen::Vector3d & ray) {
                Eigen::Vector3d back = lens.unproject(lens.project(ray));
                const double cosAngle = std::clamp(back.dot(ray), -1.0, 1.0);
                CHECK(std::acos(cosAngle) < 0.5*M_PI/180.0);   // < 0.5 deg
            });
        }
    }

    GIVEN("The optical axis")
    {
        FisheyeLens lens;
        THEN("it projects to the optical centre and unprojects back to +x")
        {
            Eigen::Vector2d px = lens.project(Eigen::Vector3d::UnitX());
            CHECK(px.x() == doctest::Approx(lens.width*0.5 - lens.centre.x()*lens.width));
            CHECK(px.y() == doctest::Approx(lens.height*0.5 - lens.centre.y()*lens.width));

            Eigen::Vector3d ray = lens.unproject(Eigen::Vector2d(lens.width*0.5 - lens.centre.x()*lens.width,
                                                                 lens.height*0.5 - lens.centre.y()*lens.width));
            CHECK(ray.x() == doctest::Approx(1.0));
        }
    }

    GIVEN("A recorded L-intersection ray from the log")
    {
        FisheyeLens lens;
        // First detection of the first BoundingBoxes sample (top-left corner)
        Eigen::Vector3d ray(0.545586823007872, 0.7825421633584279, -0.2999379621296521);
        ray.normalize();

        THEN("it projects to a pixel inside the 1280x1024 image")
        {
            Eigen::Vector2d px = lens.project(ray);
            CHECK(lens.inFrontOfCamera(ray));
            CHECK(px.x() >= 0.0);
            CHECK(px.x() < lens.width);
            CHECK(px.y() >= 0.0);
            CHECK(px.y() < lens.height);
        }
    }
}
