#include <doctest/doctest.h>
#include <cmath>
#include <Eigen/Core>
#include "../../src/CameraLens.h"

SCENARIO("CameraLens project/unproject round-trip")
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
        CameraLens lens;
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
        CameraLens lens;

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
        CameraLens lens;
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
        CameraLens lens;
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

SCENARIO("The webots calibration models a rectilinear camera")
{
    GIVEN("The webots calibration")
    {
        const CameraLens * webots = lensByName("webots");
        REQUIRE(webots != nullptr);

        THEN("it is the undistorted 640x480 pinhole from left_camera.yaml")
        {
            CHECK(webots->projection == LensProjection::Rectilinear);
            CHECK(webots->width == 640.0);
            CHECK(webots->height == 480.0);
            CHECK(webots->focalLength == doctest::Approx(0.5));
            CHECK(webots->k.isZero());
        }

        THEN("the frame size selects it, and 1280x1024 selects a fisheye")
        {
            REQUIRE(lensForResolution(640.0, 480.0) != nullptr);
            CHECK(lensForResolution(640.0, 480.0)->name == "webots");
            REQUIRE(lensForResolution(1280.0, 1024.0) != nullptr);
            CHECK(lensForResolution(1280.0, 1024.0)->projection == LensProjection::Equidistant);
            CHECK(lensForResolution(1920.0, 1080.0) == nullptr);
        }

        THEN("the horizontal edges of the image sit at +/- 45 deg")
        {
            // fieldOfView 1.5707 rad in nugus.proto, i.e. a 90 deg horizontal FOV
            const Eigen::Vector3d left  = webots->unproject(Eigen::Vector2d(0.0, webots->height*0.5));
            const Eigen::Vector3d right = webots->unproject(Eigen::Vector2d(webots->width, webots->height*0.5));
            CHECK(std::atan2(left.y(), left.x())   == doctest::Approx(M_PI/4.0).epsilon(1e-9));
            CHECK(std::atan2(right.y(), right.x()) == doctest::Approx(-M_PI/4.0).epsilon(1e-9));
        }

        THEN("project and unproject are exact inverses (no distortion to approximate)")
        {
            for (double thetaDeg = 0.0; thetaDeg <= 55.0; thetaDeg += 5.0)
                for (double phiDeg = 0.0; phiDeg < 360.0; phiDeg += 45.0)
                {
                    const double theta = thetaDeg*M_PI/180.0, phi = phiDeg*M_PI/180.0;
                    Eigen::Vector3d ray(std::cos(theta), std::sin(theta)*std::cos(phi), std::sin(theta)*std::sin(phi));
                    ray.normalize();
                    const Eigen::Vector3d back = webots->unproject(webots->project(ray));
                    CHECK(back.x() == doctest::Approx(ray.x()).epsilon(1e-9));
                    CHECK(back.y() == doctest::Approx(ray.y()).epsilon(1e-9));
                    CHECK(back.z() == doctest::Approx(ray.z()).epsilon(1e-9));
                }
        }

        THEN("a ray at the horizon stays finite and lands outside the image")
        {
            // tan(90 deg) is infinite; the model clamps rather than emitting nan,
            // so inImage() can still reject the pixel.
            const Eigen::Vector2d px = webots->project(Eigen::Vector3d(1e-4, 1.0, 0.0).normalized());
            CHECK(std::isfinite(px.x()));
            CHECK(std::isfinite(px.y()));
            CHECK_FALSE(webots->inImage(px));
        }
    }

    GIVEN("The same ray through the fisheye and the pinhole")
    {
        THEN("they disagree by degrees, so the calibration cannot be shared")
        {
            const CameraLens fisheye = *lensByName("sarah");
            const CameraLens pinhole = *lensByName("webots");
            // 40 deg off axis, expressed as a fraction of each image width
            const double theta = 40.0*M_PI/180.0;
            const Eigen::Vector3d ray(std::cos(theta), std::sin(theta), 0.0);
            const double rFisheye = (fisheye.project(ray).x() - fisheye.width*0.5)/fisheye.width;
            const double rPinhole = (pinhole.project(ray).x() - pinhole.width*0.5)/pinhole.width;
            CHECK(std::abs(rFisheye - rPinhole) > 0.05);
        }
    }
}
