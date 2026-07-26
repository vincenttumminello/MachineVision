/**
 * @file CameraLens.h
 * @brief NUbots-compatible camera projection (ray <-> pixel) and the named
 *        calibrations of the units that produced the recordings.
 *
 * Re-implements the projection used by the NUbots vision pipeline
 * (shared/utility/vision/projection.hpp) so that unit rays in the camera frame
 * {c} recorded in the log can be drawn back onto the source video, and pixels
 * can be unprojected to rays (out-of-field landmark work).
 *
 * Camera frame convention (NUbots): x is the optical axis (viewing direction),
 * y points to the left of the image, z points up. Pixel coordinates have (0,0)
 * at the top-left, x to the right, y down.
 *
 * The lens parameters (focal length, centre offset and distortion coefficients)
 * are normalised by the image width, exactly as in the NUbots camera configs, so
 * a calibration here is a transcription of one YAML file:
 *
 *   real robots  module/input/Camera/data/config/<robot>/Cameras/Left.yaml
 *   webots       module/platform/Webots/data/config/WebotsCameras/left_camera.yaml
 *
 * The projection model is NOT the same across those: the real robots wear a
 * Lensagon BF10M19828S118C fisheye (EQUIDISTANT, strongly distorted, 1280x1024),
 * while the simulated camera is a plain 90 deg rectilinear pinhole with no
 * distortion at 640x480 (NUWebots protos/robot/nugus/nugus.proto sets the Camera
 * node's `spherical FALSE` and `fieldOfView 1.5707`, whence focal_length =
 * (640/2)/tan(1.5707/2)/640 = 0.5). Replaying a webots recording through a
 * fisheye calibration therefore does not merely shift the re-projection, it
 * bends it -- which is why the calibration is selected per dataset rather than
 * hardcoded. See lensForResolution() and the --lens option in main.cpp.
 */
#ifndef CAMERALENS_H
#define CAMERALENS_H

#include <algorithm>
#include <array>
#include <cmath>
#include <format>
#include <string>
#include <string_view>
#include <Eigen/Core>

/// @brief How the incidence angle theta maps to radius on the sensor.
enum class LensProjection
{
    Equidistant,    ///< r = f*theta        (the NUbots fisheye)
    Rectilinear     ///< r = f*tan(theta)   (a pinhole; webots' simulated camera)
};

/**
 * @brief Camera lens model with radial distortion (NUbots-compatible).
 *
 * Defaults to sarah's fisheye calibration, the robot that captured the data and
 * data2 ground-truth recordings. Prefer naming a calibration explicitly
 * (lensByName / lensForResolution) over relying on this default.
 */
struct CameraLens
{
    std::string name = "sarah";   ///< Which calibration this is, for logging

    double width  = 1280.0;     ///< Image width [px]
    double height = 1024.0;     ///< Image height [px]

    LensProjection projection = LensProjection::Equidistant;

    // All of the following are normalised by the image width, per the NUbots convention.
    double focalLength = 0.34690945742400775;                           ///< Normalised focal length
    Eigen::Vector2d centre{0.02072339174622414, -0.0011612242293956145};  ///< Normalised optical-centre offset
    Eigen::Vector2d k{0.38553542593448015, 0.1498415334589703};         ///< Radial distortion coefficients [k1, k2]

    /// @brief Inverse-distortion polynomial coefficients (undistorted radius -> distorted radius).
    Eigen::Vector4d inverseCoefficients() const
    {
        const double k0 = k(0), k1 = k(1);
        return Eigen::Vector4d(
            -k0,
            3.0*k0*k0 - k1,
            -12.0*k0*k0*k0 + 8.0*k0*k1,
            55.0*k0*k0*k0*k0 - 55.0*k0*k0*k1 + 5.0*k1*k1);
    }

    /// @brief Map an ideal (undistorted) radius to the distorted radius (used when projecting).
    double distort(double r) const
    {
        const Eigen::Vector4d ik = inverseCoefficients();
        const double r2 = r*r;
        return r*(1.0 + ik(0)*r2 + ik(1)*r2*r2 + ik(2)*r2*r2*r2 + ik(3)*r2*r2*r2*r2);
    }

    /// @brief Map a distorted radius back to the ideal radius (used when unprojecting).
    double undistort(double r) const
    {
        const double r2 = r*r;
        return r*(1.0 + k(0)*r2 + k(1)*r2*r2);
    }

    /// @brief Undistorted radius for an incidence angle, per the projection model.
    double radiusForAngle(double theta) const
    {
        if (projection == LensProjection::Rectilinear)
        {
            // tan diverges at 90 deg. Clamping just short keeps the radius (and
            // hence the pixel, and hence the distortion polynomial) finite; the
            // result lands far outside the image, so inImage() still rejects it.
            constexpr double kMaxTheta = 89.9*M_PI/180.0;
            return focalLength*std::tan(std::clamp(theta, 0.0, kMaxTheta));
        }
        return focalLength*theta;
    }

    /// @brief Incidence angle for an undistorted radius, per the projection model.
    double angleForRadius(double r) const
    {
        return projection == LensProjection::Rectilinear ? std::atan(r/focalLength)
                                                         : r/focalLength;
    }

    /**
     * @brief Project a unit ray in {c} to a pixel coordinate (x right, y down).
     * @param ray Unit vector in the camera frame (x optical axis, y left, z up)
     * @return Pixel coordinate; check inFrontOfCamera()/inImage() for validity
     */
    Eigen::Vector2d project(const Eigen::Vector3d & ray) const
    {
        const double x = std::clamp(ray.x(), -1.0, 1.0);
        const double theta = std::acos(x);
        const double sinTheta = std::sqrt(std::max(1.0 - x*x, 1e-12));

        const double rUndist = radiusForAngle(theta);
        const double rDist = distort(rUndist);       // Normalised distorted radius

        // Screen offset (normalised by width) in left/up axes, then to pixels.
        const double scale = (sinTheta > 1e-9 ? rDist/sinTheta : 0.0)*width;
        const double screenLeft = scale*ray.y();
        const double screenUp   = scale*ray.z();

        return Eigen::Vector2d(width*0.5  - screenLeft - centre.x()*width,
                               height*0.5 - screenUp   - centre.y()*width);
    }

    /**
     * @brief Unproject a pixel coordinate (x right, y down) to a unit ray in {c}.
     */
    Eigen::Vector3d unproject(const Eigen::Vector2d & px) const
    {
        const double screenLeft = width*0.5  - px.x() - centre.x()*width;
        const double screenUp    = height*0.5 - px.y() - centre.y()*width;
        const double rDist = std::sqrt(screenLeft*screenLeft + screenUp*screenUp)/width;
        if (rDist <= 0.0)
        {
            return Eigen::Vector3d::UnitX();
        }
        const double rUndist = undistort(rDist);
        const double theta = angleForRadius(rUndist);
        const double sinTheta = std::sin(theta);
        const double norm = std::sqrt(screenLeft*screenLeft + screenUp*screenUp);
        return Eigen::Vector3d(std::cos(theta),
                               sinTheta*screenLeft/norm,
                               sinTheta*screenUp/norm);
    }

    /// @brief True if the ray points into the camera's forward hemisphere.
    static bool inFrontOfCamera(const Eigen::Vector3d & ray) { return ray.x() > 1e-3; }

    /// @brief True if a pixel lies within the image bounds.
    bool inImage(const Eigen::Vector2d & px) const
    {
        return px.x() >= 0.0 && px.x() < width && px.y() >= 0.0 && px.y() < height;
    }

    /// @brief One-line description for logging.
    std::string describe() const
    {
        return std::format("{} ({}, {}x{}, f={:.5f}, centre=[{:.5f}, {:.5f}], k=[{:.5f}, {:.5f}])",
                           name,
                           projection == LensProjection::Rectilinear ? "RECTILINEAR" : "EQUIDISTANT",
                           static_cast<int>(width), static_cast<int>(height),
                           focalLength, centre.x(), centre.y(), k(0), k(1));
    }
};

/**
 * @brief The calibrations transcribed from the NUbots configs.
 *
 * Adding a robot is a matter of copying projection/focal_length/centre/k out of
 * its Left.yaml and the sensor size out of its `settings:` block.
 */
inline const std::array<CameraLens, 5> & lensCatalogue()
{
    static const std::array<CameraLens, 5> catalogue{{
        // <robot>/Cameras/Left.yaml -- Lensagon BF10M19828S118C on a FLIR BFS-U3-13Y3C-C
        {"sarah",   1280.0, 1024.0, LensProjection::Equidistant, 0.34690945742400775,
         {0.02072339174622414, -0.0011612242293956145}, {0.38553542593448015, 0.1498415334589703}},
        {"frankie", 1280.0, 1024.0, LensProjection::Equidistant, 0.34,
         {0.02072339174622414, -0.0011612242293956145}, {0.38553542593448015, 0.1498415334589703}},
        {"kevin",   1280.0, 1024.0, LensProjection::Equidistant, 0.34607022208588906,
         {-0.008429632815260277, 0.003767345048721181}, {0.4718296027381958, 0.22726744036437643}},
        {"billie",  1280.0, 1024.0, LensProjection::Equidistant, 0.39,
         {-0.01370165657132854, -0.013645629277120971}, {0.4667339, 0.150171181}},
        // WebotsCameras/left_camera.yaml -- an undistorted 90 deg pinhole, not a fisheye
        {"webots",   640.0,  480.0, LensProjection::Rectilinear, 0.5,
         {0.0, 0.0}, {0.0, 0.0}},
    }};
    return catalogue;
}

/// @brief Look a calibration up by name; nullptr if there is no such name.
inline const CameraLens * lensByName(std::string_view name)
{
    for (const CameraLens & lens : lensCatalogue())
    {
        if (lens.name == name) return &lens;
    }
    return nullptr;
}

/// @brief Comma-separated list of the available calibration names, for help text.
inline std::string lensNames()
{
    std::string names;
    for (const CameraLens & lens : lensCatalogue())
    {
        if (!names.empty()) names += ", ";
        names += lens.name;
    }
    return names;
}

/**
 * @brief Guess the calibration from the recorded frame size.
 *
 * The two sources in play have distinct sensor sizes (1280x1024 for the real
 * cameras, 640x480 for the simulated one), so the video itself says which family
 * of calibration a recording needs. It cannot say *which robot* took a real
 * recording -- that returns sarah, whose recordings this project was built on.
 * Pass --lens for anything else.
 *
 * @return The matching calibration, or nullptr if the size matches none of them
 */
inline const CameraLens * lensForResolution(double width, double height)
{
    for (const CameraLens & lens : lensCatalogue())
    {
        if (lens.width == width && lens.height == height) return &lens;
    }
    return nullptr;
}

#endif
