/**
 * @file FisheyeLens.h
 * @brief NUbots-compatible fisheye camera projection (ray <-> pixel).
 *
 * Re-implements the projection used by the NUbots vision pipeline
 * (shared/utility/vision/projection.hpp) so that unit rays in the camera frame
 * {c} recorded in the log can be drawn back onto the source video, and pixels
 * can be unprojected to rays (for future out-of-field landmark work).
 *
 * Camera frame convention (NUbots): x is the optical axis (viewing direction),
 * y points to the left of the image, z points up. Pixel coordinates have (0,0)
 * at the top-left, x to the right, y down.
 *
 * The lens parameters (focal length, centre offset and distortion coefficients)
 * are normalised by the image width, exactly as in the NUbots camera configs
 * (module/input/Camera/data/config/<robot>/Cameras/Left.yaml). The default
 * values are frankie's calibration, which is the robot that captured the
 * recording used here: the ground-truth lens embedded in the recorded
 * CompressedImage messages (projection EQUIDISTANT, focal_length 0.34,
 * centre [0.02072, -0.00116], k [0.38554, 0.14984]) matches frankie's config
 * exactly. Using another unit's calibration (e.g. kevin's) shifts and scales
 * the projection so re-projected detections no longer line up with the frame.
 */
#ifndef FISHEYELENS_H
#define FISHEYELENS_H

#include <algorithm>
#include <cmath>
#include <Eigen/Core>

/**
 * @brief Equidistant fisheye lens model with radial distortion (NUbots-compatible).
 */
struct FisheyeLens
{
    double width  = 1280.0;     ///< Image width [px]
    double height = 1024.0;     ///< Image height [px]

    // All of the following are normalised by the image width, per the NUbots convention.
    // Values are frankie's Left.yaml calibration (the robot that made the recording).
    double focalLength = 0.34;                                        ///< Normalised focal length
    Eigen::Vector2d centre{0.02072339174622414, -0.0011612242293956145}; ///< Normalised optical-centre offset
    Eigen::Vector2d k{0.38553542593448015, 0.1498415334589703};      ///< Radial distortion coefficients [k1, k2]

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

        const double rUndist = focalLength*theta;   // Equidistant projection
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
        const double theta = rUndist/focalLength;    // Equidistant inverse
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
};

#endif
