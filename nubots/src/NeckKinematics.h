/**
 * @file NeckKinematics.h
 * @brief The torso-from-camera extrinsic, built from the head servos rather than differenced poses.
 *
 * Everything else in this pipeline recovers the extrinsic as
 *
 *     Tbc = Htw * Hcw^-1
 *
 * which is correct and, for anything consumed once per frame, accurate enough.
 * It has one property that matters here: both factors carry the world-referenced
 * torso attitude, and the two messages they come from are not stamped at the
 * same instant. The world rotation therefore cancels only to the extent that the
 * clocks agree, and what survives is a spurious rotation of order
 * (body rate) x (clock skew). On `data2` that skew runs to a few milliseconds
 * and the residual reaches 0.5 rad/s of apparent head motion -- on a recording
 * whose head servos do not move at all, through the entire 108 s, by more than
 * 1.3 degrees.
 *
 * Against a 0.25 rad landmark sigma that is invisible. Against the rotation
 * between two consecutive video frames -- a median of 6.9 px, about 15 mrad --
 * it is the larger of the two signals, and a flow model built on it measures
 * mostly clock skew. The joints have none of that problem: they are small, slow,
 * and read once.
 *
 * ## The convention, and how it was established
 *
 *     Rbc = Rz(headYaw) * Ry(headPitch) * Rmount
 *
 * with NUbots ServoID 18 = HEAD_YAW and 19 = HEAD_PITCH. This was not assumed
 * from the enum: over the 856 vision frames of `data`, which contains a 104 deg
 * yaw and 29 deg pitch head scan, the quantity Rneck^T * (Htw*Hcw^-1) is
 * constant to a median 0.02 deg under this form. The alternatives are not close
 * -- 8.4 deg for a flipped pitch, 34 deg for a flipped yaw, and 19.8 deg for
 * ignoring the neck entirely -- so the fit identifies the chain rather than
 * merely being consistent with it. The p90 of 1.0 deg is the clock skew above,
 * which is precisely what this class exists to keep out of the flow model.
 *
 * `Rmount` is calibrated per recording rather than hard-coded, so a different
 * robot (or a camera remounted between sessions) needs no code change. On `data`
 * it comes out within 1.7 deg of the identity, i.e. the camera frame is very
 * nearly the head frame.
 */
#ifndef NECKKINEMATICS_H
#define NECKKINEMATICS_H

#include <vector>
#include <Eigen/Core>
#include "Pose.hpp"
#include "rotation.hpp"
#include "SensorLog.h"

/**
 * @brief Torso-from-camera rotation from the head servos, with a calibrated mount.
 */
class NeckKinematics
{
public:
    /**
     * @brief The neck chain alone: yaw about the torso z axis, then pitch about the rotated y.
     * @param yaw HEAD_YAW present position [rad]
     * @param pitch HEAD_PITCH present position [rad]
     */
    static Eigen::Matrix3d neckRotation(double yaw, double pitch)
    {
        return rotz(yaw)*roty(pitch);
    }

    /**
     * @brief Calibrate the fixed head-to-camera mount against the logged extrinsic.
     *
     * Averages Rneck^T * (Htw * Hcw^-1) over the frames where the robot is
     * turning slowly, since that is where the clock skew the logged extrinsic
     * suffers from is smallest. The chordal mean is re-orthonormalised, which is
     * the standard projection of an averaged rotation back onto SO(3).
     *
     * @param sensors Sensor stream (needs headValid)
     * @param vision Vision stream (supplies Hcw and the frame times to pair on)
     * @param maxRate Body rate above which a frame is too skewed to calibrate on [rad/s]
     * @return Number of frames the calibration used; 0 leaves the mount at identity
     */
    std::size_t calibrate(const std::vector<SensorsSample> & sensors,
                          const std::vector<VisionSample> & vision,
                          double maxRate = 0.1);

    /**
     * @brief Torso-from-camera rotation at a sensor sample.
     *
     * Falls back to the calibrated mount alone if the sample carries no head
     * servo data, which is the right answer for a log that predates the field
     * or a robot with a fixed head.
     */
    Eigen::Matrix3d Rbc(const SensorsSample & s) const
    {
        return s.headValid ? Eigen::Matrix3d(neckRotation(s.headYaw, s.headPitch)*mount_) : mount_;
    }

    /// @brief The calibrated head-to-camera mount rotation.
    const Eigen::Matrix3d & mount() const { return mount_; }

    /// @brief True if calibrate() found usable frames.
    bool calibrated() const { return calibrated_; }

    /**
     * @brief RMS angle between the servo-derived extrinsic and the logged one, after calibration.
     *
     * Reported by the run summary. A small value says the two agree and the
     * servo chain is sound; the value it does NOT go to is zero, because the
     * logged extrinsic carries the clock skew and the servo one does not.
     */
    double residualRms() const { return residualRms_; }

private:
    Eigen::Matrix3d mount_ = Eigen::Matrix3d::Identity();
    bool calibrated_ = false;
    double residualRms_ = 0.0;
};

#endif
