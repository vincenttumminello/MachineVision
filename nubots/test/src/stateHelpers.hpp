/**
 * @file stateHelpers.hpp
 * @brief Building localisation states in tests now that attitude is a quaternion.
 *
 * The state carries (w, x, y, z) at iQuat, so a test can no longer write an
 * attitude by streaming three Euler angles into the vector. These wrap the
 * conversion so tests keep reading in roll/pitch/yaw, which is what they are
 * actually about.
 */
#ifndef TEST_STATEHELPERS_HPP
#define TEST_STATEHELPERS_HPP

#include <Eigen/Core>
#include "../../src/rotation.hpp"
#include "../../src/SystemLocalisation.h"

/// @brief State vector from a pose in roll-pitch-yaw plus camera-mount bias.
inline Eigen::VectorXd makeState(double x, double y, double z,
                                 double roll, double pitch, double yaw,
                                 double biasRoll = 0.0, double biasPitch = 0.0)
{
    Eigen::VectorXd s = Eigen::VectorXd::Zero(SystemLocalisation::nx);
    s.head<3>() << x, y, z;
    s.segment<4>(SystemLocalisation::iQuat) = rpy2quat(Eigen::Vector3d(roll, pitch, yaw));
    s(SystemLocalisation::iBias) = biasRoll;
    s(SystemLocalisation::iBias + 1) = biasPitch;
    return s;
}

/// @brief Roll, pitch and yaw of a state's attitude [rad].
inline Eigen::Vector3d stateRpy(const Eigen::VectorXd & s)
{
    return SystemLocalisation::attitudeRpy(s);
}

/// @brief Heading of a state [rad].
inline double stateYaw(const Eigen::VectorXd & s)
{
    return SystemLocalisation::heading(s);
}

#endif
