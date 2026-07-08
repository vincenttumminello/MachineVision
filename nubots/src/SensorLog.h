/**
 * @file SensorLog.h
 * @brief Ingestion of a recorded NUbots humanoid soccer sensor log (NDJSON) plus video frame
 *        alignment for a synchronised playback video.
 *
 * The recorded log is a newline-delimited JSON file where every line is one message of the form
 * @code
 * {"type": "<message type>", "timestamp": <int, microseconds since epoch>, "data": {...}}
 * @endcode
 * Only five message types are of interest for state estimation and are parsed here:
 *   - message.input.Sensors
 *   - message.vision.BoundingBoxes
 *   - message.behaviour.state.WalkState
 *   - message.localisation.Field
 *   - message.vision.FieldLines
 *
 * @see SensorLog
 */

#ifndef SENSORLOG_H
#define SENSORLOG_H

#include <filesystem>
#include <string>
#include <vector>
#include <Eigen/Core>
#include "Pose.hpp"

/**
 * @brief A single message.input.Sensors sample (IMU + torso pose estimate at capture time)
 */
struct SensorsSample
{
    double t;                       ///< capture time [s since epoch]
    Pose<double> Htw;                ///< world -> torso
    Eigen::Vector3d accelerometer;  ///< [m/s^2], torso frame
    Eigen::Vector3d gyroscope;      ///< [rad/s], torso frame
};

/**
 * @brief A single detected bounding box within a message.vision.BoundingBoxes sample
 */
struct Detection
{
    std::string name;                    ///< YOLO class name
    double confidence;
    Eigen::Matrix<double, 3, 4> corners; ///< unit rays in camera frame {c}; columns TL, TR, BR, BL
};

/**
 * @brief A single message.vision.BoundingBoxes sample (camera pose + detections at capture time)
 */
struct VisionSample
{
    double t;                       ///< image capture time [s since epoch]
    int videoFrame;                 ///< index into frameTimes / Left.mp4, or -1 if no matching frame
    Pose<double> Hcw;                ///< world -> camera
    std::vector<Detection> detections;
};

/**
 * @brief A single message.behaviour.state.WalkState sample
 */
struct WalkStateSample
{
    double t;
    std::string state;
    Eigen::Vector3d velocityTarget; ///< vx [m/s], vy [m/s], wz [rad/s]
};

/**
 * @brief A single message.localisation.Field sample (NUbots' own NLopt-based estimate, for comparison)
 */
struct FieldBaselineSample
{
    double t;
    Pose<double> Hfw;                ///< world -> field
    double cost;
};

/**
 * @brief A single message.vision.FieldLines sample (field-line points as camera rays)
 */
struct LinePointsSample
{
    double t;                                       ///< image capture time [s since epoch]
    int videoFrame;                                 ///< index into frameTimes / Left.mp4, or -1
    Pose<double> Hcw;                               ///< world -> camera
    Eigen::Matrix<double, 3, Eigen::Dynamic> rays;  ///< unit rays in camera frame {c}, one column per field-line point
};

/**
 * @brief Parses a recorded NUbots sensor log (NDJSON) and aligns vision samples to video frames.
 *
 * Only five message types are parsed (message.input.Sensors, message.vision.BoundingBoxes,
 * message.behaviour.state.WalkState, message.localisation.Field, message.vision.FieldLines);
 * all other message types are skipped cheaply. Each resulting stream is time-ordered.
 */
class SensorLog
{
public:
    /**
     * @brief Load and parse a recorded sensor log and a matching video timecode file
     * @param jsonPath      Path to the NDJSON recorded sensor log
     * @param timecodePath  Path to the video timecode file (one float per line, ms since video start)
     */
    SensorLog(const std::filesystem::path & jsonPath, const std::filesystem::path & timecodePath);

    std::vector<SensorsSample> sensors;              ///< time-ordered
    std::vector<VisionSample> vision;                ///< time-ordered
    std::vector<WalkStateSample> walk;                ///< time-ordered
    std::vector<FieldBaselineSample> fieldBaseline;  ///< time-ordered
    std::vector<LinePointsSample> linePoints;        ///< time-ordered
    std::vector<double> frameTimes;                  ///< [s since epoch] per video frame
    double t0 = 0;                                    ///< earliest sample time across all streams [s]
};

#endif
