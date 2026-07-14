/**
 * @file SensorLog.h
 * @brief Ingestion of a recorded NUbots humanoid soccer sensor log (NDJSON) plus video frame
 *        alignment for a synchronised playback video.
 *
 * The recorded log is a newline-delimited JSON file where every line is one message of the form
 * @code
 * {"type": "<message type>", "timestamp": <int, microseconds since epoch>, "data": {...}}
 * @endcode
 * Only six message types are of interest for state estimation and are parsed here:
 *   - message.input.Sensors
 *   - message.vision.BoundingBoxes
 *   - message.behaviour.state.WalkState
 *   - message.localisation.Field
 *   - message.vision.FieldLines
 *   - message.input.MotionCapture (OptiTrack ground truth; evaluation only)
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
 * @brief A single message.input.MotionCapture sample (OptiTrack rigid-body ground truth).
 *
 * The pose is the raw Motive rigid body in the mocap frame {m}; the mapping to the
 * field frame (and the rigid-body-to-torso extrinsics) is applied by the consumer,
 * since it is a property of the capture-volume calibration, not of the log format.
 * Ground truth is for evaluation/visualisation only and must never reach the estimator.
 */
struct MocapSample
{
    double t;                       ///< receive time [s since epoch]
    Eigen::Vector3d position;       ///< rigid-body position in {m} [m]
    Eigen::Matrix3d R;              ///< rigid-body orientation in {m} (from the streamed quaternion)
    bool valid = false;             ///< OptiTrack trackingValid flag
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
 * Only six message types are parsed (message.input.Sensors, message.vision.BoundingBoxes,
 * message.behaviour.state.WalkState, message.localisation.Field, message.vision.FieldLines,
 * message.input.MotionCapture); all other message types are skipped cheaply. Each resulting
 * stream is time-ordered.
 */
class SensorLog
{
public:
    /**
     * @brief Load and parse a recorded sensor log and a matching video timecode file
     * @param jsonPath      Path to the NDJSON recorded sensor log
     * @param timecodePath  Path to the video timecode file (one Unix timestamp in seconds per line,
     *                      one line per video frame, in frame order)
     */
    SensorLog(const std::filesystem::path & jsonPath, const std::filesystem::path & timecodePath);

    std::vector<SensorsSample> sensors;              ///< time-ordered
    std::vector<VisionSample> vision;                ///< time-ordered
    std::vector<WalkStateSample> walk;                ///< time-ordered
    std::vector<FieldBaselineSample> fieldBaseline;  ///< time-ordered
    std::vector<LinePointsSample> linePoints;        ///< time-ordered
    std::vector<MocapSample> mocap;                  ///< time-ordered (empty if the log has no mocap)
    std::vector<double> frameTimes;                  ///< [s since epoch] per video frame
    double t0 = 0;                                    ///< earliest sample time across all streams [s]
};

#endif
