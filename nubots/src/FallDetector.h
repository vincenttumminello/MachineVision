/**
 * @file FallDetector.h
 * @brief Posture timeline (upright / falling / fallen / getting up) for a recorded log.
 */
#ifndef FALLDETECTOR_H
#define FALLDETECTOR_H

#include <cstddef>
#include <string>
#include <vector>
#include "SensorLog.h"

/**
 * @brief Coarse posture of the robot.
 *
 * Only the distinction between UPRIGHT and everything else drives the filter;
 * the finer states exist so the diagnostics say which phase a suppressed window
 * belonged to.
 */
enum class Posture
{
    UPRIGHT,        ///< Standing or walking; kinematics, gravity and vision are all usable
    FALLING,        ///< Toppling: attitude is slewing fast and the support polygon is gone
    FALLEN,         ///< On the ground
    GETTING_UP      ///< Executing a getup, or settling immediately after one
};

const char * to_string(Posture p);

/**
 * @class FallDetector
 * @brief Resolves, for any time in a recorded log, whether the robot was upright.
 *
 * Almost every assumption the localiser makes is an upright-robot assumption:
 * MeasurementGravity assumes the accelerometer reads gravity (quasi-static),
 * MeasurementKinematicHeight assumes the support-leg chain measures torso height
 * above the ground, MeasurementFieldLandmarks assumes the predicted bearings are
 * within a 0.35 rad gate of the measured ones, and SideDisambiguator assumes the
 * pose used to triangulate background corners is roughly right. A fall breaks all
 * four at once, so the filter needs to know when it is happening and stop
 * believing them.
 *
 * Two sources, in order of preference:
 *
 *  1. The robot's own stability flags (SensorLog::stability), when the log has
 *     them. This is the authoritative signal: the behaviour module knows it has
 *     entered a getup script, which no amount of IMU inspection can infer.
 *
 *  2. Otherwise a detector derived from the accelerometer. Averaged over a short
 *     window the specific force is dominated by gravity, so the angle between it
 *     and the torso z axis gives torso tilt without needing the kinematic chain
 *     (which, in the recorded logs, keeps reporting a near-upright torso and a
 *     0.44 m torso height even through a 34 deg lean -- it cannot be trusted to
 *     report a fall).
 *
 * The timeline is built once, up front, and queried by time; this mirrors how
 * the rest of the replay pipeline consumes the log and keeps the per-frame cost
 * to a binary search.
 */
class FallDetector
{
public:
    /**
     * @brief Thresholds for the derived (accelerometer) detector and the common settle tail.
     */
    struct Options
    {
        double tiltEnter    = 50.0*M_PI/180.0;  ///< Smoothed tilt that declares a fall [rad]
        double tiltExit     = 25.0*M_PI/180.0;  ///< Smoothed tilt at or below which the robot is upright again [rad]
        double smoothWindow = 0.30;             ///< Half-width of the accelerometer averaging window [s]
        double minFallen    = 0.40;             ///< Tilt must stay past tiltEnter this long to count as a fall [s]
                                                ///< (walking already spikes the raw accelerometer past 90 deg
                                                ///< for single samples, so a transient must not trip it)
        double settle       = 1.00;             ///< Treat the robot as still recovering for this long after it
                                                ///< is upright again [s]: the getup script's final stand is not
                                                ///< where the walk engine's odometry becomes meaningful again
    };

    /**
     * @brief Build a posture timeline for a log.
     * @param sensors   Time-ordered IMU samples (absolute time [s])
     * @param stability Time-ordered stability flags; empty falls back to the derived detector
     * @param t0        Time origin subtracted from sample times [s]
     * @param options   Detector thresholds
     */
    FallDetector(const std::vector<SensorsSample> & sensors,
                 const std::vector<StabilitySample> & stability,
                 double t0,
                 const Options & options);

    /// As above with default thresholds (a defaulted argument cannot name Options here).
    FallDetector(const std::vector<SensorsSample> & sensors,
                 const std::vector<StabilitySample> & stability,
                 double t0);

    /**
     * @brief Posture at time @p t (relative to t0). UPRIGHT outside the timeline's span.
     */
    Posture at(double t) const;

    /**
     * @brief True when the posture came from the robot's own flags rather than the IMU.
     */
    bool usingFlags() const { return usingFlags_; }

    /**
     * @brief A contiguous non-upright window, for reporting.
     */
    struct Interval
    {
        double start;       ///< First non-upright time [s]
        double end;         ///< Time the robot is upright again (end of the settle tail) [s]
    };

    /**
     * @brief The non-upright windows found, in time order.
     */
    const std::vector<Interval> & intervals() const { return intervals_; }

    /**
     * @brief State names that did not match any known posture, deduplicated.
     *
     * Non-empty means the log used an enum spelling this class does not know, and
     * those samples were treated as upright. The caller is expected to report it:
     * a silently unrecognised "FALLEN" would disable every protection here.
     */
    const std::vector<std::string> & unrecognisedStates() const { return unrecognised_; }

    /**
     * @brief Map a stability-flag state name onto a posture.
     * @param state Raw state name from the log (case-insensitive)
     * @param known Set to false if the name matched nothing
     */
    static Posture postureFromStateName(std::string_view state, bool & known);

    Options options;

private:
    /// (time, posture) breakpoints; posture holds until the next entry.
    struct Entry
    {
        double t;
        Posture posture;
    };
    std::vector<Entry> timeline_;
    std::vector<Interval> intervals_;
    std::vector<std::string> unrecognised_;
    bool usingFlags_ = false;

    void buildFromFlags(const std::vector<StabilitySample> & stability, double t0);
    void buildFromAccelerometer(const std::vector<SensorsSample> & sensors, double t0);
    void buildIntervals();
};

#endif
