/**
 * @file FlowTracker.h
 * @brief Sparse optical flow between consecutive video frames, as far-field ray correspondences.
 *
 * The rotational component of the optical flow field is independent of depth:
 * for a feature at infinite range the bearing transforms by the inter-frame
 * camera rotation alone,
 *
 *     u_c(k) = R_c(k)c(k-1) * u_c(k-1),
 *
 * with no translation term at all. That is what makes flow worth carrying here.
 * The gyroscope sees omegaBb + bGyro and cannot separate the two; flow sees the
 * rotation itself, with no bias of its own, so the difference between them is
 * the bias -- and the yaw-rate component of that bias is the one the upstream
 * Mahony filter is structurally blind to and the one that turns into heading
 * drift. See MeasurementFlowRotation for the measurement built on top of this.
 *
 * The depth independence only holds for distant features, so this class hands
 * back only the correspondences it believes are far: the same test the
 * out-of-field pipeline uses (at or above the horizon, or intersecting the
 * ground plane beyond the carpet). A carpet feature 1.5 m away shifts by d/Z --
 * about 3 px per frame at walking pace -- which is several times the tracking
 * noise and would pull the fit.
 *
 * Tracking is pyramidal Lucas-Kanade with a forward-backward consistency check.
 * Features persist across frames and are topped up only when the surviving
 * count falls below a threshold, so the Shi-Tomasi detector does not run on
 * every frame.
 */
#ifndef FLOWTRACKER_H
#define FLOWTRACKER_H

#include <vector>
#include <Eigen/Core>
#include <opencv2/core.hpp>
#include "CameraLens.h"
#include "FieldMap.h"
#include "OutOfFieldFeatures.h"
#include "Pose.hpp"

/**
 * @brief One feature tracked from the previous frame into the current one.
 */
struct FlowMatch
{
    Eigen::Vector3d uPrev;      ///< Unit ray in the previous frame's camera frame {c(k-1)}
    Eigen::Vector3d uCurr;      ///< Unit ray in the current frame's camera frame {c(k)}
    Eigen::Vector2d pxPrev;     ///< Pixel position in the previous frame (for the viewer)
    Eigen::Vector2d pxCurr;     ///< Pixel position in the current frame (for the viewer)
};

/**
 * @brief The correspondences between one pair of consecutive frames.
 *
 * `valid` is false on the first frame, after a tracking dropout, and whenever
 * the frame interval falls outside the window the constant-rate model is good
 * for. A caller that sees `valid == false` should simply not build a
 * measurement; the tracker has still advanced its internal state.
 */
struct FlowFrame
{
    bool valid = false;                 ///< True if matches are usable as a measurement
    double dt = 0.0;                    ///< Interval between the two frames [s]
    Eigen::Matrix3d RbcPrev = Eigen::Matrix3d::Identity();  ///< Body-from-camera rotation at k-1
    Eigen::Matrix3d RbcCurr = Eigen::Matrix3d::Identity();  ///< Body-from-camera rotation at k
    std::vector<FlowMatch> matches;     ///< Far-field correspondences

    // Per-frame tallies, for the run summary.
    int nTracked = 0;                   ///< Survived LK and the forward-backward check
    int nNear = 0;                      ///< ... of which discarded as near-field (parallax)
    int nDetected = 0;                  ///< New features added by the detector this frame
};

/**
 * @brief Tracks sparse features between consecutive frames and classifies them by range.
 */
class FlowTracker
{
public:
    /**
     * @brief Detection, tracking and gating options.
     */
    struct Options
    {
        int maxFeatures       = 400;    ///< Cap on tracked features
        int redetectBelow     = 250;    ///< Top up the feature set when fewer than this survive
        double qualityLevel   = 0.01;   ///< Shi-Tomasi quality, relative to the best corner
        double minDistance    = 12.0;   ///< Minimum spacing between features [px]
        int imageBorder       = 20;     ///< Reject features within this many pixels of the edge

        int lkWindow          = 21;     ///< Lucas-Kanade search window (square) [px]
        int lkPyramidLevels   = 3;      ///< Pyramid levels (0 = no pyramid)
        double fbThreshold    = 1.0;    ///< Max forward-backward round-trip error [px]

        // The constant-rate model behind the measurement integrates omegaBb over
        // the interval, so a long gap is not merely a weaker measurement but a
        // wrong one; and Lucas-Kanade itself fails on the large displacements a
        // long gap produces. Frames are nominally ~30 Hz, so the window is
        // generous either side of that and only excludes genuine dropouts.
        double minDt          = 0.005;  ///< Below this the rotation is lost in tracking noise [s]
        double maxDt          = 0.20;   ///< Above this the constant-rate model does not hold [s]

        std::size_t minMatches = 20;    ///< Fewer far-field matches than this is not a measurement

        /// @brief Range classification (shared with the out-of-field pipeline).
        OutOfFieldDetector::Options farField{};
    };

    /**
     * @brief Construct the tracker.
     * @param lens Lens model, for pixel <-> ray conversion
     * @param dims Field dimensions defining the carpet extent (for the far-field test)
     * @param options Detection, tracking and gating options
     */
    FlowTracker(const CameraLens & lens, const FieldDimensions & dims, const Options & options);

    /// @brief Construct with default options.
    FlowTracker(const CameraLens & lens, const FieldDimensions & dims);

    /**
     * @brief Track into a new frame and return the far-field correspondences.
     *
     * Always advances the tracker, whether or not the result is usable.
     *
     * @param gray Grayscale frame (CV_8UC1, full lens resolution)
     * @param t Frame capture time [s]
     * @param Tbc Camera pose w.r.t. the torso at this frame (from kinematics)
     * @param Tfc Estimated camera pose in {f}, used only to classify features by range
     * @return The correspondences, with `valid` set if they are usable
     */
    FlowFrame track(const cv::Mat & gray, double t, const Pose<double> & Tbc, const Pose<double> & Tfc);

    /**
     * @brief Discard the tracking history.
     *
     * Call when the video stream jumps (a seek, or frames skipped): the next
     * frame then re-detects rather than tracking across the discontinuity.
     */
    void reset();

    Options options;

private:
    /// @brief Top up prevPts_ with new corners, keeping clear of the ones already tracked.
    void detect(const cv::Mat & gray);

    const CameraLens & lens_;
    OutOfFieldDetector farField_;       ///< Supplies the at-or-beyond-the-carpet test

    cv::Mat prevGray_;                  ///< Previous frame (empty until the first track())
    std::vector<cv::Point2f> prevPts_;  ///< Features held in the previous frame
    double prevT_ = 0.0;                ///< Previous frame time [s]
    Eigen::Matrix3d prevRbc_ = Eigen::Matrix3d::Identity();  ///< Body-from-camera rotation at the previous frame
};

#endif
