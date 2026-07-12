/**
 * @file LocalisationViewer.h
 * @brief Two-panel interactive visualiser for the field-localisation pipeline.
 *
 * Left panel: the source camera frame with YOLO detections, field-line points,
 * associated map landmarks and the predicted horizon re-projected onto the
 * fisheye image (NUbots-exact projection). Right panel: a top-down field view
 * with the estimated pose, every live hypothesis and its weight, the posterior
 * covariance, the NUbots baseline and the on-ground landmark residuals.
 *
 * The pipeline first records a lightweight ViewerFrame per processed vision
 * sample (no images), then this viewer replays them, seeking the video per
 * frame so the user can scrub forwards and backwards with bounded memory.
 */
#ifndef LOCALISATIONVIEWER_H
#define LOCALISATIONVIEWER_H

#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>
#include <Eigen/Core>
#include <opencv2/core.hpp>
#include "FieldMap.h"
#include "FisheyeLens.h"
#include "Pose.hpp"

/// @brief A single YOLO detection for display (raw unit-ray corners).
struct DetectionView
{
    std::string name;                        ///< YOLO class name
    double confidence = 0.0;
    Eigen::Matrix<double, 3, 4> corners;     ///< Unit rays in {c}: TL, TR, BR, BL
    bool used = false;                        ///< True if the estimator uses this class as a landmark
};

/// @brief One accepted detection-to-landmark association.
struct AssociationView
{
    Eigen::Vector3d measRay;     ///< Measurement unit ray in {c} (box centre / post base)
    Eigen::Vector3d landmark;    ///< Associated map landmark position in {f}
    int type = 0;                ///< LandmarkType (for colour)
};

/// @brief One out-of-field corner feature (side disambiguation) for the camera panel.
struct OofFeatureView
{
    Eigen::Vector2d px = Eigen::Vector2d::Zero();   ///< Pixel position (full lens resolution)
    int status = 0;   ///< 0: on-carpet (rejected), 1: out-of-field, 2: associated to a map landmark
};

/// @brief One pose hypothesis for the top-down panel.
struct HypothesisView
{
    Eigen::Vector2d pos = Eigen::Vector2d::Zero();  ///< (x, y) in {f}
    double yaw = 0.0;                                ///< Heading in {f}
    Eigen::Matrix2d cov = Eigen::Matrix2d::Identity();///< Position covariance
    double weight = 1.0;                             ///< Normalised mixture weight
};

/// @brief Everything needed to render both panels for one processed frame.
struct ViewerFrame
{
    int videoFrame = -1;                     ///< Index into the video, or -1 if unmatched
    double t = 0.0;                          ///< Time since log start [s]

    Pose<double> Tfc;                        ///< Estimated camera pose in {f} (for re-projection)
    Eigen::Vector2d estPos = Eigen::Vector2d::Zero();
    double estYaw = 0.0;
    Eigen::Matrix2d estCov = Eigen::Matrix2d::Identity();

    std::vector<HypothesisView> hypotheses;
    std::vector<DetectionView> detections;
    std::vector<AssociationView> associations;
    std::vector<OofFeatureView> oofFeatures;   ///< Out-of-field corner features (may be empty)
    Eigen::Matrix<double, 3, Eigen::Dynamic> lineRays;  ///< Field-line rays in {c} (may be empty)

    bool hasBaseline = false;
    Eigen::Vector2d basePos = Eigen::Vector2d::Zero();
    double baseYaw = 0.0;
    double baseCost = 0.0;

    std::size_t nAssoc = 0;
    std::size_t nCand = 0;
    double updateMs = 0.0;
    bool hasError = false;
    double errXY = 0.0;
    double errYaw = 0.0;
};

/**
 * @brief Renders and replays ViewerFrames as a two-panel interactive window.
 */
class LocalisationViewer
{
public:
    /**
     * @brief Construct the viewer.
     * @param map       Field map (drawn on the top-down panel and re-projected onto the camera)
     * @param lens      Fisheye lens model for ray<->pixel projection
     * @param videoPath Path to the source video (seeked per frame)
     */
    LocalisationViewer(const FieldMap & map, const FisheyeLens & lens, const std::filesystem::path & videoPath);

    /**
     * @brief Replay the recorded frames interactively.
     * @param frames Recorded per-frame view data
     * @param mode   1 = auto-play (space toggles pause), 2 = step (advance on key)
     * @param snapshotDir Optional directory for 's' to save the current composite (may be empty)
     */
    void run(const std::vector<ViewerFrame> & frames, int mode, const std::filesystem::path & snapshotDir = {});

    /**
     * @brief Render every recorded frame to an mp4 video (no display required).
     * @param frames  Recorded per-frame view data
     * @param outPath Output video path (e.g. out/localisation.mp4)
     * @param fps     Output frame rate (defaults to the true capture rate)
     */
    void exportVideo(const std::vector<ViewerFrame> & frames, const std::filesystem::path & outPath, double fps = 10.0) const;

    /// @brief Render a single composite (camera | top-down) image without a window.
    cv::Mat renderComposite(const std::vector<ViewerFrame> & frames, std::size_t idx, const cv::Mat & rawFrame) const;

private:
    cv::Mat renderCameraPanel(const ViewerFrame & f, const cv::Mat & rawFrame, int panelH) const;
    cv::Mat renderTopDownPanel(const std::vector<ViewerFrame> & frames, std::size_t idx, int panelH) const;

    const FieldMap & map_;
    FisheyeLens lens_;
    std::filesystem::path videoPath_;
};

#endif
