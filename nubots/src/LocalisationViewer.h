/**
 * @file LocalisationViewer.h
 * @brief Two-panel interactive visualiser for the field-localisation pipeline.
 *
 * Left panel: the source camera frame with YOLO detections, field-line points,
 * associated map landmarks and the predicted horizon re-projected onto the
 * fisheye image (NUbots-exact projection). Data association is the thing this
 * panel exists to debug, so both landmark streams are colour-keyed by what the
 * estimator did with them: every YOLO detection reads as associated, gated out,
 * below confidence or not-a-landmark-class, and every out-of-field corner and
 * map landmark reads as in-FOV, associated, missed or rejected. The key itself
 * is drawn bottom-left ('k' hides it). Right panel (toggled with '3'):
 * either a top-down field view with the estimated pose, every live hypothesis
 * and its weight, the posterior covariance, the NUbots baseline, the mocap
 * ground truth and the on-ground landmark residuals; or an orbitable 3D view
 * of the same scene with 3-sigma ellipsoids on everything that carries a
 * covariance (pose estimate, hypotheses, out-of-field landmarks) -- the place
 * to inspect the height axis, which the top-down view hides.
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

/// @brief What the landmark measurement did with a YOLO detection.
enum class DetectionStatus
{
    UNUSED_CLASS = 0,   ///< Not a mapped landmark class (ball, robot): never a measurement
    LOW_CONFIDENCE,     ///< Mapped class, but below the measurement confidence threshold
    UNASSOCIATED,       ///< Usable detection that no map landmark claimed: outside the association
                        ///< gate, or beaten to its landmark by a closer detection
    ASSOCIATED          ///< Associated to a map landmark and used in the update
};

/// @brief A single YOLO detection for display (raw unit-ray corners).
struct DetectionView
{
    std::string name;                        ///< YOLO class name
    double confidence = 0.0;
    Eigen::Matrix<double, 3, 4> corners;     ///< Unit rays in {c}: TL, TR, BR, BL
    DetectionStatus status = DetectionStatus::UNUSED_CLASS;
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
    int status = 0;   ///< SideDisambiguator::FeatureStatus
};

/// @brief One out-of-field map landmark projected into the camera panel.
struct OofLandmarkProjView
{
    Eigen::Vector2d px = Eigen::Vector2d::Zero();       ///< Predicted pixel position at the estimate
    Eigen::Vector2d matchPx = Eigen::Vector2d::Zero();  ///< Matched corner's pixel (associated only)
    int status = 0;   ///< SideDisambiguator::LandmarkStatus
    bool far = false; ///< Bearing-only landmark
};

/// @brief One pose hypothesis for the top-down panel.
struct HypothesisView
{
    Eigen::Vector2d pos = Eigen::Vector2d::Zero();  ///< (x, y) in {f}
    double yaw = 0.0;                                ///< Heading in {f}
    Eigen::Matrix2d cov = Eigen::Matrix2d::Identity();///< Position covariance (xy)
    Eigen::Vector3d pos3 = Eigen::Vector3d::Zero(); ///< Full position in {f} (3D panel)
    Eigen::Matrix3d cov3 = Eigen::Matrix3d::Identity();///< Full position covariance (3D panel)
    double weight = 1.0;                             ///< Normalised mixture weight
};

/// @brief One out-of-field map landmark for the 3D panel.
struct OofLandmarkView
{
    Eigen::Vector3d pos = Eigen::Vector3d::Zero();  ///< Position in {f}
    Eigen::Matrix3d cov = Eigen::Matrix3d::Identity();///< Position covariance in {f}
    bool far = false;                                ///< Bearing-only: the position is an assumed range
                                                     ///< along the measured bearing, not a triangulation
    int status = 0;                                  ///< SideDisambiguator::LandmarkStatus this frame
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
    Eigen::Vector3d estPos3 = Eigen::Vector3d::Zero();   ///< Full torso position estimate (3D panel)
    Eigen::Matrix3d estCov3 = Eigen::Matrix3d::Identity();///< Full position covariance (3D panel)

    std::vector<HypothesisView> hypotheses;
    std::vector<DetectionView> detections;
    std::vector<AssociationView> associations;
    std::vector<OofFeatureView> oofFeatures;   ///< Out-of-field corner features (may be empty)
    std::vector<OofLandmarkProjView> oofLandmarkProj;  ///< Out-of-field landmarks projected into the camera panel
    std::vector<OofLandmarkView> oofLandmarks; ///< Out-of-field map landmarks (3D panel; may be empty)
    Eigen::Matrix<double, 3, Eigen::Dynamic> lineRays;  ///< Field-line rays in {c} (may be empty)

    bool hasBaseline = false;
    Eigen::Vector2d basePos = Eigen::Vector2d::Zero();
    double baseYaw = 0.0;
    double baseCost = 0.0;

    std::size_t nAssoc = 0;
    std::size_t nCand = 0;
    double updateMs = 0.0;
    bool hasSide = false;               ///< Side-disambiguator ran on this frame
    double sideLlr = 0.0;               ///< Accumulated own-vs-mirror evidence [nats]
    std::size_t nOofLandmarks = 0;      ///< Out-of-field map size
    bool sideFrozen = false;            ///< Map building frozen (pose/side in doubt)
    double oofVisibleMargin = 0.0;      ///< Border inside which a landmark counts as predicted-visible [px]
    bool hasError = false;
    double errXY = 0.0;
    double errYaw = 0.0;

    // Motion-capture ground truth (evaluation only; never seen by the estimator).
    bool hasTruth = false;
    Eigen::Vector3d truthPos = Eigen::Vector3d::Zero(); ///< Marker-body position in {f} (z is the
                                                        ///< marker height, ~6 cm above the torso origin)
    double truthYaw = 0.0;                              ///< Torso yaw in {f} (extrinsics-corrected)
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
     * @param fps     Output frame rate; <= 0 derives the true capture rate from the frame times
     */
    void exportVideo(const std::vector<ViewerFrame> & frames, const std::filesystem::path & outPath, double fps = 0.0) const;

    /// @brief Render a single composite (camera | top-down or 3D) image without a window.
    cv::Mat renderComposite(const std::vector<ViewerFrame> & frames, std::size_t idx, const cv::Mat & rawFrame) const;

    /// @brief Select the right-hand pane: false = top-down (default), true = 3D map.
    void setRightPane3D(bool on) { show3D_ = on; }

    /// @brief Show the per-panel colour keys ('k' in the interactive loop).
    void setShowLegend(bool on) { showLegend_ = on; }

private:
    cv::Mat renderCameraPanel(const ViewerFrame & f, const cv::Mat & rawFrame, int panelH) const;
    cv::Mat renderTopDownPanel(const std::vector<ViewerFrame> & frames, std::size_t idx, int panelH) const;
    cv::Mat render3DPanel(const std::vector<ViewerFrame> & frames, std::size_t idx, int panelH) const;

    const FieldMap & map_;
    FisheyeLens lens_;
    std::filesystem::path videoPath_;

    // Right-pane mode and 3D orbit state (adjusted from the interactive loop /
    // mouse callback while rendering stays logically const).
    bool show3D_ = false;
    bool showLegend_ = true;                ///< Draw the per-panel colour keys
    double orbitAz_ = -135.0*M_PI/180.0;    ///< Azimuth about field z [rad]
    double orbitEl_ = 35.0*M_PI/180.0;      ///< Elevation above the ground plane [rad]
    double orbitDist_ = 11.0;               ///< Camera distance from the target [m]
};

#endif
