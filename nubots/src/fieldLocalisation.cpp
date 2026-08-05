#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <limits>
#include <print>
#include <stdexcept>
#include <string>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include "FallDetector.h"
#include "FieldMap.h"
#include "fieldLocalisation.h"
#include "CameraLens.h"
#include "GaussianInfo.hpp"
#include "LocalisationViewer.h"
#include "MeasurementBodyRates.h"
#include "MeasurementFieldLandmarks.h"
#include "MeasurementFieldLines.h"
#include "MeasurementGravity.h"
#include "MeasurementKinematicHeight.h"
#include "MeasurementQuaternionNorm.h"
#include "SideDisambiguator.h"
#include "Pose.hpp"
#include "rotation.hpp"
#include "SensorLog.h"
#include "SystemLocalisation.h"

// Wrap angle to [-pi, pi]
static double wrapAngle(double a)
{
    return std::atan2(std::sin(a), std::cos(a));
}

// Choose the camera calibration for a recording.
//
// An explicit name always wins (and is fatal if unknown -- silently replaying
// through the wrong lens produces a plausible-looking but wrong trajectory,
// which is far worse than stopping). Otherwise the recorded frame size decides:
// the fisheye cameras record 1280x1024 and the simulated one 640x480, so the
// video identifies which family of calibration applies without any per-dataset
// configuration to keep in sync.
static CameraLens selectLens(const std::filesystem::path & videoPath, const std::string & lensName)
{
    if (!lensName.empty())
    {
        if (const CameraLens * named = lensByName(lensName)) return *named;
        throw std::runtime_error(std::format("Unknown --lens '{}'; expected one of: {}", lensName, lensNames()));
    }

    cv::VideoCapture cap(videoPath.string());
    if (cap.isOpened())
    {
        const double w = cap.get(cv::CAP_PROP_FRAME_WIDTH);
        const double h = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        if (const CameraLens * detected = lensForResolution(w, h)) return *detected;
        std::println("WARNING: no calibration for {}x{} frames; falling back to {}. Pass --lens=<{}>.",
                     static_cast<int>(w), static_cast<int>(h), CameraLens{}.name, lensNames());
    }
    else
    {
        std::println("WARNING: could not open {} to detect the camera; falling back to {}. Pass --lens=<{}>.",
                     videoPath.string(), CameraLens{}.name, lensNames());
    }
    return CameraLens{};
}

// Choose the field the recording was made on.
//
// The lab field and the webots field are different sizes (6.8 x 5 m against
// 9 x 6 m), so this is not a detail: replaying against the wrong one puts every
// landmark metres from where the robot sees it, and the estimate absorbs the
// difference as pose error. An explicit --field always wins; otherwise it
// follows the camera, since a recording made through the simulated camera was
// made in the simulated world.
static FieldDimensions selectField(const CameraLens & lens, const std::string & fieldName)
{
    if (!fieldName.empty())
    {
        if (const std::optional<FieldDimensions> named = fieldByName(fieldName)) return *named;
        throw std::runtime_error(std::format("Unknown --field '{}'; expected one of: {}", fieldName, fieldNames()));
    }
    return lens.name == "webots" ? webotsFieldDimensions() : FieldDimensions{};
}

// Nearest sample index in a time-ordered vector, by member time extractor
template <typename T, typename F>
static std::size_t nearestIndex(const std::vector<T> & v, double t, F && timeOf)
{
    assert(!v.empty());
    auto it = std::lower_bound(v.begin(), v.end(), t, [&](const T & s, double time) { return timeOf(s) < time; });
    if (it == v.begin()) return 0;
    if (it == v.end()) return v.size() - 1;
    std::size_t i = static_cast<std::size_t>(it - v.begin());
    return (timeOf(v[i]) - t < t - timeOf(v[i-1])) ? i : i - 1;
}

// ---------------------------------------------------------------------------
// Motion-capture ground truth (OptiTrack, data2 recordings).
//
// The mocap stream is the EVALUATION REFERENCE ONLY: it is never given to the
// estimator, which must localise from the robot's own sensors alone.
//
// Frame: the capture volume was calibrated z-up with its origin at the field
// centre and the ground-plane axes along the field lines, rotated -90 deg
// relative to the field frame: field x = mocap y, field y = -mocap x. This was
// verified against the data (candidate rotations scored against the trusted
// NUbots baseline: the -90 deg mapping fits at 0.39 m RMSE; the other three at
// 2.3-3.2 m).
//
// Rigid-body extrinsics: the Motive rigid-body axes are defined arbitrarily at
// creation, so the marker-body yaw leads the torso yaw by a constant. The
// constant below is a one-time calibration: the circular-mean offset against
// the trusted NUbots baseline over 1988 samples (std 2.2 deg; sanity-checked
// against the walk direction). Its residual error is bounded by the baseline's
// own systematic yaw bias (~1-2 deg).
//
// No positional lever-arm correction is applied: the rigid-body origin is the
// marker centroid, which sits on the torso (checked from the streamed marker
// positions), so truth position is torso position to within a few cm -- except
// in z, where the markers sit ~6 cm above the torso origin. Truth z is reported
// as the raw marker height for exactly that reason: "correcting" it would hide
// genuine height-axis errors behind a fitted constant.
// ---------------------------------------------------------------------------
namespace mocaptruth
{
    /// Ground-plane rotation {m} -> {f}: field x = mocap y, field y = -mocap x.
    const Eigen::Matrix3d Rfm = (Eigen::Matrix3d() << 0, 1, 0,
                                                     -1, 0, 0,
                                                      0, 0, 1).finished();
    /// Marker-body yaw minus torso yaw [rad] (Motive rigid-body definition).
    constexpr double yawOffset = 59.07*M_PI/180.0;
}

/// @brief Where a run's ground truth came from. The two are not interchangeable.
enum class TruthSource
{
    NONE,
    MOCAP,      ///< OptiTrack markers (data, data2): rBFf is the MARKER body, ~6 cm above the torso
    SIMULATOR   ///< Webots supervisor (data4_webots): rBFf is the torso origin itself
};

/// @brief Human-readable name of a truth source.
static const char * to_string(TruthSource src)
{
    switch (src)
    {
        case TruthSource::MOCAP:     return "MOCAP";
        case TruthSource::SIMULATOR: return "SIMULATOR";
        case TruthSource::NONE:      break;
    }
    return "none";
}

/// @brief One ground-truth pose in the field frame.
struct TruthSample
{
    double t;               ///< Time since log start [s]
    Eigen::Vector3d rBFf;   ///< Reference-body position in {f}; see TruthSource for what z means
    double yaw;             ///< Torso yaw in {f}
};

/// @brief Convert the raw mocap stream to field-frame ground truth.
static std::vector<TruthSample> buildTruth(const std::vector<MocapSample> & mocap, double t0)
{
    std::vector<TruthSample> truth;
    truth.reserve(mocap.size());
    for (const MocapSample & m : mocap)
    {
        if (!m.valid || !std::isfinite(m.t))
        {
            continue;
        }
        TruthSample s;
        s.t = m.t - t0;
        s.rBFf = mocaptruth::Rfm*m.position;
        const Eigen::Vector3d bx = mocaptruth::Rfm*m.R.col(0);   // marker-body x axis in {f}
        s.yaw = std::atan2(bx.y(), bx.x()) - mocaptruth::yawOffset;
        truth.push_back(s);
    }
    return truth;
}

/**
 * @brief Convert the simulator's RobotPoseGroundTruth stream to field-frame truth.
 *
 * Far less work than the mocap path, and deliberately so: Hft is already the
 * torso pose in the field frame the estimator works in (verified against this
 * log's own NUbots baseline, which tracks it to ~0.2 m over a whole run), so
 * there is no capture-volume rotation to undo, no marker body offset above the
 * torso, and no yaw extrinsic. Anything more here would be inventing a
 * correction the simulator has already applied.
 */
static std::vector<TruthSample> buildTruth(const std::vector<GroundTruthSample> & gt, double t0)
{
    std::vector<TruthSample> truth;
    truth.reserve(gt.size());
    for (const GroundTruthSample & g : gt)
    {
        TruthSample s;
        s.t = g.t - t0;
        s.rBFf = g.Hft.translationVector;
        s.yaw = std::atan2(g.Hft.rotationMatrix(1, 0), g.Hft.rotationMatrix(0, 0));
        truth.push_back(s);
    }
    return truth;
}

// Colours (BGR) shared between the plotted marks and the legend so the two
// can never drift apart.
namespace plotcolour
{
    const cv::Scalar field(255, 255, 255);      ///< Field lines
    const cv::Scalar goalPost(0, 220, 220);     ///< Goal posts (yellow)
    const cv::Scalar baseline(180, 180, 180);   ///< NUbots baseline (grey)
    const cv::Scalar truth(80, 220, 80);        ///< Mocap ground truth (green)
    const cv::Scalar estimateStart(255, 128, 0);///< Our estimate at the start of the run (blue)
    const cv::Scalar estimateEnd(0, 128, 255);  ///< Our estimate at the end of the run (orange)
    const cv::Scalar covariance(255, 0, 200);   ///< 2-sigma position ellipse (magenta)

    // Interpolate the estimate trajectory colour by time fraction [0, 1].
    inline cv::Scalar estimate(double frac)
    {
        return cv::Scalar(255*(1 - frac), 128, 255*frac);
    }
}

// Top-down field renderer for the exported trajectory figure.
//
// Rendering niceties (this is the headline figure of a run):
//  - two-tone mowing stripes and a darker border strip so the carpet reads as a
//    pitch rather than a flat green rectangle;
//  - trajectories drawn as connected anti-aliased polylines (not dot spray);
//  - covariance ellipses accumulated on an overlay and alpha-blended so dozens
//    of rings shade the corridor instead of scribbling over the trails;
//  - a title band tall enough for the title and the RMSE subtitle;
//  - goal structure (posts + net box) drawn behind each goal line.
class FieldPlot
{
public:
    FieldPlot(const FieldDimensions & dims, double pixelsPerMetre = 100.0)
        : dims_(dims)
        , scale_(pixelsPerMetre)
    {
        const double borderm = dims.borderStripMinWidth + 0.35;
        fieldW_ = static_cast<int>((dims.fieldLength + 2*borderm)*scale_);
        fieldH_ = static_cast<int>((dims.fieldWidth  + 2*borderm)*scale_);
        width_  = fieldW_;
        height_ = topMargin_ + fieldH_ + bottomMargin_;

        img_ = cv::Mat(height_, width_, CV_8UC3, cv::Scalar(34, 32, 30));
        drawField();
    }

    cv::Point2i toPixel(double xf, double yf) const
    {
        // Field x to the right, field y up in the image (offset below the title band)
        return {static_cast<int>(std::lround(fieldW_/2.0 + xf*scale_)),
                static_cast<int>(std::lround(topMargin_ + fieldH_/2.0 - yf*scale_))};
    }

    void drawField()
    {
        const int lw = std::max(2, static_cast<int>(dims_.lineWidth*scale_));
        const double hl = dims_.fieldLength/2, hw = dims_.fieldWidth/2;
        const double hbx = fieldW_/(2.0*scale_), hby = fieldH_/(2.0*scale_);

        // Border strip (darker) under the playing field (striped)
        cv::rectangle(img_, toPixel(-hbx, hby), toPixel(hbx, -hby), cv::Scalar(24, 78, 24), cv::FILLED);
        for (int i = 0; ; ++i)
        {
            const double x0 = -hl + i*1.0;
            if (x0 >= hl) break;
            const double x1 = std::min(x0 + 1.0, hl);
            const cv::Scalar green = (i % 2 == 0) ? cv::Scalar(33, 105, 33) : cv::Scalar(29, 96, 29);
            cv::rectangle(img_, toPixel(x0, hw), toPixel(x1, -hw), green, cv::FILLED);
        }

        const cv::Scalar & white = plotcolour::field;
        auto rect = [&](double x0, double y0, double x1, double y1, const cv::Scalar & col)
        {
            cv::rectangle(img_, toPixel(x0, y0), toPixel(x1, y1), col, lw, cv::LINE_AA);
        };
        rect(-hl, -hw, hl, hw, white);                                              // Boundary
        cv::line(img_, toPixel(0, -hw), toPixel(0, hw), white, lw, cv::LINE_AA);    // Halfway line
        cv::circle(img_, toPixel(0, 0), static_cast<int>(dims_.centreCircleDiameter/2*scale_), white, lw, cv::LINE_AA);
        cv::circle(img_, toPixel(0, 0), lw, white, cv::FILLED, cv::LINE_AA);        // Centre mark
        for (int s : {-1, 1})
        {
            rect(s*hl, -dims_.goalAreaWidth/2, s*(hl - dims_.goalAreaLength), dims_.goalAreaWidth/2, white);
            rect(s*hl, -dims_.penaltyAreaWidth/2, s*(hl - dims_.penaltyAreaLength), dims_.penaltyAreaWidth/2, white);
            cv::drawMarker(img_, toPixel(s*(hl - dims_.penaltyMarkDistance), 0), white, cv::MARKER_CROSS, 10, lw, cv::LINE_AA);

            // Goal: net box behind the goal line, posts on it
            rect(s*hl, -dims_.goalWidth/2, s*(hl + dims_.goalDepth), dims_.goalWidth/2, cv::Scalar(150, 150, 150));
            for (int t : {-1, 1})
            {
                const cv::Point2i p = toPixel(s*hl, t*dims_.goalWidth/2);
                const int r = std::max(3, static_cast<int>(dims_.goalpostWidth*0.75*scale_));
                cv::circle(img_, p, r, plotcolour::goalPost, cv::FILLED, cv::LINE_AA);
                cv::circle(img_, p, r, cv::Scalar(30, 30, 30), 1, cv::LINE_AA);
            }
        }
    }

    /// @brief Draw the title and optional subtitle in the top band (two clear lines).
    void drawTitle(const std::string & title, const std::string & subtitle = "")
    {
        cv::putText(img_, title, cv::Point(16, 30), cv::FONT_HERSHEY_SIMPLEX, 0.72,
                    cv::Scalar(245, 245, 245), 2, cv::LINE_AA);
        if (!subtitle.empty())
        {
            // Shrink to fit: RMSE summaries vary in length and must not clip.
            double scale = 0.46;
            while (scale > 0.30
                   && cv::getTextSize(subtitle, cv::FONT_HERSHEY_SIMPLEX, scale, 1, nullptr).width > width_ - 32)
            {
                scale -= 0.02;
            }
            cv::putText(img_, subtitle, cv::Point(16, topMargin_ - 12), cv::FONT_HERSHEY_SIMPLEX, scale,
                        cv::Scalar(185, 185, 185), 1, cv::LINE_AA);
        }
    }

    /// @brief Draw the legend explaining every plotted mark in the bottom band.
    void drawLegend()
    {
        const int y0 = topMargin_ + fieldH_ + 14;
        const int rowH = 34;
        const int col1 = 18;
        const int col2 = static_cast<int>(fieldW_*0.47);
        const int col3 = static_cast<int>(fieldW_*0.80);
        const int textDx = 34;

        auto label = [&](int x, int row, const std::string & text)
        {
            cv::putText(img_, text, cv::Point(x + textDx, y0 + row*rowH + 5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.42, cv::Scalar(225, 225, 225), 1, cv::LINE_AA);
        };
        auto lineSwatch = [&](int x, int row, const cv::Scalar & colour, int thick)
        {
            cv::line(img_, cv::Point(x, y0 + row*rowH), cv::Point(x + 24, y0 + row*rowH),
                     colour, thick, cv::LINE_AA);
        };

        // Row 1: our estimate (gradient swatch), baseline, goal posts
        for (int dx = 0; dx <= 24; ++dx)
        {
            cv::line(img_, cv::Point(col1 + dx, y0 + rowH - 3), cv::Point(col1 + dx, y0 + rowH + 3),
                     plotcolour::estimate(dx/24.0), 1);
        }
        label(col1, 1, "Our estimate (blue start -> orange end)");
        lineSwatch(col2, 1, plotcolour::baseline, 2);
        label(col2, 1, "NUbots baseline (localisation.Field)");
        cv::circle(img_, cv::Point(col3 + 12, y0 + rowH), 5, plotcolour::goalPost, cv::FILLED, cv::LINE_AA);
        cv::circle(img_, cv::Point(col3 + 12, y0 + rowH), 5, cv::Scalar(30, 30, 30), 1, cv::LINE_AA);
        label(col3, 1, "Goal posts");

        // Row 2: mocap truth, 2-sigma ellipse, field lines
        lineSwatch(col1, 2, plotcolour::truth, 2);
        label(col1, 2, "Mocap ground truth (held-out reference)");
        cv::ellipse(img_, cv::Point(col2 + 12, y0 + 2*rowH), cv::Size(11, 7), 0, 0, 360,
                    plotcolour::covariance, 1, cv::LINE_AA);
        label(col2, 2, "2-sigma position uncertainty");
        lineSwatch(col3, 2, plotcolour::field, 2);
        label(col3, 2, "Field lines (map)");
    }

    void drawPoint(double xf, double yf, const cv::Scalar & colour, int radius = 2)
    {
        cv::circle(img_, toPixel(xf, yf), radius, colour, -1, cv::LINE_AA);
    }

    void drawSegment(double x0, double y0, double x1, double y1, const cv::Scalar & colour, int thick = 2)
    {
        cv::line(img_, toPixel(x0, y0), toPixel(x1, y1), colour, thick, cv::LINE_AA);
    }

    /// @brief Ring marker (trajectory start/end emphasis).
    void drawRing(double xf, double yf, const cv::Scalar & colour, int radius, int thick = 2)
    {
        cv::circle(img_, toPixel(xf, yf), radius, colour, thick, cv::LINE_AA);
    }

    /// @brief Accumulate a covariance ellipse on the translucent overlay (see blendOverlay).
    void drawCovarianceEllipse(double xf, double yf, const Eigen::Matrix2d & P, const cv::Scalar & colour, double nSigma = 2.0)
    {
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> es(P);
        if (es.info() != Eigen::Success) return;
        const Eigen::Vector2d val = es.eigenvalues().cwiseMax(0.0);
        const Eigen::Vector2d v = es.eigenvectors().col(1);         // Major axis
        double angleDeg = std::atan2(-v.y(), v.x())*180.0/M_PI;     // Image y is flipped
        cv::Size axes(std::max(1, static_cast<int>(nSigma*std::sqrt(val(1))*scale_)),
                      std::max(1, static_cast<int>(nSigma*std::sqrt(val(0))*scale_)));
        if (overlay_.empty()) overlay_ = img_.clone();
        cv::ellipse(overlay_, toPixel(xf, yf), axes, angleDeg, 0, 360, colour, 1, cv::LINE_AA);
    }

    /// @brief Blend the accumulated ellipse overlay into the figure.
    void blendOverlay(double alpha = 0.45)
    {
        if (overlay_.empty()) return;
        cv::addWeighted(img_, 1.0 - alpha, overlay_, alpha, 0.0, img_);
        overlay_.release();
    }

    const cv::Mat & image() const { return img_; }

private:
    FieldDimensions dims_;
    double scale_;
    int width_, height_;
    int fieldW_ = 0, fieldH_ = 0;
    const int topMargin_ = 66;      ///< Title band height [px] (title + subtitle lines)
    const int bottomMargin_ = 106;  ///< Legend band height [px]
    cv::Mat img_;
    cv::Mat overlay_;               ///< Covariance ellipses, blended once at the end
};

// Baseline-free initial pose from the first usable vision frame. The NUbots
// baseline is never read by the estimator; it solves the field pose purely from
// the YOLO landmark rays and the field map.
//
// Roll, pitch and torso height are taken from the kinematic chain (the odometry
// world frame is gravity-aligned, so its z axis coincides with field-up); only
// (x, y, yaw) are unknown and are found by a coarse global grid search that
// maximises the landmark measurement log-likelihood (association + robust
// residual reuse MeasurementFieldLandmarks, so scoring matches the filter's own
// model).
//
// On-field landmarks leave a 180 deg (own-half vs opponent-half) ambiguity that
// they fundamentally cannot resolve. It is broken with external game context:
// the robot always starts in its own half, so ownHalfXSign (the sign of field-x
// for the starting half, from the GameController team side in deployment) selects
// between the global maximum and its mirror.
//
// This is start-of-game initialisation only, and must stay that way. The
// ownHalfXSign prior is true exactly once -- at kick-off -- so re-running this
// mid-game (say, to recover from a fall) would force a correctly localised robot
// standing in the opponent half back across the halfway line. Recovery from a
// fall is handled instead by widening the belief around the pose already held.
static bool solveInitialPose(const SensorLog & log, const FieldMap & map,
                             double t0, double ownHalfXSign,
                             const FallDetector & falls,
                             Eigen::VectorXd & eta0Out, double & tInitOut, std::size_t & visIdxOut)
{
    const FieldDimensions & dims = map.dims;
    const double halfL = dims.fieldLength/2 + dims.borderStripMinWidth;
    const double halfW = dims.fieldWidth/2  + dims.borderStripMinWidth;
    const double dxy   = 0.35;                 // grid step, position [m]
    const double dyaw  = 18.0*M_PI/180.0;      // grid step, heading [rad]
    const std::size_t minAssoc = 4;            // constraints needed to trust a solve

    for (std::size_t vi = 0; vi < log.vision.size(); ++vi)
    {
        const VisionSample & v = log.vision[vi];
        if (v.detections.empty()) continue;
        if (!v.Hcw.rotationMatrix.allFinite() || !v.Hcw.translationVector.allFinite()) continue;
        // roll, pitch and torso height below are taken from the kinematic chain and
        // held fixed while (x, y, yaw) are searched, so a frame captured mid-topple
        // would anchor the whole solve to an attitude that is about to be wrong.
        if (falls.at(v.t - t0) != Posture::UPRIGHT) continue;

        std::size_t k = nearestIndex(log.sensors, v.t, [](const SensorsSample & s) { return s.t; });
        if (std::abs(log.sensors[k].t - v.t) > 0.1) continue;

        // Camera pose w.r.t. torso, and gravity-aligned torso attitude/height.
        // Camera pose w.r.t. torso from the kinematic chain (odometry cancels):
        // Tbc = Htw * Hcw^{-1}
        Pose<double> Tbc = log.sensors[k].Htw*v.Hcw.inverse();
        Pose<double> Twt = log.sensors[k].Htw.inverse();
        const Eigen::Vector3d rpyTorso = rot2rpy(Twt.rotationMatrix);
        const double roll0 = rpyTorso.x();
        const double pitch0 = rpyTorso.y();
        const double z0 = Twt.translationVector.z();
        const double tInit = v.t - t0;

        // Probe system used only to drive association/likelihood scoring.
        const Eigen::MatrixXd Stmp = Eigen::MatrixXd::Identity(SystemLocalisation::nx, SystemLocalisation::nx)*0.01;
        SystemLocalisation probe(GaussianInfo<double>::fromSqrtMoment(
                                     Eigen::VectorXd::Zero(SystemLocalisation::nx), Stmp));

        double bestScore = -std::numeric_limits<double>::infinity();
        std::size_t bestAssoc = 0;
        Eigen::VectorXd bestEta = Eigen::VectorXd::Zero(SystemLocalisation::nx);

        for (double x = -halfL; x <= halfL; x += dxy)
        {
            for (double y = -halfW; y <= halfW; y += dxy)
            {
                for (double yaw = -M_PI; yaw < M_PI; yaw += dyaw)
                {
                    Eigen::VectorXd cand(SystemLocalisation::nx);
                    cand.setZero();
                    cand.head<3>() << x, y, z0;
                    cand.segment<4>(SystemLocalisation::iQuat) =
                        rpy2quat(Eigen::Vector3d(roll0, pitch0, yaw));
                    probe.resetTo(GaussianInfo<double>::fromSqrtMoment(cand, Stmp), tInit);
                    MeasurementFieldLandmarks meas(tInit, v, Tbc, map, probe);
                    if (meas.numAssociated() < minAssoc) continue;
                    // Robust log-likelihood already rewards inliers and floors
                    // outliers at the clutter density, so it favours the pose
                    // that explains the most rays well.
                    const double score = meas.logLikelihood(cand, probe);
                    if (score > bestScore)
                    {
                        bestScore = score;
                        bestAssoc = meas.numAssociated();
                        bestEta = cand;
                    }
                }
            }
        }

        if (bestAssoc < minAssoc) continue;   // not enough to localise on this frame; try the next

        // Resolve the own-half/opponent-half symmetry from game context: force the
        // solution onto the starting half. The mirror has identical likelihood, so
        // this loses no fit; it only picks the physically valid side.
        Eigen::VectorXd mirrorEta = SystemLocalisation::mirrorState(bestEta);
        if (ownHalfXSign != 0.0 && bestEta(0)*ownHalfXSign < 0.0)
        {
            std::swap(bestEta, mirrorEta);
        }

        // DIAGNOSTIC: confirm the two halves are indistinguishable from on-field
        // landmarks (near-equal log-likelihood) — the side comes from the prior.
        probe.resetTo(GaussianInfo<double>::fromSqrtMoment(mirrorEta, Stmp), tInit);
        MeasurementFieldLandmarks mmir(tInit, v, Tbc, map, probe);
        double mirrorScore = mmir.numAssociated() > 0 ? mmir.logLikelihood(mirrorEta, probe)
                                                       : -std::numeric_limits<double>::infinity();
        std::println("  symmetry check: chosen-half logLik {:.2f} vs opponent-half logLik {:.2f} "
                     "(near-equal => side set by the start-half prior, not the landmarks)",
                     bestScore, mirrorScore);

        eta0Out = bestEta;
        tInitOut = tInit;
        visIdxOut = vi;
        return true;
    }
    return false;
}

void runFieldLocalisation(const std::filesystem::path & dataDir, int interactive, const std::filesystem::path & outputDirectory,
                          const std::string & lensName, const std::string & fieldName, int verbosity)
{
    const std::filesystem::path jsonPath = dataDir / "recorded_data.json";
    const std::filesystem::path timecodePath = dataDir / "Left_timecode.txt";
    assert(std::filesystem::exists(jsonPath));
    assert(std::filesystem::exists(timecodePath));

    std::println("Loading sensor log: {}", jsonPath.string());
    SensorLog log(jsonPath, timecodePath);
    const double t0 = log.t0;
    std::println("Loaded {} sensor, {} vision, {} walk, {} baseline samples spanning {:.1f} s",
                 log.sensors.size(), log.vision.size(), log.walk.size(), log.fieldBaseline.size(),
                 (log.sensors.empty() ? 0.0 : log.sensors.back().t - t0));

    // Odometry-derived body twist input buffer
    if (log.clockRealigned)
    {
        std::println("Clock: payload and envelope timestamps differ by {:.1f} s (webots simulation time vs "
                     "wall clock); the envelope-stamped streams have been shifted onto the payload clock",
                     log.clockOffset);
    }

    std::vector<BodyTwistSample> twists = SystemLocalisation::twistFromOdometry(log.sensors, t0);
    std::println("Derived {} body twist samples from odometry", twists.size());

    // Posture timeline. The robot's own stability flags are authoritative when the
    // log carries them; otherwise the tilt of the smoothed accelerometer stands in.
    const FallDetector falls(log.sensors, log.stability, t0);
    std::println("Posture: from {} -- {} non-upright window(s)",
                 falls.usingFlags() ? "the robot's stability flags"
                                    : "the accelerometer (log has no stability stream)",
                 falls.intervals().size());
    for (const FallDetector::Interval & iv : falls.intervals())
    {
        std::println("  not upright t={:.2f}..{:.2f} s ({:.2f} s)", iv.start, iv.end, iv.end - iv.start);
    }
    if (!falls.unrecognisedStates().empty())
    {
        // Silently mapping an unknown name to "upright" would disable every fall
        // protection in the filter, so it is surfaced rather than swallowed.
        std::print("  WARNING: unrecognised stability state(s) treated as upright:");
        for (const std::string & s : falls.unrecognisedStates())
        {
            std::print(" {}", s);
        }
        std::println("");
    }

    // Motion-capture ground truth (evaluation only; see mocaptruth above).
    // Ground truth, from whichever stream this recording carries. The simulator's
    // is preferred where both exist: it is the pose the world was actually built
    // from rather than an inference from markers. A log can carry neither, which
    // is not an error -- it only disables the comparisons.
    const TruthSource truthSource = !log.groundTruth.empty() ? TruthSource::SIMULATOR
                                  : !log.mocap.empty()      ? TruthSource::MOCAP
                                                            : TruthSource::NONE;
    const std::vector<TruthSample> truth = truthSource == TruthSource::SIMULATOR
                                         ? buildTruth(log.groundTruth, t0)
                                         : buildTruth(log.mocap, t0);
    if (!truth.empty())
    {
        std::println("Ground truth: {} {} samples spanning t={:.1f}..{:.1f} s", truth.size(),
                     truthSource == TruthSource::SIMULATOR ? "simulator (RobotPoseGroundTruth)"
                                                           : "mocap (MotionCapture)",
                     truth.front().t, truth.back().t);
    }
    else
    {
        std::println("Ground truth: none in this log (truth comparisons disabled)");
    }

    // Camera calibration, shared by the out-of-field feature pipeline and the
    // visualiser. The real robots and webots do not merely differ in their
    // numbers, they use different projection models (see CameraLens.h), so this
    // has to follow the recording rather than be fixed. --lens names one
    // explicitly; otherwise the recorded frame size picks it.
    const CameraLens lens = selectLens(dataDir / "Left.mp4", lensName);
    std::println("Camera calibration: {}", lens.describe());

    // Field landmark map. Selected after the camera because, absent an explicit
    // --field, the camera is what says which world this recording came from.
    const FieldDimensions fieldDims = selectField(lens, fieldName);
    FieldMap map(fieldDims);
    std::println("Field: {} ({:.1f} x {:.1f} m, goal width {:.2f} m, border strip {:.2f} m)",
                 fieldDims.name, fieldDims.fieldLength, fieldDims.fieldWidth,
                 fieldDims.goalWidth, fieldDims.borderStripMinWidth);

    // Out-of-field side disambiguation. On-field landmarks are invariant under
    // the field's 180 deg symmetry, but the background scenery is not: corner
    // features beyond the field carpet carry the own-half/opponent-half
    // information the landmarks lack (mid-game kidnap recovery, complementing
    // the start-half prior used at initialisation). The disambiguator maps
    // background corners online and compares how well the estimate and its
    // mirror explain them; a sustained mirror preference flips the filter.
    constexpr bool useOutOfField = true;
    SideDisambiguator sideDis(lens, map.dims);

    // Read a tuning override from the environment, for ablation sweeps.
    auto envDouble = [](const char * name, double fallback) {
        if (const char * e = std::getenv(name)) { return std::atof(e); }
        return fallback;
    };

    // Simulated kidnap for verification: KIDNAP_T=<seconds> mirrors the filter
    // state once at that time WITHOUT telling the disambiguator, emulating an
    // unnoticed symmetry flip that it must detect and correct.
    double kidnapT = std::numeric_limits<double>::quiet_NaN();
    if (const char * kidnapEnv = std::getenv("KIDNAP_T"))
    {
        kidnapT = std::atof(kidnapEnv);
        std::println("Simulated kidnap armed: state will be mirrored at t={:.1f} s", kidnapT);
    }
    bool kidnapDone = false;

    // Simulated fall for verification: FALL_T=<seconds> FALL_DURATION=<seconds>
    // forces the posture to FALLEN over that window. Neither recording contains a
    // real fall (the torso never leaves 0.43-0.44 m and tilt peaks at 34 deg), so
    // this is the only way to exercise the suppression, prediction-only coast,
    // inflation and recovery path on real data. Note what it does and does not
    // show: the robot really is upright throughout, so this demonstrates that the
    // filter survives and reconverges after a blind window, not that the posture
    // detector fires on a genuine topple.
    double fallT = std::numeric_limits<double>::quiet_NaN();
    double fallDuration = 3.0;
    if (const char * fallEnv = std::getenv("FALL_T"))
    {
        fallT = std::atof(fallEnv);
        if (const char * durEnv = std::getenv("FALL_DURATION"))
        {
            fallDuration = std::atof(durEnv);
        }
        std::println("Simulated fall armed: posture forced to fallen over t={:.1f}..{:.1f} s",
                     fallT, fallT + fallDuration);
    }
    cv::VideoCapture videoCap;
    if (useOutOfField)
    {
        videoCap.open((dataDir / "Left.mp4").string());
        if (!videoCap.isOpened())
        {
            std::println("WARNING: could not open {} - out-of-field features disabled", (dataDir / "Left.mp4").string());
        }
    }
    // Sequential fetch of a video frame by index (vision samples are time-ordered,
    // so targets are non-decreasing; a backwards seek is supported but not expected).
    int videoNextFrame = 0;
    cv::Mat videoFrameBgr;
    auto fetchVideoFrame = [&](int target) -> bool
    {
        if (!videoCap.isOpened() || target < 0) return false;
        if (target < videoNextFrame - 1)
        {
            videoCap.set(cv::CAP_PROP_POS_FRAMES, target);
            videoNextFrame = target;
        }
        if (target == videoNextFrame - 1) return !videoFrameBgr.empty();   // Same frame as last fetch
        while (videoNextFrame <= target)
        {
            if (!videoCap.read(videoFrameBgr)) return false;
            videoNextFrame++;
        }
        return !videoFrameBgr.empty();
    };

    // Baseline-free initialisation: solve the first usable vision frame's pose
    // from the landmark detections and the field map. The NUbots baseline is
    // used ONLY for later comparison/plotting, never by the estimator.
    // External game context (RoboCup): the robot always starts in its own half,
    // which breaks the field's 180-degree symmetry that on-field landmarks cannot.
    // Sign of field-x for the starting half; supply from the GameController team
    // side in deployment. For this recording the robot starts in the +x half.
    constexpr double ownHalfXSign = +1.0;

    Eigen::VectorXd eta0;
    double tInit = 0.0;
    std::size_t initVisIdx = 0;
    auto ticInit = std::chrono::steady_clock::now();
    bool solved = solveInitialPose(log, map, t0, ownHalfXSign, falls, eta0, tInit, initVisIdx);
    double initMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ticInit).count();
    std::println("Initial global grid solve took {:.1f} ms", initMs);
    if (!solved)
    {
        std::println("ERROR: could not solve an initial pose from the landmark detections");
        return;
    }
    std::println("Initialising at t={:.3f} s from first-frame landmark solve "
                 "(x={:.2f} m, y={:.2f} m, yaw={:.1f} deg)",
                 tInit, eta0(0), eta0(1), SystemLocalisation::heading(eta0)*180.0/M_PI);

    // Deliberately loose prior: the grid localises coarsely and the recursive
    // updates sharpen it over the first (stationary) seconds. z, roll and pitch
    // come from kinematics so their prior is tight; (x, y, yaw) start wide.
    // The attitude entries are quaternion-component std devs: a body rotation of
    // s maps to 0.5*s on the components (see quaternionSigma), so the 0.05 rad
    // roll/pitch and 0.5 rad yaw priors become 0.025 and 0.25. Yaw dominates, and
    // it is not separable across components, so all four carry the loose figure --
    // the tight roll/pitch prior is re-established within a frame by gravity.
    // The velocity states start at zero (the robot is stationary at kick-off) with
    // a prior wide enough to cover a walk if it is not; the gyroscope bias starts
    // at zero with a prior covering a few deg/s, which is the scale of the drift
    // it exists to absorb.
    Eigen::MatrixXd S0 = Eigen::MatrixXd::Zero(SystemLocalisation::nx, SystemLocalisation::nx);
    // GYRO_BIAS=off also pins the initial bias covariance, since the random-walk
    // PSD alone only limits drift -- the state would still be estimated down from
    // its prior. Both have to be shut off to hold it at zero.
    const bool freezeBias = [] {
        const char * e = std::getenv("GYRO_BIAS");
        return e != nullptr && std::string(e) == "off";
    }();
    // Small enough to be frozen against the ~2 deg/s under test, large enough
    // that the information form (which inverts this) stays conditioned.
    const double s0GyroBias = freezeBias ? 1e-4 : 0.05;
    S0.diagonal() << 1.0, 1.0, 0.05,             // position
                     0.25, 0.25, 0.25, 0.25,     // quaternion
                     0.30, 0.30, 0.10,           // body linear velocity [m/s]
                     0.50, 0.50, 0.50,           // body angular velocity [rad/s]
                     s0GyroBias, s0GyroBias, s0GyroBias,   // gyroscope bias [rad/s] (~3 deg/s)
                     0.02, 0.02;                 // camera mount bias
    auto p0 = GaussianInfo<double>::fromSqrtMoment(eta0, S0);

    SystemLocalisation system(p0);
    system.resetTo(p0, tInit);
    system.setVerbosity(verbosity);

    // Multi-hypothesis field-symmetry handling. When enabled the belief is a
    // Gaussian mixture seeded with the initial pose and its 180 deg mirror.
    // On-field landmarks cannot separate the two (each fits its own mirror-
    // partner landmarks equally well), so the pair sits at 50/50 until the
    // asymmetric out-of-field evidence, folded into the weights each frame
    // (SideDisambiguator -> addSideLogEvidence), collapses the mirror; a
    // mid-game kidnap re-seeds it (spawnMirror) for the weights to re-resolve.
    //
    // Default OFF: this pipeline initialises from the known start half, so the
    // side is already certain and the bank only adds an early two-component
    // window before it prunes back to the single-hypothesis path (identical
    // final accuracy on the recorded data). It is the mechanism to enable for a
    // robot that can start on an unknown side or be displaced without a
    // GameController signal.
    constexpr bool useHypothesisBank = false;
    if (useHypothesisBank)
    {
        system.initialiseHypotheses();
    }

    // Output accumulators
    struct Record
    {
        double t;
        Eigen::VectorXd mean;
        Eigen::VectorXd sigma;
        Eigen::Vector3d rpy;        ///< Attitude derived from the state quaternion [rad]
        Eigen::Vector3d attStd;     ///< Attitude std devs in the field tangent [rad]
        std::size_t nAssoc, nCand;
        double baseX, baseY, baseYaw, baseCost;
        double errXY, errYaw;
        double updateMs;
        double sideLlr;         ///< Accumulated own-vs-mirror out-of-field evidence [nats]
        std::size_t oofAssoc;   ///< Out-of-field corners associated to map landmarks
        double truthX, truthY, truthZ, truthYaw;    ///< Mocap ground truth (NaN if unmatched)
        double errXYTruth, errYawTruth;             ///< Our estimate vs truth
    };
    std::vector<Record> records;
    records.reserve(log.vision.size());

    // Per-frame data for the interactive visualiser. Captured for the live viewer
    // (interactive) and also when exporting, so --export can render the mp4.
    const bool captureFrames = interactive > 0 || !outputDirectory.empty();
    std::vector<ViewerFrame> viewFrames;
    if (captureFrames)
    {
        viewFrames.reserve(log.vision.size());
    }

    // The belief as the viewer needs it. Shared by processed frames and by the
    // frames the estimator skipped, because the replay timeline is the video:
    // playback must not jump over a fall, and during one the predicted belief is
    // precisely what is worth watching drift against the footage.
    auto captureBelief = [&](ViewerFrame & vf, const Eigen::VectorXd & mean) {
        vf.estPos = mean.head<2>();
        vf.estYaw = SystemLocalisation::heading(mean);
        const Eigen::Matrix3d Ppos = system.density.cov().topLeftCorner<3, 3>();
        vf.estCov = Ppos.topLeftCorner<2, 2>();
        vf.estPos3 = mean.head<3>();
        vf.estCov3 = Ppos;

        // Hypotheses (falls back to the single density when the bank is off).
        const std::vector<double> hw = system.hypothesisWeights();
        if (system.hypotheses().empty())
        {
            HypothesisView h;
            h.pos = vf.estPos;
            h.yaw = vf.estYaw;
            h.cov = vf.estCov;
            h.pos3 = vf.estPos3;
            h.cov3 = vf.estCov3;
            h.weight = 1.0;
            vf.hypotheses.push_back(h);
        }
        else
        {
            const auto & comps = system.hypotheses();
            for (std::size_t i = 0; i < comps.size(); ++i)
            {
                const Eigen::VectorXd m = comps[i].mean();
                HypothesisView h;
                h.pos = m.head<2>();
                h.yaw = SystemLocalisation::heading(m);
                h.pos3 = m.head<3>();
                h.cov3 = comps[i].cov().topLeftCorner<3, 3>();
                h.cov = h.cov3.topLeftCorner<2, 2>();
                h.weight = i < hw.size() ? hw[i] : 0.0;
                vf.hypotheses.push_back(h);
            }
        }
    };

    // Ground truth at a time, for the top-down trail (evaluation only).
    auto captureTruth = [&](ViewerFrame & vf, double tt) {
        if (truth.empty()) return;
        const std::size_t ti = nearestIndex(truth, tt, [](const TruthSample & s) { return s.t; });
        if (std::abs(truth[ti].t - tt) >= 0.05) return;
        vf.hasTruth = true;
        vf.truthPos = truth[ti].rBFf;
        vf.truthYaw = truth[ti].yaw;
    };

    // A frame the filter did not update on. The overlay still gets the predicted
    // camera pose where the log carries one, so the horizon and the re-projected
    // map keep being drawn -- watching them slide off the scene is the clearest
    // picture of what a blind window costs. Detections and associations are left
    // empty because there genuinely were none used.
    auto captureSkipped = [&](const VisionSample & vv, double tt, FrameSkip why) {
        if (!captureFrames) return;
        ViewerFrame vf;
        vf.videoFrame = vv.videoFrame;
        vf.t = tt;
        vf.skip = why;
        const Eigen::VectorXd mean = system.density.mean();
        captureBelief(vf, mean);
        captureTruth(vf, tt);
        if (vv.Hcw.rotationMatrix.allFinite() && vv.Hcw.translationVector.allFinite() && !log.sensors.empty())
        {
            const std::size_t sk = nearestIndex(log.sensors, vv.t, [](const SensorsSample & s) { return s.t; });
            if (std::abs(log.sensors[sk].t - vv.t) <= 0.1)
            {
                const Pose<double> Tbc = log.sensors[sk].Htw*vv.Hcw.inverse();
                vf.Tfc = SystemLocalisation::fieldPose<double>(mean)*Tbc
                         *Pose<double>(SystemLocalisation::cameraBiasRotation<double>(mean), Eigen::Vector3d::Zero());
            }
        }
        viewFrames.push_back(std::move(vf));
    };

    // The NUbots baseline is itself an estimate whose cost spikes when it is
    // lost; restrict the headline comparison to samples where it is trustworthy.
    const double trustedBaselineCost = 1.0;

    double sumSqErrXY = 0, sumSqErrYaw = 0, sumMs = 0, maxMs = 0;
    double sumSqErrXYTrusted = 0, sumSqErrYawTrusted = 0;
    std::size_t nCompared = 0, nTrusted = 0, nUpdates = 0, nSkipped = 0;

    // Ground-truth comparison accumulators: ours over every truth-matched frame,
    // plus a head-to-head over the frames where the NUbots baseline also reported
    // a (finite) pose, so both estimators are judged on identical samples.
    double sumSqTruthXY = 0, sumSqTruthYaw = 0;
    std::size_t nTruth = 0;
    double sumSqTruthXYBoth = 0, sumSqTruthYawBoth = 0;
    double sumSqBaseTruthXY = 0, sumSqBaseTruthYaw = 0;
    std::size_t nTruthBoth = 0;
    double sumZerr = 0, sumSqZerr = 0;      ///< est z - truth marker z
    double sumOofMs = 0, maxOofMs = 0;
    std::size_t nOofFrames = 0, sumOofOut = 0, sumOofTotal = 0, sumOofAssoc = 0, nSideFlips = 0;

    // DIAGNOSTIC: mean landmark reprojection residual at a given pose, using that
    // pose's own associations. Lets us ask which pose the vision data supports
    // (our estimate vs the baseline) without treating either as ground truth.
    auto meanReprojResidual = [&](const Eigen::VectorXd & eta, const VisionSample & vv,
                                  const Pose<double> & TbcArg) -> std::pair<double, std::size_t>
    {
        const Eigen::MatrixXd Stmp = Eigen::MatrixXd::Identity(SystemLocalisation::nx, SystemLocalisation::nx)*0.01;
        SystemLocalisation probe(GaussianInfo<double>::fromSqrtMoment(eta, Stmp));
        probe.resetTo(GaussianInfo<double>::fromSqrtMoment(eta, Stmp), 0.0);
        MeasurementFieldLandmarks m(0.0, vv, TbcArg, map, probe);
        const auto & U = m.measuredRays();
        if (U.cols() == 0) return {std::numeric_limits<double>::quiet_NaN(), 0};
        Eigen::Matrix<double, 3, Eigen::Dynamic> P = m.predictRays<double>(eta);
        double s = 0;
        for (Eigen::Index j = 0; j < U.cols(); ++j)
            s += std::acos(std::clamp(U.col(j).dot(P.col(j)), -1.0, 1.0));
        return {s/U.cols(), static_cast<std::size_t>(U.cols())};
    };
    double sumResidSelf = 0, sumResidBase = 0;
    std::size_t nResid = 0;

    // DIAGNOSTIC: capture one representative stationary frame to probe how sharply
    // the landmark residual rises as the pose is perturbed (i.e. the geometric
    // observability / honest uncertainty in each direction).
    VisionSample statFrame;
    Pose<double> statTbc;
    Eigen::VectorXd statMean;
    bool haveStat = false;

    double lastRespawnT = -std::numeric_limits<double>::infinity();  ///< Bank-mode mirror re-seed cooldown

    // How long the *Disturbed process noise PSDs stay in force once the robot
    // stops being upright.
    //
    // What a fall does to the pose is a bounded event, not a diffusion: the torso
    // moves while it topples and while it is levered back upright, and in between
    // it lies still on the carpet. Running the disturbed PSDs for the whole fallen
    // window models the still part as a random walk at 0.40 m/sqrt(s), so the
    // belief's width ends up reporting how long the robot lay there rather than
    // how far it could have gone -- 1.5 m of position std after 12 s down, for a
    // robot that has not moved since the second it landed. (Measured across the
    // two real falls in data3_webots, the NUbots baseline moves 0.0 m and 0.6 m.)
    // So the disturbed PSDs cover the dynamic part and then stand down, and the
    // event-sized part of the uncertainty is the one-shot inflation below.
    //
    // This windows the PSDs ONLY. Discarding the walk-engine linear velocity runs
    // for the whole time the robot is down: that is a claim about whether the signal
    // means anything, not about how long the event lasts, and it does not come back
    // after two seconds. See SystemLocalisation::setPosture.
    const double disturbedWindow = 2.0;                 ///< [s]

    // Belief handed back on recovery from a fall (see the recovery block below).
    // The position figure is deliberately at SideDisambiguator's maxPosStd so that
    // recovering also freezes background-map building; the yaw figure is what a
    // getup can plausibly reorient the robot by without the gyroscope catching it.
    //
    // FALL_GATE=off and FALL_INFLATE=<m>,<deg> exist to measure this design rather
    // than assume it: the first replays straight through a fall with every update
    // applied, the second changes what recovery hands back (0,0 disables it). Both
    // default to the behaviour described above.
    const bool fallGate = [] {
        const char * e = std::getenv("FALL_GATE");
        return e == nullptr || std::string_view(e) != "off";
    }();
    double recoveryPosStd = 0.50;                 ///< [m]
    double recoveryYawStd = 60.0*M_PI/180.0;      ///< [rad]
    if (const char * inflateEnv = std::getenv("FALL_INFLATE"))
    {
        const std::string s(inflateEnv);
        const std::size_t comma = s.find(',');
        recoveryPosStd = std::atof(s.substr(0, comma).c_str());
        recoveryYawStd = comma == std::string::npos ? 0.0
                       : std::atof(s.substr(comma + 1).c_str())*M_PI/180.0;
    }
    if (!fallGate)
    {
        std::println("FALL_GATE=off: measurement updates are NOT suppressed while not upright");
    }
    std::println("Recovery inflation: sigma_xy +{:.2f} m, sigma_yaw +{:.1f} deg",
                 recoveryPosStd, recoveryYawStd*180.0/M_PI);

    Posture prevPosture = Posture::UPRIGHT;
    double fallStartT = std::numeric_limits<double>::quiet_NaN();
    std::size_t nFalls = 0, nFallenFrames = 0;
    std::size_t nGravityRejected = 0;   ///< Frames where the specific force was not quasi-static
    std::size_t nZupt = 0;              ///< Zero-velocity updates applied while fallen

    for (const VisionSample & v : log.vision)
    {
        const double t = v.t - t0;
        if (t < tInit)
        {
            nSkipped++;
            // Still shown: the replay covers the whole recording, and the frames
            // before initialisation are where the grid solve's input can be
            // eyeballed. The belief drawn on them is the seeded prior, which is
            // what the filter is about to start from.
            captureSkipped(v, t, FrameSkip::BEFORE_INIT);
            continue;
        }

        // Posture gate. This used to suppress every update while the robot was not
        // upright, on the grounds that all four measurement models assume an
        // upright robot. That was only half right. The models that break are the
        // ones whose *assumption* breaks -- kinematic height needs the support leg
        // on the ground, and the accelerometer only reads gravity when the robot
        // is not being accelerated. The landmark and out-of-field models are plain
        // geometry: given the right attitude they are as valid face-down as
        // standing. What actually forced blanket suppression was the roll-pitch-yaw
        // state, which could not represent a fallen robot without passing through
        // gimbal lock; with a quaternion (see SystemLocalisation) it can, so the
        // gate is now per-model rather than all-or-nothing. That matters because a
        // fall is exactly when the estimate is most at risk: a robot that spins
        // while toppling or getting up changes its heading, and only measurements
        // taken during the event can catch it.
        //
        // FALL_GATE=off makes the whole run read as upright, so the recovery block
        // below never fires either -- the ablation is "no posture handling at all",
        // not "suppression off but recovery still on".
        Posture posture = fallGate ? falls.at(t) : Posture::UPRIGHT;
        if (fallGate && std::isfinite(fallT) && t >= fallT && t < fallT + fallDuration)
        {
            posture = Posture::FALLEN;      // Simulated fall (see FALL_T above)
        }
        const bool upright = posture == Posture::UPRIGHT;
        if (!upright)
        {
            if (prevPosture == Posture::UPRIGHT)
            {
                fallStartT = t;
                std::println("NOT UPRIGHT at t={:.2f} s ({}): suppressing kinematic height; landmark, "
                             "out-of-field and (quasi-static) gravity updates continue", t, to_string(posture));
            }
            nFallenFrames++;
        }
        // Posture goes in before anything can `continue` past it, and the fall start
        // is already set above, so the elapsed time is measured from THIS episode
        // rather than the previous one. The velocity discard runs for the whole time
        // the robot is down; only the elevated PSDs are windowed. See setPosture.
        system.params.disturbedWindow = disturbedWindow;
        // GYRO_BIAS=off freezes the bias state at its initial value, to tell a bias
        // that is measuring a sensor from one that is absorbing modelling error: if
        // freezing it IMPROVES the fit, the state was soaking up someone else's
        // inconsistency rather than estimating a gyroscope.
        {
            static const bool freezeGyroBias = [] {
                const char * e = std::getenv("GYRO_BIAS");
                return e != nullptr && std::string(e) == "off";
            }();
            if (freezeGyroBias) system.params.sigmaGyroBias = 1e-5;
        }
        const double fallElapsed = (!upright && std::isfinite(fallStartT)) ? t - fallStartT : 0.0;
        system.setPosture(upright, fallElapsed);

        // Recovery from a fall. The mean is kept: a fall and getup move the torso
        // well under a metre, so the pre-fall position is still the best estimate
        // available, and re-solving the pose globally would be worse -- the
        // start-of-game grid search resolves field symmetry from the known starting
        // half, a prior that is simply false once play is under way. What a fall
        // actually destroys is confidence, above all in yaw, so that is what is
        // given back. The inflated position std also exceeds SideDisambiguator's
        // maxPosStd, which freezes background-map building until the filter has
        // reconverged, so no landmarks are triangulated from the recovering pose.
        // Recovery fires on the transition back to upright, not merely on "the
        // last frame was not upright" -- now that not-upright frames run their
        // updates through this same path, the latter would re-inflate every frame
        // of the fall.
        if (prevPosture != Posture::UPRIGHT && upright)
        {
            // Yaw uncertainty is about the field z axis, which on the quaternion
            // states is a rank-one block rather than a single diagonal element --
            // there is no "the yaw element" any more.
            const Eigen::VectorXd xr = system.density.mean();
            Eigen::MatrixXd extraCov = Eigen::MatrixXd::Zero(SystemLocalisation::nx, SystemLocalisation::nx);
            extraCov(0, 0) = extraCov(1, 1) = recoveryPosStd*recoveryPosStd;
            const Eigen::Vector4d jYaw = SystemLocalisation::attitudeTangentField(xr).col(2);
            extraCov.block<4, 4>(SystemLocalisation::iQuat, SystemLocalisation::iQuat) =
                recoveryYawStd*recoveryYawStd*jYaw*jYaw.transpose();
            system.inflateCovariance(extraCov);

            // A fall is also a chance to have been turned around without the
            // landmarks noticing, and they can never notice: the two symmetric
            // field poses fit each other's landmarks identically. Re-seed the
            // mirror so the out-of-field evidence has something to switch to.
            if (useHypothesisBank && system.numHypotheses() == 1)
            {
                system.spawnMirror();
                lastRespawnT = t;
            }
            std::println("RECOVERED at t={:.2f} s after {:.2f} s not upright: inflated to "
                         "sigma_xy {:.2f} m, sigma_yaw {:.1f} deg ({} hypotheses)",
                         t, t - fallStartT, std::sqrt(system.density.cov()(0, 0)),
                         std::sqrt(SystemLocalisation::yawVariance(system.density.mean(),
                                                                   system.density.cov()))*180.0/M_PI,
                         system.numHypotheses());
            nFalls++;
        }
        prevPosture = posture;

        // Nearest odometry/IMU sample to this frame. Needed by the body-rate
        // measurements below as well as by the camera extrinsic further down.
        std::size_t k = nearestIndex(log.sensors, v.t, [](const SensorsSample & s) { return s.t; });

        constexpr bool useGravity = true;
        constexpr bool useKinematicHeight = true;
        constexpr bool useGyroscope = true;
        // Gyroscope noise is small and the sensor is trustworthy; the walk-engine
        // odometry velocity is neither, so it is admitted with a sigma comparable
        // to the walk speed itself -- present as evidence, never as truth. The
        // zero-velocity update is the most confident of the three because a robot
        // lying on the carpet really is not going anywhere.
        // SIGMA_GYRO / SIGMA_ODOM_VEL / SIGMA_VEL override these, for sweeping the
        // rate tuning against ground truth rather than guessing at it.
        static const double sigmaGyro    = envDouble("SIGMA_GYRO", 0.02);       ///< [rad/s]
        static const double sigmaOdomVel = envDouble("SIGMA_ODOM_VEL", 0.15);   ///< [m/s]
        constexpr double sigmaZupt        = 0.02;   ///< [m/s] lying still
        constexpr double sigmaZuptDynamic = 0.30;   ///< [m/s] toppling or getting up

        // Body-rate measurements go in before the no-detections gate below, because
        // neither of them has anything to do with whether YOLO found something. A
        // fallen robot's camera is in the carpet, so those are exactly the frames
        // that return no detections -- and exactly the frames where an unmeasured
        // velocity state would integrate the pre-fall gait straight off the field.
        // (Measurement::process predicts to t itself, so the predictAll below is a
        // zero-dt no-op once these have run.)
        // Body-rate measurements. These used to be a known input to the process
        // model; they are measurements of the velocity states now, so their noise
        // is modelled rather than asserted and the gyroscope's bias is estimated
        // rather than subtracted by heuristic up front.
        //
        // The gyroscope runs unconditionally, including through a fall: it is the
        // one sensor that measures a topple honestly, and its reading is valid
        // whatever the robot's posture.
        if (useGyroscope && log.sensors[k].gyroscope.allFinite())
        {
            MeasurementGyroscope mg(t, log.sensors[k].gyroscope, sigmaGyro);
            system.process(mg);
        }
        // The walk-engine odometry velocity, on the other hand, describes the gait
        // the engine believes it is executing. Upright that is loose but real
        // information; on the ground it is fiction, and during a getup it is a
        // scripted flail that is not locomotion. So it is suppressed while not
        // upright -- the same per-model gating as kinematic height, which is a
        // tidier way to say it than the special case that used to live inside the
        // process model's input().
        if (upright)
        {
            std::size_t vi = nearestIndex(twists, t, [](const BodyTwistSample & s) { return s.t; });
            if (!twists.empty() && std::abs(twists[vi].t - t) < 0.1 && twists[vi].vBb.allFinite())
            {
                MeasurementBodyVelocity mv(t, twists[vi].vBb, sigmaOdomVel);
                system.process(mv);
            }
        }
        else
        {
            // Zero-velocity update. A robot lying on the carpet is not translating,
            // and saying so is the most confident measurement available. It is also
            // the only thing stopping the pre-fall walking velocity from integrating
            // across the whole fall, which with velocity in the state is exactly how
            // the estimate walks off the field: measured across data4_webots,
            // leaving the toppling and getting-up frames unmeasured let the error
            // compound 0.12 -> 0.62 -> 1.76 m over two falls and never recover.
            //
            // It is applied through the whole non-upright window, not just while
            // FALLEN, because "no measurement" is not the neutral choice it looks
            // like -- it says the robot may still be travelling at whatever it was
            // doing when it fell, which is the one thing it is certainly not doing.
            // The sigma is what changes with posture: lying still the robot really
            // is stationary, whereas toppling and being levered upright genuinely
            // move the torso, just not anywhere.
            const double sigma = posture == Posture::FALLEN ? sigmaZupt : sigmaZuptDynamic;
            MeasurementBodyVelocity zupt = MeasurementBodyVelocity::stationary(t, sigma);
            system.process(zupt);
            nZupt++;
        }

        // Prediction used to happen only inside Event::process, so a frame that
        // produced no usable measurement advanced neither the state nor the clock.
        // A face-down fall produces exactly that (no detections at all), and the
        // filter would emerge from it holding its pre-fall mean at its pre-fall
        // covariance -- confidently wrong rather than honestly uncertain.
        //
        // This sits AFTER the recovery block on purpose. A getup ends in motion
        // blur, so the frame the robot first reads upright again is very often one
        // with no detections; skipping out above it meant recovery never ran on
        // those falls, and the filter came out holding pre-fall confidence in a mean
        // that had moved -- an association gate too narrow to ever re-acquire.
        const bool poseFinite = v.Hcw.rotationMatrix.allFinite() && v.Hcw.translationVector.allFinite();
        if (v.detections.empty() || !poseFinite)
        {
            system.predictAll(t);
            nSkipped++;
            captureSkipped(v, t, !poseFinite ? FrameSkip::BAD_POSE : FrameSkip::NO_DETECTIONS);
            continue;
        }

        Pose<double> Tbc = log.sensors[k].Htw*v.Hcw.inverse();

        // Simulated kidnap: mirror the filter state once, without notifying the
        // side disambiguator, and let it detect and correct the flip.
        if (!kidnapDone && std::isfinite(kidnapT) && t >= kidnapT)
        {
            system.resetTo(SystemLocalisation::mirrorDensity(system.density), t);
            kidnapDone = true;
            Eigen::VectorXd xk = system.density.mean();
            std::println("KIDNAP: state mirrored at t={:.2f} s -> x={:.2f} m, y={:.2f} m, yaw={:.1f} deg",
                         t, xk(0), xk(1), SystemLocalisation::heading(xk)*180.0/M_PI);
        }

        // Measurement toggles (for ablation experiments)
        constexpr bool useFieldLines = false;

        Eigen::Matrix<double, 3, Eigen::Dynamic> frameLineRays;  // captured for the viewer

        auto tic = std::chrono::steady_clock::now();
        // Route every measurement through system.process(): with the hypothesis
        // bank inactive this is identical to meas.process(system); with it active
        // each measurement is applied to every mixture component and its
        // log-evidence updates the component weights.
        MeasurementFieldLandmarks meas(t, v, Tbc, map, system);
        system.process(meas);
        if (useFieldLines && !log.linePoints.empty())
        {
            std::size_t li = nearestIndex(log.linePoints, v.t, [](const LinePointsSample & s) { return s.t; });
            const LinePointsSample & lp = log.linePoints[li];
            if (std::abs(lp.t - v.t) < 0.05)
            {
                std::size_t lk = nearestIndex(log.sensors, lp.t, [](const SensorsSample & s) { return s.t; });
                Pose<double> TbcLines = log.sensors[lk].Htw*lp.Hcw.inverse();
                MeasurementFieldLines measLines(t, lp, TbcLines, map, system);
                system.process(measLines);
                frameLineRays = lp.rays;
            }
        }
        // Unit-norm pseudo-measurement. The four attitude states carry three
        // degrees of freedom and quat2rot normalises, so |q| is invisible to
        // every other model here; without this the MAP Hessian is singular along
        // it and the Newton step has a flat direction to wander down. Applied
        // first so the rest of the frame's updates see a belief that is on the
        // sphere. See MeasurementQuaternionNorm.
        {
            MeasurementQuaternionNorm mqn(t);
            system.process(mqn);
        }
        // Gravity is valid whenever the torso is not being accelerated -- which is
        // true of a robot lying still on the carpet and false of one in free fall
        // or hitting the ground, regardless of posture. Gating on the specific
        // force magnitude tests that directly, which is both the right condition
        // during a fall and a better one than "upright" while walking. The
        // threshold is deliberately loose: ordinary gait swings the magnitude by
        // a couple of m/s^2, and the model already carries a 1 m/s^2 noise.
        constexpr double kQuasiStaticTol = 3.0;    ///< [m/s^2] from standard gravity
        if (useGravity && log.sensors[k].accelerometer.allFinite())
        {
            const double aMag = log.sensors[k].accelerometer.norm();
            if (std::abs(aMag - 9.80665) < kQuasiStaticTol)
            {
                MeasurementGravity mg(t, log.sensors[k].accelerometer);
                system.process(mg);
            }
            else
            {
                nGravityRejected++;
            }
        }
        // Torso height above ground assumes the support leg reaches the ground, so
        // this is the one model a fall genuinely invalidates: lying down, the
        // chain still reports a near-upright 0.44 m torso and would fight the
        // attitude the other measurements are establishing.
        if (useKinematicHeight && upright)
        {
            // Torso height above ground from the odometry/kinematic chain
            double h = log.sensors[k].Htw.inverse().translationVector.z();
            if (std::isfinite(h))
            {
                MeasurementKinematicHeight mh(t, h);
                system.process(mh);
            }
        }
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tic).count();
        nUpdates++;
        sumMs += ms;
        maxMs = std::max(maxMs, ms);

        if (!haveStat && t > 6.0 && t < 8.0 && meas.numAssociated() >= 6)
        {
            statFrame = v;
            statTbc = Tbc;
            statMean = system.density.mean();
            haveStat = true;
        }

        // Record state and comparison to NUbots baseline
        Record r;
        r.t = t;
        r.mean = system.density.mean();
        r.sigma = system.density.cov().diagonal().cwiseSqrt();
        r.rpy = SystemLocalisation::attitudeRpy(r.mean);
        r.attStd = SystemLocalisation::attitudeStd(r.mean, system.density.cov());
        r.nAssoc = meas.numAssociated();
        r.nCand = meas.numCandidates();
        r.updateMs = ms;
        r.baseX = r.baseY = r.baseYaw = r.baseCost = r.errXY = r.errYaw = std::numeric_limits<double>::quiet_NaN();
        r.sideLlr = std::numeric_limits<double>::quiet_NaN();
        r.oofAssoc = 0;
        r.truthX = r.truthY = r.truthZ = r.truthYaw = std::numeric_limits<double>::quiet_NaN();
        r.errXYTruth = r.errYawTruth = std::numeric_limits<double>::quiet_NaN();

        if (!log.fieldBaseline.empty())
        {
            std::size_t bi = nearestIndex(log.fieldBaseline, v.t, [](const FieldBaselineSample & s) { return s.t; });
            const FieldBaselineSample & b = log.fieldBaseline[bi];
            if (std::abs(b.t - v.t) < 0.15 && b.Hfw.rotationMatrix.allFinite() && b.Hfw.translationVector.allFinite())
            {
                std::size_t sk = nearestIndex(log.sensors, b.t, [](const SensorsSample & s) { return s.t; });
                Pose<double> TfbBase = b.Hfw*log.sensors[sk].Htw.inverse();
                Eigen::Vector3d rpyBase = rot2rpy(TfbBase.rotationMatrix);
                r.baseX = TfbBase.translationVector.x();
                r.baseY = TfbBase.translationVector.y();
                r.baseYaw = rpyBase.z();
                r.baseCost = b.cost;
                r.errXY = (r.mean.head<2>() - TfbBase.translationVector.head<2>()).norm();
                r.errYaw = wrapAngle(r.rpy(2) - r.baseYaw);
                sumSqErrXY += r.errXY*r.errXY;
                sumSqErrYaw += r.errYaw*r.errYaw;
                nCompared++;

                // DIAGNOSTIC: which pose does the vision support during the initial
                // stationary window? Compare landmark reprojection residual at our
                // estimate vs the baseline pose.
                if (t < 10.0 && std::isfinite(b.cost) && b.cost < trustedBaselineCost)
                {
                    Eigen::VectorXd baseEta = Eigen::VectorXd::Zero(SystemLocalisation::nx);
                    baseEta.head<3>() = TfbBase.translationVector;
                    baseEta.segment<4>(SystemLocalisation::iQuat) = rot2quat(TfbBase.rotationMatrix);
                    auto [rs, ns] = meanReprojResidual(r.mean, v, Tbc);
                    auto [rb, nb] = meanReprojResidual(baseEta, v, Tbc);
                    if (ns > 0 && nb > 0)
                    {
                        sumResidSelf += rs;
                        sumResidBase += rb;
                        nResid++;
                    }
                }
                if (std::isfinite(b.cost) && b.cost < trustedBaselineCost)
                {
                    sumSqErrXYTrusted += r.errXY*r.errXY;
                    sumSqErrYawTrusted += r.errYaw*r.errYaw;
                    nTrusted++;
                }
            }
        }
        // Mocap ground truth at this frame (evaluation only). The mocap stream
        // runs at ~120 Hz, so the nearest sample is within a few ms.
        if (!truth.empty())
        {
            std::size_t ti = nearestIndex(truth, t, [](const TruthSample & s) { return s.t; });
            const TruthSample & g = truth[ti];
            if (std::abs(g.t - t) < 0.05)
            {
                r.truthX = g.rBFf.x();
                r.truthY = g.rBFf.y();
                r.truthZ = g.rBFf.z();
                r.truthYaw = g.yaw;
                r.errXYTruth = (r.mean.head<2>() - g.rBFf.head<2>()).norm();
                r.errYawTruth = wrapAngle(r.rpy(2) - g.yaw);
                sumSqTruthXY += r.errXYTruth*r.errXYTruth;
                sumSqTruthYaw += r.errYawTruth*r.errYawTruth;
                sumZerr += r.mean(2) - g.rBFf.z();
                sumSqZerr += (r.mean(2) - g.rBFf.z())*(r.mean(2) - g.rBFf.z());
                nTruth++;
                if (std::isfinite(r.baseX))
                {
                    const double be = std::hypot(r.baseX - g.rBFf.x(), r.baseY - g.rBFf.y());
                    const double by = wrapAngle(r.baseYaw - g.yaw);
                    sumSqBaseTruthXY += be*be;
                    sumSqBaseTruthYaw += by*by;
                    sumSqTruthXYBoth += r.errXYTruth*r.errXYTruth;
                    sumSqTruthYawBoth += r.errYawTruth*r.errYawTruth;
                    nTruthBoth++;
                }
            }
        }

        // Estimated camera pose in {f} at the posterior mean (with mount-bias
        // correction), shared by the out-of-field pipeline and the visualiser.
        const Pose<double> TfcEst = SystemLocalisation::fieldPose<double>(r.mean)*Tbc
                 *Pose<double>(SystemLocalisation::cameraBiasRotation<double>(r.mean), Eigen::Vector3d::Zero());

        // Out-of-field side disambiguation from the raw video frame: map the
        // background corners and compare the pose against its 180 deg mirror.
        SideDisambiguator::FrameResult side;
        bool sideRan = false;
        if (useOutOfField && fetchVideoFrame(v.videoFrame))
        {
            auto ticOof = std::chrono::steady_clock::now();
            cv::Mat gray;
            cv::cvtColor(videoFrameBgr, gray, cv::COLOR_BGR2GRAY);

            // Camera pose under the mirrored state (same kinematics, mirrored torso pose).
            Eigen::VectorXd mirrorEta = SystemLocalisation::mirrorState(r.mean);
            const Pose<double> TfcMirror = SystemLocalisation::fieldPose<double>(mirrorEta)*Tbc
                     *Pose<double>(SystemLocalisation::cameraBiasRotation<double>(mirrorEta), Eigen::Vector3d::Zero());

            side = sideDis.process(t, gray, TfcEst, TfcMirror,
                                   std::max(r.sigma(0), r.sigma(1)), r.attStd(2),
                                   std::abs(log.sensors[k].gyroscope.z()), r.rpy(2));
            sideRan = true;
            double oofMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ticOof).count();
            sumOofMs += oofMs;
            maxOofMs = std::max(maxOofMs, oofMs);
            nOofFrames++;
            sumOofTotal += side.nFeatures;
            sumOofOut += side.nOutOfField;
            sumOofAssoc += side.nAssociated;
            r.sideLlr = side.llr;
            r.oofAssoc = side.nAssociated;

            if (nOofFrames % 100 == 0 || side.llr < 0.0)
            {
                std::println("  side t={:6.1f}s: {} landmarks, {} candidates, assoc {}/{} visible {}/{} own/mirror, "
                             "{} corners rejected, llr {:+.1f}{} turn {:+.0f}deg{}",
                             t, side.nLandmarks, side.nCandidates, side.nAssociated, side.nAssociatedMirror,
                             side.nVisibleOwn, side.nVisibleMirror, side.nOutlier, side.llr,
                             side.mapFrozen ? " [frozen]" : "",
                             side.turnSinceMatch*180.0/M_PI,
                             side.blindTurnBlocked ? " [turn-blocked]" : "");
            }

            if (useHypothesisBank)
            {
                // Bank mode: the background evidence is the ONLY thing that can
                // separate the two symmetric hypotheses (landmarks leave them at
                // 50/50), so fold each frame's own-vs-mirror log-ratio into the
                // mixture weights. A sustained lead collapses the mirror; the
                // representative switching sides IS the correction, done smoothly
                // by the weights rather than by a discontinuous state flip.
                system.addSideLogEvidence(side.sideDelta);

                // If the mirror has already been pruned and the disambiguator is
                // now sure the lone belief is wrong (mid-game kidnap), re-seed the
                // alternative so the weights have something to switch to. A
                // cooldown stops repeated spawns while flipRequested stays latched
                // (the LLR takes a few seconds to climb back after the switch).
                if (side.flipRequested && system.numHypotheses() == 1
                    && t - lastRespawnT > sideDis.options.flipCooldown)
                {
                    system.spawnMirror();
                    lastRespawnT = t;
                    nSideFlips++;
                    std::println("SIDE RESPAWN at t={:.2f} s (llr={:+.1f}): re-seeded the mirror hypothesis "
                                 "for the weights to resolve", t, side.llr);
                }
            }
            else if (side.flipRequested)
            {
                // Single-hypothesis mode: the background evidence says we are on
                // the wrong side, so mirror the belief and tell the disambiguator
                // the flip happened.
                system.resetTo(SystemLocalisation::mirrorDensity(system.density), t);
                sideDis.notifyFlipApplied(t);
                nSideFlips++;
                Eigen::VectorXd xf = system.density.mean();
                std::println("SIDE FLIP at t={:.2f} s (llr={:+.1f}, assoc {}/{} own/mirror): corrected to x={:.2f} m, y={:.2f} m, yaw={:.1f} deg",
                             t, side.llr, side.nAssociated, side.nAssociatedMirror,
                             xf(0), xf(1), SystemLocalisation::heading(xf)*180.0/M_PI);
            }
        }
        records.push_back(r);

        // Capture the per-frame view for the interactive visualiser / mp4 export.
        if (captureFrames)
        {
            ViewerFrame vf;
            vf.videoFrame = v.videoFrame;
            vf.t = t;
            vf.Tfc = TfcEst;
            captureBelief(vf, r.mean);

            // Raw YOLO detections (boxes), each flagged with what the landmark
            // measurement did with it: associated, gated out, dropped for low
            // confidence, or never a landmark class at all.
            auto usedClass = [](const std::string & n) {
                return n == "L-intersection" || n == "T-intersection" || n == "X-intersection" || n == "goal post";
            };
            std::vector<DetectionStatus> detStatus(v.detections.size(), DetectionStatus::UNUSED_CLASS);
            for (std::size_t i = 0; i < v.detections.size(); ++i)
            {
                if (usedClass(v.detections[i].name))
                {
                    detStatus[i] = DetectionStatus::LOW_CONFIDENCE;
                }
            }
            for (const auto & o : meas.detectionOutcomes())
            {
                // Every detection that reached association cleared the class and
                // confidence filters, so it is at worst UNASSOCIATED.
                detStatus[o.detection] = o.associated ? DetectionStatus::ASSOCIATED
                                                      : DetectionStatus::UNASSOCIATED;
            }
            for (std::size_t i = 0; i < v.detections.size(); ++i)
            {
                if (!v.detections[i].corners.allFinite()) continue;
                DetectionView dv;
                dv.name = v.detections[i].name;
                dv.confidence = v.detections[i].confidence;
                dv.corners = v.detections[i].corners;
                dv.status = detStatus[i];
                vf.detections.push_back(std::move(dv));
            }

            // Accepted associations (measured ray <-> map landmark).
            const auto & um = meas.measuredRays();
            const auto & lmk = meas.associatedLandmarks();
            for (Eigen::Index j = 0; j < um.cols() && j < lmk.cols(); ++j)
            {
                AssociationView av;
                av.measRay = um.col(j);
                av.landmark = lmk.col(j);
                vf.associations.push_back(av);
            }

            // Out-of-field corner features (on-carpet rejects kept for mask verification).
            if (sideRan)
            {
                for (std::size_t i = 0; i < side.features.size(); ++i)
                {
                    vf.oofFeatures.push_back({side.features[i].px, side.featureStatus[i]});
                }
                // Map landmarks projected into this camera, with what became of
                // each (in FOV / associated / missed / culled as an outlier).
                vf.oofLandmarkProj.reserve(side.landmarkViews.size());
                for (const SideDisambiguator::LandmarkView & lv : side.landmarkViews)
                {
                    vf.oofLandmarkProj.push_back({lv.px, lv.matchPx, lv.status, lv.far});
                }
                vf.hasSide = true;
                vf.sideLlr = side.llr;
                vf.nOofLandmarks = side.nLandmarks;
                vf.sideFrozen = side.mapFrozen;
                vf.oofVisibleMargin = sideDis.options.visibleMargin;

                // Snapshot of the out-of-field landmark map for the 3D panel.
                vf.oofLandmarks.reserve(sideDis.landmarks().size());
                for (const SideDisambiguator::Landmark & lm : sideDis.landmarks())
                {
                    vf.oofLandmarks.push_back({lm.rPFf, lm.P, lm.far, lm.lastStatus});
                }
            }

            vf.lineRays = frameLineRays;
            if (std::isfinite(r.truthX))
            {
                vf.hasTruth = true;
                vf.truthPos = Eigen::Vector3d(r.truthX, r.truthY, r.truthZ);
                vf.truthYaw = r.truthYaw;
            }
            if (std::isfinite(r.baseX))
            {
                vf.hasBaseline = true;
                vf.basePos = Eigen::Vector2d(r.baseX, r.baseY);
                vf.baseYaw = r.baseYaw;
                vf.baseCost = r.baseCost;
                vf.hasError = true;
                vf.errXY = r.errXY;
                vf.errYaw = r.errYaw;
            }
            vf.nAssoc = r.nAssoc;
            vf.nCand = r.nCand;
            vf.updateMs = r.updateMs;
            viewFrames.push_back(std::move(vf));
        }
    }

    // Rebuild the viewer timeline as the video timeline. Vision samples are not
    // one-to-one with video frames -- messages get dropped and the two streams
    // run at slightly different rates -- so one entry per sample silently drops
    // footage. Walking the frames instead and emitting a carried-forward entry
    // wherever no sample landed makes the replay play the recording through, at
    // the cost of some frames on which the overlay simply does not change.
    if (captureFrames && !viewFrames.empty())
    {
        // What a frame with no sample of its own inherits. Held by value: a
        // reference into `dense` would dangle as the vector grows.
        ViewerFrame carried;
        carried.skip = FrameSkip::NO_SAMPLE;
        auto carry = [&carried](const ViewerFrame & src) {
            carried.Tfc = src.Tfc;
            carried.estPos = src.estPos;
            carried.estYaw = src.estYaw;
            carried.estCov = src.estCov;
            carried.estPos3 = src.estPos3;
            carried.estCov3 = src.estCov3;
            carried.hypotheses = src.hypotheses;
        };
        carry(viewFrames.front());   // Leading frames show the seeded prior

        const int nVideo = static_cast<int>(log.frameTimes.size());
        std::vector<ViewerFrame> dense;
        dense.reserve(static_cast<std::size_t>(nVideo) + viewFrames.size());
        std::size_t cursor = 0;
        for (int f = 0; f < nVideo; ++f)
        {
            // Samples belonging at or before this frame. A sample that matched no
            // frame at all (videoFrame == -1) is emitted where the walk reaches
            // it, which is its right place in time since samples are in order.
            while (cursor < viewFrames.size()
                   && (viewFrames[cursor].videoFrame < 0 || viewFrames[cursor].videoFrame <= f))
            {
                carry(viewFrames[cursor]);
                dense.push_back(std::move(viewFrames[cursor]));
                ++cursor;
            }
            if (dense.empty() || dense.back().videoFrame != f)
            {
                ViewerFrame gap = carried;
                gap.videoFrame = f;
                gap.t = log.frameTimes[static_cast<std::size_t>(f)] - t0;
                captureTruth(gap, gap.t);
                dense.push_back(std::move(gap));
            }
        }
        // Samples timestamped past the end of the video, if any.
        for (; cursor < viewFrames.size(); ++cursor)
        {
            dense.push_back(std::move(viewFrames[cursor]));
        }

        std::println("Viewer timeline: {} frames over {} video frames ({} carried forward where no "
                     "vision sample matched)",
                     dense.size(), nVideo, dense.size() - viewFrames.size());
        viewFrames.swap(dense);
    }

    // Summary
    std::println("");
    std::println("Processed {} vision updates ({} skipped, of which {} while not upright)",
                 nUpdates, nSkipped, nFallenFrames);
    {
        // Does the bias state actually converge? That is the whole reason it exists,
        // and RMSE alone will not say -- a bias stuck at zero looks fine until the
        // heading drifts. The old heuristic measured ~0.75 deg/s of z-bias on data2
        // by averaging quiet samples, so that is the number to recognise here.
        const Eigen::VectorXd muEnd = system.density.mean();
        const Eigen::MatrixXd Pend = system.density.cov();
        const Eigen::Vector3d b = SystemLocalisation::gyroBias(muEnd);
        const Eigen::Vector3d bs = Pend.block<3, 3>(SystemLocalisation::iGyroBias,
                                                    SystemLocalisation::iGyroBias)
                                       .diagonal().cwiseMax(0.0).cwiseSqrt();
        std::println("Estimated gyro bias: [{:+.3f}, {:+.3f}, {:+.3f}] deg/s "
                     "(sigma [{:.3f}, {:.3f}, {:.3f}] deg/s)",
                     b.x()*180.0/M_PI, b.y()*180.0/M_PI, b.z()*180.0/M_PI,
                     bs.x()*180.0/M_PI, bs.y()*180.0/M_PI, bs.z()*180.0/M_PI);
        std::println("Final body velocity: [{:+.3f}, {:+.3f}, {:+.3f}] m/s",
                     SystemLocalisation::bodyVelocity(muEnd).x(),
                     SystemLocalisation::bodyVelocity(muEnd).y(),
                     SystemLocalisation::bodyVelocity(muEnd).z());
    }
    std::println("Zero-velocity updates applied while not upright: {}", nZupt);
    std::println("Falls recovered from: {} ({} frames not upright, {} gravity updates rejected as non-quasi-static)",
                 nFalls, nFallenFrames, nGravityRejected);
    {
        Eigen::VectorXd xFinal = system.density.mean();
        std::println("Final camera mount bias estimate: roll {:.2f} deg, pitch {:.2f} deg",
                     xFinal(SystemLocalisation::iBias)*180.0/M_PI,
                     xFinal(SystemLocalisation::iBias + 1)*180.0/M_PI);
    }
    std::println("Mean update time {:.2f} ms, max {:.2f} ms", nUpdates ? sumMs/nUpdates : 0.0, maxMs);
    if (nCompared > 0)
    {
        std::println("vs NUbots baseline over {} samples: RMSE position {:.3f} m, yaw {:.2f} deg",
                     nCompared, std::sqrt(sumSqErrXY/nCompared), std::sqrt(sumSqErrYaw/nCompared)*180.0/M_PI);
    }
    if (nTrusted > 0)
    {
        std::println("vs trusted baseline (cost < {:.1f}) over {} samples: RMSE position {:.3f} m, yaw {:.2f} deg",
                     trustedBaselineCost, nTrusted, std::sqrt(sumSqErrXYTrusted/nTrusted), std::sqrt(sumSqErrYawTrusted/nTrusted)*180.0/M_PI);
    }
    if (nTruth > 0)
    {
        std::println("");
        std::println("vs {} GROUND TRUTH over {} samples: ours RMSE position {:.3f} m, yaw {:.2f} deg",
                     to_string(truthSource), nTruth,
                     std::sqrt(sumSqTruthXY/nTruth), std::sqrt(sumSqTruthYaw/nTruth)*180.0/M_PI);
        if (nTruthBoth > 0)
        {
            std::println("head-to-head on the {} samples where the NUbots baseline also reported:", nTruthBoth);
            std::println("  ours            RMSE position {:.3f} m, yaw {:.2f} deg",
                         std::sqrt(sumSqTruthXYBoth/nTruthBoth), std::sqrt(sumSqTruthYawBoth/nTruthBoth)*180.0/M_PI);
            std::println("  NUbots baseline RMSE position {:.3f} m, yaw {:.2f} deg",
                         std::sqrt(sumSqBaseTruthXY/nTruthBoth), std::sqrt(sumSqBaseTruthYaw/nTruthBoth)*180.0/M_PI);
        }
        const double zMean = sumZerr/nTruth;
        const double zStd = std::sqrt(std::max(0.0, sumSqZerr/nTruth - zMean*zMean));
        // What "truth z" is differs by source, and so does the figure to expect.
        std::println("height: est z - truth z = {:+.3f} m mean, {:.3f} m std ({})", zMean, zStd,
                     truthSource == TruthSource::SIMULATOR
                         ? "truth z is the torso origin, so 0 is the target"
                         : "markers sit ~6 cm above the torso origin, so ~-0.06 m is expected");
    }
    if (nOofFrames > 0)
    {
        std::println("Out-of-field features: {:.0f} corners/frame ({:.0f} out-of-field, {:.1f} associated), {:.2f} ms mean / {:.2f} ms max over {} frames",
                     static_cast<double>(sumOofTotal)/nOofFrames, static_cast<double>(sumOofOut)/nOofFrames,
                     static_cast<double>(sumOofAssoc)/nOofFrames, sumOofMs/nOofFrames, maxOofMs, nOofFrames);
        std::println("Side disambiguator: {} map landmarks, final llr {:+.1f} nats, {} flips",
                     sideDis.landmarks().size(), sideDis.llr(), nSideFlips);
        const SideDisambiguator::Stats & ss = sideDis.stats();
        std::println("  map funnel: {} promote attempts ({} parallax-wait, tri fail {} geom / {} range / {} chi2, {} on-carpet, {} jittery) -> {} point + {} bearing-only ({} upgraded); culled {} chi2 / {} miss",
                     ss.promoteAttempts, ss.parallaxWait, ss.triFailGeometry, ss.triFailRange, ss.triFailChi2,
                     ss.backgroundFail, ss.farSpreadFail, ss.promoted, ss.promotedFar, ss.upgraded,
                     ss.landmarkCulledChi2, ss.landmarkCulledMiss);
        // Depth quality of the surviving map. Bearing-only landmarks carry no
        // triangulated depth at all -- they sit at the assumed range along their
        // bearing -- so they cluster on a shell around whichever camera position
        // anchored them. Worth stating plainly: it is the dominant visual feature
        // of the 3D map and looks like a bug if you do not know to expect it.
        {
            std::vector<double> extFar, extPoint;
            for (const SideDisambiguator::Landmark & lm : sideDis.landmarks())
            {
                Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(lm.P);
                const double ext = 3.0*std::sqrt(std::max(0.0, es.eigenvalues()(2)));
                (lm.far ? extFar : extPoint).push_back(ext);
            }
            auto median = [](std::vector<double> & v) {
                if (v.empty()) return std::numeric_limits<double>::quiet_NaN();
                std::sort(v.begin(), v.end());
                return v[v.size()/2];
            };
            std::println("  map depth: {} bearing-only at the assumed {:.1f} m (median 3-sigma {:.1f} m along the "
                         "unobserved depth axis), {} triangulated (median 3-sigma {:.1f} m)",
                         extFar.size(), sideDis.options.assumedRange, median(extFar),
                         extPoint.size(), median(extPoint));
        }
    }
    if (nResid > 0)
    {
        std::println("Stationary landmark reprojection residual: ours {:.2f} deg vs baseline {:.2f} deg over {} frames "
                     "(lower = better explains the detections)",
                     sumResidSelf/nResid*180.0/M_PI, sumResidBase/nResid*180.0/M_PI, nResid);
    }
    if (haveStat)
    {
        std::println("Residual-vs-offset probe at a stationary frame (converged x={:.2f}, y={:.2f}):",
                     statMean(0), statMean(1));
        for (int axis = 0; axis < 2; ++axis)
        {
            std::string s = axis == 0 ? "  dx" : "  dy";
            for (double d : {-1.0, -0.7, -0.5, -0.3, 0.0, 0.3, 0.5, 0.7, 1.0})
            {
                Eigen::VectorXd eta = statMean;
                eta(axis) += d;
                auto [res, n] = meanReprojResidual(eta, statFrame, statTbc);
                s += std::format("  {:+.2f}m:{:.2f}deg", d, res*180.0/M_PI);
            }
            std::println("{}", s);
        }
    }

    // Trajectory plot: trails as connected polylines (baseline underneath, truth,
    // then our estimate on top), start/end markers, then the covariance corridor
    // blended translucently so it shades rather than scribbles.
    FieldPlot plot(map.dims);
    for (std::size_t i = 1; i < records.size(); ++i)
    {
        // Skip teleports: the baseline re-localises in jumps, and a straight
        // line across such a jump is not a path the robot took.
        if (std::isfinite(records[i-1].baseX) && std::isfinite(records[i].baseX)
            && std::hypot(records[i].baseX - records[i-1].baseX,
                          records[i].baseY - records[i-1].baseY) < 0.5)
        {
            plot.drawSegment(records[i-1].baseX, records[i-1].baseY,
                             records[i].baseX, records[i].baseY, plotcolour::baseline, 2);
        }
    }
    for (std::size_t i = 1; i < records.size(); ++i)
    {
        if (std::isfinite(records[i-1].truthX) && std::isfinite(records[i].truthX))
        {
            plot.drawSegment(records[i-1].truthX, records[i-1].truthY,
                             records[i].truthX, records[i].truthY, plotcolour::truth, 2);
        }
    }
    for (std::size_t i = 1; i < records.size(); ++i)
    {
        const double frac = static_cast<double>(i)/(records.size() - 1);
        plot.drawSegment(records[i-1].mean(0), records[i-1].mean(1),
                         records[i].mean(0), records[i].mean(1), plotcolour::estimate(frac), 2);
    }
    if (!records.empty())
    {
        plot.drawRing(records.front().mean(0), records.front().mean(1), plotcolour::estimate(0.0), 7);
        plot.drawRing(records.back().mean(0), records.back().mean(1), plotcolour::estimate(1.0), 7);
        for (auto it = records.rbegin(); it != records.rend(); ++it)
        {
            if (std::isfinite(it->truthX))
            {
                plot.drawRing(it->truthX, it->truthY, plotcolour::truth, 7);
                break;
            }
        }
    }
    double lastEllipse = -1e9;
    for (const Record & r : records)
    {
        if (r.t - lastEllipse >= 2.0)
        {
            Eigen::Matrix2d Pxy;
            Pxy << r.sigma(0)*r.sigma(0), 0, 0, r.sigma(1)*r.sigma(1);
            plot.drawCovarianceEllipse(r.mean(0), r.mean(1), Pxy, plotcolour::covariance);
            lastEllipse = r.t;
        }
    }
    plot.blendOverlay();

    plot.drawTitle("RoboCup field localisation (top-down)",
                   nTruth > 0
                       ? std::format("{} vision updates over {:.0f} s   |   RMSE vs mocap truth: ours {:.2f} m, {:.1f} deg{}",
                                     nUpdates, records.empty() ? 0.0 : records.back().t - records.front().t,
                                     std::sqrt(sumSqTruthXY/nTruth), std::sqrt(sumSqTruthYaw/nTruth)*180.0/M_PI,
                                     nTruthBoth > 0
                                         ? std::format("  |  baseline {:.2f} m, {:.1f} deg",
                                                       std::sqrt(sumSqBaseTruthXY/nTruthBoth), std::sqrt(sumSqBaseTruthYaw/nTruthBoth)*180.0/M_PI)
                                         : std::string{})
                   : nTrusted > 0
                       ? std::format("{} vision updates over {:.0f} s   |   RMSE vs trusted baseline: {:.2f} m, {:.1f} deg",
                                     nUpdates, records.empty() ? 0.0 : records.back().t - records.front().t,
                                     std::sqrt(sumSqErrXYTrusted/nTrusted), std::sqrt(sumSqErrYawTrusted/nTrusted)*180.0/M_PI)
                       : std::format("{} vision updates", nUpdates));
    plot.drawLegend();

    if (!outputDirectory.empty())
    {
        std::filesystem::create_directories(outputDirectory);

        // CSV export
        std::ofstream csv(outputDirectory / "field_localisation.csv");
        csv << "t,x,y,z,roll,pitch,yaw,sx,sy,sz,sroll,spitch,syaw,camBiasRoll,camBiasPitch,sCamRoll,sCamPitch,nAssoc,nCand,baseX,baseY,baseYaw,baseCost,errXY,errYaw,updateMs,sideLlr,oofAssoc,truthX,truthY,truthZ,truthYaw,errXYTruth,errYawTruth\n";
        for (const Record & r : records)
        {
            // The CSV columns stay (x, y, z, roll, pitch, yaw): the state carries a
            // quaternion now, so the attitude columns are derived rather than copied.
            csv << r.t;
            for (int i = 0; i < 3; ++i) csv << ',' << r.mean(i);
            for (int i = 0; i < 3; ++i) csv << ',' << r.rpy(i);
            for (int i = 0; i < 3; ++i) csv << ',' << r.sigma(i);
            for (int i = 0; i < 3; ++i) csv << ',' << r.attStd(i);
            csv << ',' << r.mean(SystemLocalisation::iBias) << ',' << r.mean(SystemLocalisation::iBias + 1)
                << ',' << r.sigma(SystemLocalisation::iBias) << ',' << r.sigma(SystemLocalisation::iBias + 1);
            csv << ',' << r.nAssoc << ',' << r.nCand
                << ',' << r.baseX << ',' << r.baseY << ',' << r.baseYaw << ',' << r.baseCost
                << ',' << r.errXY << ',' << r.errYaw << ',' << r.updateMs
                << ',' << r.sideLlr << ',' << r.oofAssoc
                << ',' << r.truthX << ',' << r.truthY << ',' << r.truthZ << ',' << r.truthYaw
                << ',' << r.errXYTruth << ',' << r.errYawTruth << '\n';
        }

        cv::imwrite((outputDirectory / "field_trajectory.png").string(), plot.image());
        std::println("Exported field_localisation.csv and field_trajectory.png to {}", outputDirectory.string());
    }

    // Two-panel visualiser (camera with re-projected detections/associations on
    // the left, top-down field with hypotheses on the right). With --export the
    // whole replay is rendered to an mp4 (no display needed); with --interactive
    // it also opens the live scrubbable window. Mode 1 auto-plays; mode 2 steps.
    if (captureFrames)
    {
        LocalisationViewer viewer(map, lens, dataDir / "Left.mp4");
        // VIEWER_3D=1 opens straight into the 3D pane ('3' still toggles).
        viewer.setRightPane3D(std::getenv("VIEWER_3D") != nullptr);
        if (!outputDirectory.empty() && !std::getenv("NO_MP4"))
        {
            viewer.exportVideo(viewFrames, outputDirectory / "localisation.mp4");
        }
        if (interactive > 0)
        {
            viewer.run(viewFrames, interactive, outputDirectory);
        }
    }
}
