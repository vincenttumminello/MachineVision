#include "LocalisationViewer.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <format>
#include <print>
#include <Eigen/Eigenvalues>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include "SideDisambiguator.h"   // FeatureStatus / LandmarkStatus colour key

namespace
{

constexpr int    kPanelH   = 760*1.25;    ///< Panel height [px]
constexpr int    kHeaderH  = 36*1.25;     ///< Header strip height [px]
constexpr double kHalfFov  = 75.0*M_PI/180.0;  ///< Half field-of-view for the top-down wedge

// True capture rate, derived from the recorded frame times. Container fps metadata
// is unreliable for these recordings (e.g. a 10 fps capture whose mp4 claims 25),
// and different datasets use different rates, so real-time playback speed comes
// from the median spacing of the frames actually being replayed.
double playbackFps(const std::vector<ViewerFrame> & frames)
{
    std::vector<double> dts;
    dts.reserve(frames.size());
    for (std::size_t i = 1; i < frames.size(); ++i)
    {
        const double dt = frames[i].t - frames[i-1].t;
        if (dt > 1e-4) dts.push_back(dt);
    }
    if (dts.empty()) return 10.0;
    std::nth_element(dts.begin(), dts.begin() + dts.size()/2, dts.end());
    return 1.0/dts[dts.size()/2];
}

// -------- colours (BGR) --------
cv::Scalar classColour(const std::string & name)
{
    if (name == "goal post")       return cv::Scalar(0, 215, 255);   // gold
    if (name == "L-intersection")  return cv::Scalar(90, 230, 90);   // green
    if (name == "T-intersection")  return cv::Scalar(255, 200, 60);  // cyan-blue
    if (name == "X-intersection")  return cv::Scalar(220, 90, 220);  // magenta
    if (name == "ball")            return cv::Scalar(0, 140, 255);    // orange
    if (name == "robot")           return cv::Scalar(70, 70, 255);    // red
    return cv::Scalar(200, 200, 200);
}

// Measurement ray for a detection (box centre for intersections, base for posts),
// matching MeasurementFieldLandmarks::detectionRay.
bool measurementRay(const DetectionView & d, Eigen::Vector3d & ray)
{
    if (d.name == "goal post")
    {
        ray = d.corners.col(2) + d.corners.col(3);   // BR + BL (base)
    }
    else
    {
        ray = d.corners.rowwise().sum();              // box centre
    }
    if (!ray.allFinite() || ray.norm() < 1e-9) return false;
    ray.normalize();
    return true;
}

// Draw text with a semi-opaque dark background for legibility.
void hudText(cv::Mat & img, const std::string & s, cv::Point org, double scale = 0.42,
             cv::Scalar colour = cv::Scalar(240, 240, 240), int thickness = 1)
{
    int base = 0;
    cv::Size sz = cv::getTextSize(s, cv::FONT_HERSHEY_SIMPLEX, scale, thickness, &base);
    cv::rectangle(img, org + cv::Point(-3, 3), org + cv::Point(sz.width + 3, -sz.height - 3),
                  cv::Scalar(0, 0, 0), cv::FILLED);
    cv::putText(img, s, org, cv::FONT_HERSHEY_SIMPLEX, scale, colour, thickness, cv::LINE_AA);
}

// -------- association colour key (shared by the camera panel and its legend) --------
// One palette for "what did the estimator do with this observation", used by both
// landmark streams so the two panels read the same way: green = associated,
// amber = seen but unclaimed, red = rejected, grey = never in play.
const cv::Scalar kColAssociated(120, 230, 120);   // green:   matched a map landmark
const cv::Scalar kColMissed    (60, 190, 255);    // amber:   predicted visible, nothing matched it
const cv::Scalar kColOutlier   (60, 60, 255);     // red:     gated but rejected on the evidence
const cv::Scalar kColMirror    (0, 235, 235);     // yellow:  matched only the mirrored pose
const cv::Scalar kColCandidate (230, 170, 80);    // steel:   growing a candidate track
const cv::Scalar kColUnmatched (255, 255, 0);     // cyan:    usable background corner, unclaimed
const cv::Scalar kColInactive  (130, 130, 130);   // grey:    outside the FOV margin / not in play
const cv::Scalar kColAmbiguous (200, 140, 200);   // mauve:   too smeared to discriminate the mirror
const cv::Scalar kColOffView   (85, 80, 75);      // dim:     mapped but not in view at this pose

// Status -> colour for an out-of-field map landmark, shared by both panels so
// a landmark reads the same in the camera view and in the 3D map.
cv::Scalar landmarkStatusColour(int status)
{
    switch (status)
    {
        case SideDisambiguator::LANDMARK_ASSOCIATED:     return kColAssociated;
        case SideDisambiguator::LANDMARK_MISSED:         return kColMissed;
        case SideDisambiguator::LANDMARK_AMBIGUOUS:      return kColAmbiguous;
        case SideDisambiguator::LANDMARK_EDGE:           return kColInactive;
        case SideDisambiguator::LANDMARK_CULLED_MISSING:
        case SideDisambiguator::LANDMARK_CULLED_OUTLIER: return kColOutlier;
        default:                                         return kColOffView;
    }
}

// Compact colour legend drawn in the bottom-left corner of a panel.
struct LegendEntry
{
    cv::Scalar colour;
    std::string label;
    bool ring = false;      // ring swatch (covariance) instead of a line (trail/marker)
};

void drawPanelLegend(cv::Mat & panel, const std::vector<LegendEntry> & entries)
{
    const int rowH = 20;
    const int pad = 8;
    int maxText = 0;
    for (const LegendEntry & e : entries)
    {
        maxText = std::max(maxText, cv::getTextSize(e.label, cv::FONT_HERSHEY_SIMPLEX, 0.38, 1, nullptr).width);
    }
    const int w = pad + 30 + maxText + pad;
    const int h = pad + static_cast<int>(entries.size())*rowH;
    const cv::Point org(10, panel.rows - h - 10);

    cv::Mat backing = panel(cv::Rect(org.x, org.y, std::min(w, panel.cols - org.x), h));
    backing *= 0.25;    // darken behind the legend for legibility

    for (std::size_t i = 0; i < entries.size(); ++i)
    {
        const LegendEntry & e = entries[i];
        const int cy = org.y + pad/2 + static_cast<int>(i)*rowH + rowH/2;
        if (e.ring)
        {
            cv::ellipse(panel, cv::Point(org.x + pad + 10, cy), cv::Size(9, 5), 0, 0, 360, e.colour, 1, cv::LINE_AA);
        }
        else
        {
            cv::line(panel, cv::Point(org.x + pad, cy), cv::Point(org.x + pad + 20, cy), e.colour, 2, cv::LINE_AA);
        }
        cv::putText(panel, e.label, cv::Point(org.x + pad + 30, cy + 4),
                    cv::FONT_HERSHEY_SIMPLEX, 0.38, cv::Scalar(230, 230, 230), 1, cv::LINE_AA);
    }
}

// 2x2 covariance -> ellipse axes (half-lengths in metres) and angle (deg, CCW in field frame).
void covEllipse(const Eigen::Matrix2d & P, double nSigma, double & ax, double & ay, double & angleDeg)
{
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> es(P);
    Eigen::Vector2d val = es.eigenvalues().cwiseMax(0.0);
    Eigen::Vector2d major = es.eigenvectors().col(1);
    ax = nSigma*std::sqrt(val(1));
    ay = nSigma*std::sqrt(val(0));
    angleDeg = std::atan2(major.y(), major.x())*180.0/M_PI;
}

} // namespace

LocalisationViewer::LocalisationViewer(const FieldMap & map, const FisheyeLens & lens,
                                       const std::filesystem::path & videoPath)
    : map_(map)
    , lens_(lens)
    , videoPath_(videoPath)
{}

// ===========================================================================
// Camera panel
// ===========================================================================
cv::Mat LocalisationViewer::renderCameraPanel(const ViewerFrame & f, const cv::Mat & rawFrame, int panelH) const
{
    const double scale = panelH/lens_.height;
    const int panelW = static_cast<int>(std::lround(lens_.width*scale));

    cv::Mat panel;
    if (rawFrame.empty())
    {
        panel = cv::Mat(panelH, panelW, CV_8UC3, cv::Scalar(20, 20, 20));
        hudText(panel, "no video frame", cv::Point(panelW/2 - 60, panelH/2));
    }
    else
    {
        cv::resize(rawFrame, panel, cv::Size(panelW, panelH), 0, 0, cv::INTER_AREA);
    }

    const Eigen::Matrix3d Rcf = f.Tfc.rotationMatrix.transpose();
    const Eigen::Vector3d rCFf = f.Tfc.translationVector;

    auto safe = [&](const cv::Point2d & p) { return std::abs(p.x) < 5*panelW && std::abs(p.y) < 5*panelH; };
    auto toPanel = [&](const Eigen::Vector2d & px) { return cv::Point2d(px.x()*scale, px.y()*scale); };
    auto projectPanel = [&](const Eigen::Vector3d & ray, cv::Point2d & out) -> bool {
        if (!FisheyeLens::inFrontOfCamera(ray)) return false;
        out = toPanel(lens_.project(ray));
        return safe(out);
    };
    // Ray to a field-frame point from the estimated camera pose.
    auto rayTo = [&](const Eigen::Vector3d & rPFf) { return (Rcf*(rPFf - rCFf)).normalized(); };

    // ---- horizon (great circle of camera rays perpendicular to field-up) ----
    {
        const Eigen::Vector3d nUp = (Rcf*Eigen::Vector3d::UnitZ()).normalized();  // field-up in {c}
        Eigen::Vector3d u = nUp.cross(Eigen::Vector3d::UnitX());
        if (u.norm() < 1e-6) u = nUp.cross(Eigen::Vector3d::UnitY());
        u.normalize();
        const Eigen::Vector3d v = nUp.cross(u).normalized();
        cv::Point2d prev; bool havePrev = false;
        for (int i = 0; i <= 180; ++i)
        {
            const double a = 2.0*M_PI*i/180.0;
            const Eigen::Vector3d d = std::cos(a)*u + std::sin(a)*v;
            cv::Point2d p;
            if (projectPanel(d, p))
            {
                if (havePrev) cv::line(panel, prev, p, cv::Scalar(120, 120, 120), 1, cv::LINE_AA);
                prev = p; havePrev = true;
            }
            else havePrev = false;
        }
    }

    // ---- faint predicted map landmarks at the estimate ----
    for (int t = 0; t < 4; ++t)
    {
        const auto type = static_cast<LandmarkType>(t);
        for (const Eigen::Vector3d & lm : map_.landmarks(type))
        {
            const Eigen::Vector3d ray = rayTo(lm);
            cv::Point2d p;
            if (projectPanel(ray, p) && p.x >= 0 && p.x < panelW && p.y >= 0 && p.y < panelH)
            {
                cv::drawMarker(panel, p, cv::Scalar(110, 110, 110), cv::MARKER_TILTED_CROSS, 7, 1, cv::LINE_AA);
            }
        }
    }

    // ---- visibility margin: inside this border a map landmark counts as
    // "predicted visible" and so is allowed to score and to accrue misses ----
    if (f.oofVisibleMargin > 0.0 && !f.oofLandmarkProj.empty())
    {
        const int m = static_cast<int>(std::lround(f.oofVisibleMargin*scale));
        cv::rectangle(panel, cv::Point(m, m), cv::Point(panelW - m, panelH - m),
                      cv::Scalar(70, 70, 70), 1, cv::LINE_AA);
    }

    // ---- out-of-field map landmarks predicted at the estimate ----
    // Squares (vs circles for measured corners), drawn least-to-most important so
    // the interesting statuses land on top.
    for (int pass = SideDisambiguator::LANDMARK_EDGE; pass <= SideDisambiguator::LANDMARK_CULLED_OUTLIER; ++pass)
    {
        for (const OofLandmarkProjView & lp : f.oofLandmarkProj)
        {
            if (lp.status != pass) continue;
            const cv::Point2d p = toPanel(lp.px);
            const cv::Scalar col = landmarkStatusColour(lp.status);
            const int r = lp.status == SideDisambiguator::LANDMARK_EDGE
                       || lp.status == SideDisambiguator::LANDMARK_AMBIGUOUS ? 3
                        : lp.status >= SideDisambiguator::LANDMARK_CULLED_MISSING ? 5 : 4;
            cv::rectangle(panel, p + cv::Point2d(-r, -r), p + cv::Point2d(r, r), col, 1, cv::LINE_AA);
            if (lp.far)
            {
                // Bearing-only: no depth yet, so the prediction is a direction.
                cv::drawMarker(panel, p, col, cv::MARKER_DIAMOND, 2*r, 1, cv::LINE_AA);
            }
            if (lp.status == SideDisambiguator::LANDMARK_ASSOCIATED)
            {
                // Link to the corner that claimed it. This is an INNOVATION, not a
                // frame-to-frame motion: the box is where the stored landmark
                // reprojects through the current pose estimate, the circle is where
                // the corner was actually detected. It therefore mixes landmark
                // position error (large -- most landmarks have no triangulated
                // depth), pose error and corner noise, and is not ground truth.
                cv::line(panel, p, toPanel(lp.matchPx), kColAssociated, 1, cv::LINE_AA);
            }
            else if (lp.status == SideDisambiguator::LANDMARK_CULLED_MISSING
                  || lp.status == SideDisambiguator::LANDMARK_CULLED_OUTLIER)
            {
                cv::drawMarker(panel, p, col, cv::MARKER_TILTED_CROSS, 2*r, 1, cv::LINE_AA);
            }
        }
    }

    // ---- out-of-field corner features (side disambiguation) ----
    for (int pass = SideDisambiguator::FEATURE_ON_CARPET; pass <= SideDisambiguator::FEATURE_ASSOCIATED; ++pass)
    {
        for (const OofFeatureView & of : f.oofFeatures)
        {
            if (of.status != pass) continue;
            const cv::Point2d p = toPanel(of.px);
            switch (of.status)
            {
                case SideDisambiguator::FEATURE_ON_CARPET:  // masked out, drawn faintly to verify the mask
                    cv::circle(panel, p, 1, cv::Scalar(90, 90, 90), cv::FILLED, cv::LINE_AA);
                    break;
                case SideDisambiguator::FEATURE_UNMATCHED:
                    cv::circle(panel, p, 2, kColUnmatched, 1, cv::LINE_AA);
                    break;
                case SideDisambiguator::FEATURE_CANDIDATE:
                    cv::circle(panel, p, 2, kColCandidate, 1, cv::LINE_AA);
                    break;
                case SideDisambiguator::FEATURE_OUTLIER:
                    cv::drawMarker(panel, p, kColOutlier, cv::MARKER_TILTED_CROSS, 7, 1, cv::LINE_AA);
                    break;
                case SideDisambiguator::FEATURE_MIRROR:
                    cv::circle(panel, p, 3, kColMirror, 1, cv::LINE_AA);
                    cv::circle(panel, p, 1, kColMirror, cv::FILLED, cv::LINE_AA);
                    break;
                default:    // FEATURE_ASSOCIATED
                    // Small: the landmark box and the link line drawn above
                    // already carry this match, so the dot only has to mark
                    // where the corner actually was.
                    cv::circle(panel, p, 2, kColAssociated, cv::FILLED, cv::LINE_AA);
                    break;
            }
        }
    }

    // ---- YOLO detections (curved fisheye boxes) ----
    // The box keeps its class colour (that is how you read what was detected);
    // the association outcome is carried by the measurement dot and the label,
    // which share the palette with the out-of-field landmarks above.
    for (const DetectionView & d : f.detections)
    {
        const cv::Scalar col = classColour(d.name);
        const bool associated = d.status == DetectionStatus::ASSOCIATED;
        const int thick = associated ? 2 : 1;
        // Curved edges: interpolate rays between adjacent corners (TL-TR-BR-BL)
        // and project each sub-point, so the box hugs the fisheye distortion.
        cv::Point2d prev; bool havePrev = false;
        for (int e = 0; e < 4; ++e)
        {
            const Eigen::Vector3d ra = d.corners.col(e).normalized();
            const Eigen::Vector3d rb = d.corners.col((e + 1) % 4).normalized();
            for (int s = 0; s <= 6; ++s)
            {
                const double tt = s/6.0;
                const Eigen::Vector3d r = ((1.0 - tt)*ra + tt*rb).normalized();
                cv::Point2d p;
                if (!projectPanel(r, p)) { havePrev = false; continue; }
                if (havePrev) cv::line(panel, prev, p, col, thick, cv::LINE_AA);
                prev = p; havePrev = true;
            }
        }
        // Label at the top-left corner, suffixed and tinted by the outcome.
        cv::Scalar statusCol = kColInactive;
        std::string suffix;
        switch (d.status)
        {
            case DetectionStatus::ASSOCIATED:
                statusCol = kColAssociated;
                break;
            case DetectionStatus::UNASSOCIATED:
                statusCol = kColOutlier;
                suffix = " (unassociated)";
                break;
            case DetectionStatus::LOW_CONFIDENCE:
                statusCol = kColMissed;
                suffix = " (low conf)";
                break;
            default:    // UNUSED_CLASS
                suffix = " (not a landmark)";
                break;
        }
        cv::Point2d tl;
        if (projectPanel(d.corners.col(0).normalized(), tl))
        {
            const std::string lbl = std::format("{} {:.0f}%{}", d.name, d.confidence*100.0, suffix);
            hudText(panel, lbl, cv::Point(static_cast<int>(tl.x), static_cast<int>(tl.y) - 4), 0.36,
                    d.status == DetectionStatus::UNUSED_CLASS ? col : statusCol);
        }
        // Measurement point (what the estimator would use), coloured by outcome.
        Eigen::Vector3d mray; cv::Point2d mp;
        if (d.status != DetectionStatus::UNUSED_CLASS && measurementRay(d, mray) && projectPanel(mray, mp))
        {
            if (associated) cv::circle(panel, mp, 4, statusCol, cv::FILLED, cv::LINE_AA);
            else            cv::circle(panel, mp, 4, statusCol, 1, cv::LINE_AA);
        }
    }

    // ---- associations: measured point -> predicted landmark, with residual ----
    for (const AssociationView & a : f.associations)
    {
        cv::Point2d mp, pp;
        const bool okM = projectPanel(a.measRay, mp);
        const Eigen::Vector3d predRay = rayTo(a.landmark);
        const bool okP = projectPanel(predRay, pp);
        if (okP)
        {
            cv::drawMarker(panel, pp, cv::Scalar(255, 255, 255), cv::MARKER_CROSS, 9, 1, cv::LINE_AA);
            cv::circle(panel, pp, 5, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
        }
        if (okM && okP)
        {
            const double resid = std::acos(std::clamp(a.measRay.dot(predRay), -1.0, 1.0));
            // green (aligned) -> red (large residual), saturating at ~12 deg
            const double u = std::clamp(resid/(12.0*M_PI/180.0), 0.0, 1.0);
            cv::Scalar col(0, static_cast<int>(255*(1 - u)), static_cast<int>(255*u));
            cv::line(panel, mp, pp, col, 1, cv::LINE_AA);
        }
    }

    // ---- field-line points ----
    for (Eigen::Index j = 0; j < f.lineRays.cols(); ++j)
    {
        cv::Point2d p;
        if (projectPanel(f.lineRays.col(j), p) && p.x >= 0 && p.x < panelW && p.y >= 0 && p.y < panelH)
        {
            cv::circle(panel, p, 2, cv::Scalar(255, 255, 0), cv::FILLED, cv::LINE_AA);
        }
    }

    // ---- camera HUD ----
    auto countFeatures = [&](int status) {
        return std::count_if(f.oofFeatures.begin(), f.oofFeatures.end(),
                             [&](const OofFeatureView & of) { return of.status == status; });
    };
    auto countLandmarks = [&](int status) {
        return std::count_if(f.oofLandmarkProj.begin(), f.oofLandmarkProj.end(),
                             [&](const OofLandmarkProjView & lp) { return lp.status == status; });
    };

    hudText(panel, std::format("camera  t={:.2f}s  video frame {}", f.t, f.videoFrame), cv::Point(10, 22), 0.45);
    {
        const auto nUnassoc = std::count_if(f.detections.begin(), f.detections.end(),
            [](const DetectionView & d) { return d.status == DetectionStatus::UNASSOCIATED; });
        hudText(panel, std::format("YOLO {} detections: {} associated, {} unassociated (of {} usable)",
                                   f.detections.size(), f.nAssoc, nUnassoc, f.nCand),
                cv::Point(10, 44), 0.42, cv::Scalar(200, 255, 200));
    }
    int hudY = 66;
    if (!f.oofFeatures.empty())
    {
        hudText(panel, std::format("oof corners {}: {} assoc, {} mirror, {} rejected, {} track, {} free (of {} total)",
                                   f.oofFeatures.size() - countFeatures(SideDisambiguator::FEATURE_ON_CARPET),
                                   countFeatures(SideDisambiguator::FEATURE_ASSOCIATED),
                                   countFeatures(SideDisambiguator::FEATURE_MIRROR),
                                   countFeatures(SideDisambiguator::FEATURE_OUTLIER),
                                   countFeatures(SideDisambiguator::FEATURE_CANDIDATE),
                                   countFeatures(SideDisambiguator::FEATURE_UNMATCHED),
                                   f.oofFeatures.size()),
                cv::Point(10, hudY), 0.42, cv::Scalar(255, 255, 180));
        hudY += 22;
    }
    if (!f.oofLandmarkProj.empty())
    {
        hudText(panel, std::format("oof landmarks in image {}: {} assoc, {} missed, {} ambiguous, {} at edge",
                                   f.oofLandmarkProj.size(),
                                   countLandmarks(SideDisambiguator::LANDMARK_ASSOCIATED),
                                   countLandmarks(SideDisambiguator::LANDMARK_MISSED),
                                   countLandmarks(SideDisambiguator::LANDMARK_AMBIGUOUS),
                                   countLandmarks(SideDisambiguator::LANDMARK_EDGE)),
                cv::Point(10, hudY), 0.42, cv::Scalar(255, 255, 180));
        hudY += 22;
    }
    if (f.hasSide)
    {
        hudText(panel, std::format("side llr {:+.1f} nats  oof map {} lm{}",
                                   f.sideLlr, f.nOofLandmarks, f.sideFrozen ? "  [map frozen]" : ""),
                cv::Point(10, hudY), 0.42, cv::Scalar(255, 220, 255));
    }

    // ---- colour key ----
    // Only the statuses actually present this frame, so the key stays short
    // enough to sit over the image without burying it.
    if (showLegend_)
    {
        std::vector<LegendEntry> legend;
        auto haveDetection = [&](DetectionStatus s) {
            return std::any_of(f.detections.begin(), f.detections.end(),
                               [&](const DetectionView & d) { return d.status == s; });
        };
        if (haveDetection(DetectionStatus::ASSOCIATED))
            legend.push_back({kColAssociated, "YOLO: associated to a map landmark"});
        if (haveDetection(DetectionStatus::UNASSOCIATED))
            legend.push_back({kColOutlier, "YOLO: usable but unassociated (gated out)"});
        if (haveDetection(DetectionStatus::LOW_CONFIDENCE))
            legend.push_back({kColMissed, "YOLO: below the confidence threshold"});
        if (haveDetection(DetectionStatus::UNUSED_CLASS))
            legend.push_back({kColInactive, "YOLO: not a mapped landmark class"});

        if (countFeatures(SideDisambiguator::FEATURE_ASSOCIATED) > 0)
            legend.push_back({kColAssociated, "corner: associated (o)"});
        if (countFeatures(SideDisambiguator::FEATURE_MIRROR) > 0)
            legend.push_back({kColMirror, "corner: matches the MIRROR pose only"});
        if (countFeatures(SideDisambiguator::FEATURE_OUTLIER) > 0)
            legend.push_back({kColOutlier, "corner: gated but rejected as an outlier (x)"});
        if (countFeatures(SideDisambiguator::FEATURE_CANDIDATE) > 0)
            legend.push_back({kColCandidate, "corner: growing a candidate track"});
        if (countFeatures(SideDisambiguator::FEATURE_UNMATCHED) > 0)
            legend.push_back({kColUnmatched, "corner: out-of-field, unclaimed"});

        // Map landmarks are the [] glyphs; the thin rectangle is the border
        // inside which one counts as predicted-visible ("in FOV").
        if (countLandmarks(SideDisambiguator::LANDMARK_ASSOCIATED) > 0)
            legend.push_back({kColAssociated, "landmark []: associated; line = innovation ([] predicted -> o detected)"});
        if (countLandmarks(SideDisambiguator::LANDMARK_MISSED) > 0)
            legend.push_back({kColMissed, "landmark []: in FOV, nothing matched it"});
        if (countLandmarks(SideDisambiguator::LANDMARK_AMBIGUOUS) > 0)
            legend.push_back({kColAmbiguous, "landmark []: too smeared to discriminate"});
        if (countLandmarks(SideDisambiguator::LANDMARK_EDGE) > 0)
            legend.push_back({kColInactive, "landmark []: outside the FOV margin (thin rect)"});
        if (countLandmarks(SideDisambiguator::LANDMARK_CULLED_OUTLIER) > 0
            || countLandmarks(SideDisambiguator::LANDMARK_CULLED_MISSING) > 0)
            legend.push_back({kColOutlier, "landmark []: culled this frame as an outlier (x)"});
        if (!legend.empty()) drawPanelLegend(panel, legend);
    }
    return panel;
}

// ===========================================================================
// Top-down panel
// ===========================================================================
cv::Mat LocalisationViewer::renderTopDownPanel(const std::vector<ViewerFrame> & frames, std::size_t idx, int panelH) const
{
    const FieldDimensions & dims = map_.dims;
    const double border = dims.borderStripMinWidth + 0.3;
    const double Lx = dims.fieldLength + 2*border;
    const double Ly = dims.fieldWidth + 2*border;
    const double ppm = panelH/Ly;
    const int panelW = static_cast<int>(std::lround(Lx*ppm));

    cv::Mat panel(panelH, panelW, CV_8UC3, cv::Scalar(45, 45, 45));
    const int fieldPixW = static_cast<int>(dims.fieldLength*ppm);
    const int fieldPixH = static_cast<int>(dims.fieldWidth*ppm);
    cv::rectangle(panel, cv::Point(panelW/2 - fieldPixW/2, panelH/2 - fieldPixH/2),
                  cv::Point(panelW/2 + fieldPixW/2, panelH/2 + fieldPixH/2), cv::Scalar(30, 100, 30), cv::FILLED);

    auto toPix = [&](const Eigen::Vector2d & p) {
        return cv::Point2d(panelW/2.0 + p.x()*ppm, panelH/2.0 - p.y()*ppm);
    };

    // ---- field lines ----
    const cv::Scalar white(255, 255, 255);
    const int lw = std::max(1, static_cast<int>(dims.lineWidth*ppm));
    for (const auto & seg : map_.lineSegments())
        cv::line(panel, toPix(seg.a), toPix(seg.b), white, lw, cv::LINE_AA);
    for (const auto & c : map_.lineCircles())
        cv::circle(panel, toPix(c.centre), static_cast<int>(c.radius*ppm), white, lw, cv::LINE_AA);
    for (const Eigen::Vector3d & gp : map_.landmarks(LandmarkType::GOAL_POST))
        cv::circle(panel, toPix(gp.head<2>()), std::max(3, static_cast<int>(dims.goalpostWidth*0.5*ppm)),
                   cv::Scalar(0, 215, 255), cv::FILLED, cv::LINE_AA);

    if (idx >= frames.size()) return panel;
    const ViewerFrame & cur = frames[idx];

    // ---- trajectory trails (truth beneath the estimate) ----
    for (std::size_t i = 0; i <= idx; ++i)
    {
        if (frames[i].hasTruth)
            cv::circle(panel, toPix(frames[i].truthPos.head<2>()), 1, cv::Scalar(80, 220, 80), cv::FILLED, cv::LINE_AA);
    }
    for (std::size_t i = 0; i <= idx; ++i)
    {
        const double frac = idx > 0 ? static_cast<double>(i)/idx : 0.0;
        cv::circle(panel, toPix(frames[i].estPos), 1,
                   cv::Scalar(255*(1 - frac), 128, 255*frac), cv::FILLED, cv::LINE_AA);
    }

    // ---- camera field-of-view wedge (from the estimated camera pose) ----
    {
        const Eigen::Vector3d axis = cur.Tfc.rotationMatrix*Eigen::Vector3d::UnitX();
        const double az = std::atan2(axis.y(), axis.x());
        const Eigen::Vector2d o = cur.estPos;
        for (double s : {-1.0, 1.0})
        {
            const Eigen::Vector2d dir(std::cos(az + s*kHalfFov), std::sin(az + s*kHalfFov));
            cv::line(panel, toPix(o), toPix(Eigen::Vector2d(o + 3.0*dir)), cv::Scalar(70, 130, 70), 1, cv::LINE_AA);
        }
    }

    // ---- on-ground landmark residuals (measured ground point -> map landmark) ----
    for (const AssociationView & a : cur.associations)
    {
        const Eigen::Vector3d d = cur.Tfc.rotationMatrix*a.measRay;
        if (d.z() < -1e-3)
        {
            const double lambda = -cur.Tfc.translationVector.z()/d.z();
            const Eigen::Vector2d g = cur.Tfc.translationVector.head<2>() + lambda*d.head<2>();
            cv::line(panel, toPix(g), toPix(a.landmark.head<2>()), cv::Scalar(0, 200, 255), 1, cv::LINE_AA);
            cv::circle(panel, toPix(g), 3, cv::Scalar(0, 200, 255), cv::FILLED, cv::LINE_AA);
        }
    }

    // ---- robot marker helper ----
    auto drawRobot = [&](const Eigen::Vector2d & pos, double yaw, const cv::Scalar & col, int r, int thick) {
        cv::circle(panel, toPix(pos), r, col, thick, cv::LINE_AA);
        const Eigen::Vector2d head(pos.x() + 0.45*std::cos(yaw), pos.y() + 0.45*std::sin(yaw));
        cv::line(panel, toPix(pos), toPix(head), col, thick, cv::LINE_AA);
    };
    auto drawEllipse = [&](const Eigen::Vector2d & pos, const Eigen::Matrix2d & P, const cv::Scalar & col, int thick) {
        double ax, ay, ang;
        covEllipse(P, 2.0, ax, ay, ang);
        cv::ellipse(panel, toPix(pos), cv::Size(std::max(1, int(ax*ppm)), std::max(1, int(ay*ppm))),
                    -ang, 0, 360, col, thick, cv::LINE_AA);
    };

    // ---- NUbots baseline ----
    if (cur.hasBaseline)
        drawRobot(cur.basePos, cur.baseYaw, cv::Scalar(180, 180, 180), std::max(4, int(0.15*ppm)), 2);

    // ---- mocap ground truth (green, cross-hatched so it reads as "reference") ----
    if (cur.hasTruth)
    {
        drawRobot(cur.truthPos.head<2>(), cur.truthYaw, cv::Scalar(80, 220, 80), std::max(5, int(0.16*ppm)), 2);
        cv::drawMarker(panel, toPix(cur.truthPos.head<2>()), cv::Scalar(80, 220, 80),
                       cv::MARKER_TILTED_CROSS, 9, 1, cv::LINE_AA);
    }

    // ---- hypotheses (dim), then the leading estimate (bright) ----
    // Only drawn when the bank is active (>1 component); a lone hypothesis is
    // already shown by the estimate marker below.
    if (cur.hypotheses.size() > 1)
    {
        for (const HypothesisView & h : cur.hypotheses)
        {
            const int shade = static_cast<int>(90 + 120*std::clamp(h.weight, 0.0, 1.0));
            const cv::Scalar col(shade, shade/2, 255 - shade);
            drawEllipse(h.pos, h.cov, col, 1);
            drawRobot(h.pos, h.yaw, col, std::max(3, int(0.12*ppm)), 1);
            hudText(panel, std::format("{:.0f}%", h.weight*100.0),
                    toPix(h.pos) + cv::Point2d(6, -6), 0.34, col);
        }
    }
    drawEllipse(cur.estPos, cur.estCov, cv::Scalar(255, 0, 200), 2);
    drawRobot(cur.estPos, cur.estYaw, cv::Scalar(0, 128, 255), std::max(5, int(0.16*ppm)), 2);

    // ---- top-down HUD ----
    hudText(panel, "top-down field   [3] 3D view", cv::Point(10, 22), 0.45);
    int hudY = 44;
    if (cur.hasTruth)
    {
        const double eOurs = (cur.estPos - cur.truthPos.head<2>()).norm();
        const double eYaw = std::remainder(cur.estYaw - cur.truthYaw, 2.0*M_PI);
        std::string s = std::format("err vs truth: ours {:.2f} m {:+.1f} deg", eOurs, eYaw*180.0/M_PI);
        if (cur.hasBaseline)
        {
            const double bOurs = (cur.basePos - cur.truthPos.head<2>()).norm();
            const double bYaw = std::remainder(cur.baseYaw - cur.truthYaw, 2.0*M_PI);
            s += std::format("  |  baseline {:.2f} m {:+.1f} deg", bOurs, bYaw*180.0/M_PI);
        }
        hudText(panel, s, cv::Point(10, hudY), 0.42, cv::Scalar(180, 255, 180));
        hudY += 22;
    }
    if (cur.hasError)
    {
        hudText(panel, std::format("err vs baseline: {:.2f} m, {:.1f} deg", cur.errXY, cur.errYaw*180.0/M_PI),
                cv::Point(10, hudY), 0.42, cv::Scalar(200, 255, 200));
        hudY += 22;
    }
    if (cur.hypotheses.size() > 1)
        hudText(panel, std::format("{} hypotheses", cur.hypotheses.size()),
                cv::Point(10, hudY), 0.42, cv::Scalar(180, 220, 255));

    // ---- legend ----
    if (showLegend_)
    {
        std::vector<LegendEntry> legend;
        legend.push_back({cv::Scalar(0, 128, 255), "our estimate (trail: blue start -> orange now)"});
        if (cur.hasTruth)
            legend.push_back({cv::Scalar(80, 220, 80), "mocap ground truth"});
        if (cur.hasBaseline)
            legend.push_back({cv::Scalar(180, 180, 180), "NUbots baseline"});
        legend.push_back({cv::Scalar(255, 0, 200), "2-sigma position uncertainty", true});
        if (!cur.associations.empty())
            legend.push_back({cv::Scalar(0, 200, 255), "landmark residual (ground point -> map)"});
        drawPanelLegend(panel, legend);
    }
    return panel;
}

// ===========================================================================
// 3D panel
// ===========================================================================
// Perspective wireframe of the scene in {f} under an orbitable camera. Height is
// a first-class axis here (the top-down view hides it): the estimate trail is
// drawn at its estimated z, the mocap truth at its measured z, and everything
// carrying a covariance -- the pose estimate, every hypothesis and every
// triangulated out-of-field landmark -- gets a 3-sigma ellipsoid (its three
// principal rings).
cv::Mat LocalisationViewer::render3DPanel(const std::vector<ViewerFrame> & frames, std::size_t idx, int panelH) const
{
    const FieldDimensions & dims = map_.dims;
    const double border = dims.borderStripMinWidth + 0.3;
    // Same width as the top-down panel so toggling panes keeps the window size.
    const double ppm = panelH/(dims.fieldWidth + 2*border);
    const int panelW = static_cast<int>(std::lround((dims.fieldLength + 2*border)*ppm));
    cv::Mat panel(panelH, panelW, CV_8UC3, cv::Scalar(26, 26, 30));

    // ---- orbit camera ----
    const Eigen::Vector3d target(0.0, 0.0, 0.3);
    const Eigen::Vector3d eye = target + orbitDist_*Eigen::Vector3d(
        std::cos(orbitEl_)*std::cos(orbitAz_), std::cos(orbitEl_)*std::sin(orbitAz_), std::sin(orbitEl_));
    const Eigen::Vector3d fwd = (target - eye).normalized();
    Eigen::Vector3d right = fwd.cross(Eigen::Vector3d::UnitZ());
    if (right.norm() < 1e-6) right = Eigen::Vector3d::UnitY();
    right.normalize();
    const Eigen::Vector3d up = right.cross(fwd);
    const double focal = panelH/(2.0*std::tan(24.0*M_PI/180.0));   // ~48 deg vertical FOV

    auto project = [&](const Eigen::Vector3d & p, cv::Point2d & out) -> bool {
        const Eigen::Vector3d d = p - eye;
        const double zc = d.dot(fwd);
        if (zc < 0.2) return false;
        out.x = panelW/2.0 + focal*d.dot(right)/zc;
        out.y = panelH/2.0 - focal*d.dot(up)/zc;
        return std::abs(out.x) < 4.0*panelW && std::abs(out.y) < 4.0*panelH;
    };
    auto line3 = [&](const Eigen::Vector3d & a, const Eigen::Vector3d & b,
                     const cv::Scalar & col, int thick = 1) {
        cv::Point2d pa, pb;
        if (project(a, pa) && project(b, pb)) cv::line(panel, pa, pb, col, thick, cv::LINE_AA);
    };
    auto polyline3 = [&](auto && pointAt, int n, const cv::Scalar & col, int thick = 1, bool closed = true) {
        cv::Point2d prev; bool havePrev = false;
        for (int i = 0; i <= (closed ? n : n - 1); ++i)
        {
            cv::Point2d p;
            if (project(pointAt(i % n), p))
            {
                if (havePrev) cv::line(panel, prev, p, col, thick, cv::LINE_AA);
                prev = p; havePrev = true;
            }
            else havePrev = false;
        }
    };
    // 3-sigma ellipsoid wireframe: the three principal rings of P. Ellipsoids
    // larger than maxExtent are skipped: a barely-constrained landmark's ring
    // would span the whole scene and bury everything legible under it.
    auto ellipsoid3 = [&](const Eigen::Vector3d & c, const Eigen::Matrix3d & P,
                          const cv::Scalar & col, int thick = 1, double maxExtent = 1e9) {
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(P);
        if (es.info() != Eigen::Success) return;
        const Eigen::Vector3d s = es.eigenvalues().cwiseMax(0.0).cwiseSqrt()*3.0;
        if (s.maxCoeff() > maxExtent) return;
        const Eigen::Matrix3d V = es.eigenvectors();
        const int axPair[3][2] = {{0, 1}, {0, 2}, {1, 2}};
        for (const auto & pr : axPair)
        {
            polyline3([&](int i) -> Eigen::Vector3d {
                const double a = 2.0*M_PI*i/36.0;
                return c + std::cos(a)*s(pr[0])*V.col(pr[0]) + std::sin(a)*s(pr[1])*V.col(pr[1]);
            }, 36, col, thick);
        }
    };

    // ---- ground: carpet outline, 1 m grid, field lines, goals ----
    {
        const double hx = dims.fieldLength/2 + border;
        const double hy = dims.fieldWidth/2 + border;
        const cv::Scalar grid(45, 55, 45);
        for (double x = std::ceil(-hx); x <= hx; x += 1.0)
            line3({x, -hy, 0.0}, {x, hy, 0.0}, grid);
        for (double y = std::ceil(-hy); y <= hy; y += 1.0)
            line3({-hx, y, 0.0}, {hx, y, 0.0}, grid);
        const cv::Scalar carpet(40, 90, 40);
        const Eigen::Vector3d c0(-hx, -hy, 0), c1(hx, -hy, 0), c2(hx, hy, 0), c3(-hx, hy, 0);
        line3(c0, c1, carpet); line3(c1, c2, carpet); line3(c2, c3, carpet); line3(c3, c0, carpet);
    }
    const cv::Scalar white(235, 235, 235);
    for (const auto & seg : map_.lineSegments())
        line3({seg.a.x(), seg.a.y(), 0.0}, {seg.b.x(), seg.b.y(), 0.0}, white);
    for (const auto & c : map_.lineCircles())
        polyline3([&](int i) -> Eigen::Vector3d {
            const double a = 2.0*M_PI*i/72.0;
            return {c.centre.x() + c.radius*std::cos(a), c.centre.y() + c.radius*std::sin(a), 0.0};
        }, 72, white);
    // Goals: posts up to the (kid-size) 0.8 m crossbar.
    constexpr double kGoalHeight = 0.8;
    const cv::Scalar goalCol(0, 215, 255);
    for (int sx : {-1, 1})
    {
        const double gx = sx*dims.fieldLength/2;
        for (int sy : {-1, 1})
        {
            const double gy = sy*dims.goalWidth/2;
            line3({gx, gy, 0.0}, {gx, gy, kGoalHeight}, goalCol, 2);
        }
        line3({gx, -dims.goalWidth/2, kGoalHeight}, {gx, dims.goalWidth/2, kGoalHeight}, goalCol, 2);
    }

    if (idx >= frames.size()) return panel;
    const ViewerFrame & cur = frames[idx];

    // ---- out-of-field landmark map (background structure used for side disambiguation) ----
    // Depth is what this view exists to show, so it must not imply precision the
    // map does not have. Most landmarks are bearing-only: fitFar() parks them at
    // an assumed range along the measured bearing with a 3-sigma radial spread of
    // half that range again, so they genuinely do lie on a shell around whichever
    // camera position anchored them. Drawing a full ellipsoid for those fills the
    // screen; drawing nothing (the previous behaviour) hid the fact entirely.
    // Instead, anything whose 3-sigma ellipsoid is longer than kEllipsoidMax in
    // its dominant direction is drawn as that principal axis alone -- an "the
    // landmark is somewhere along here" bar -- and only genuinely localised
    // landmarks get the full three-ring ellipsoid.
    constexpr double kEllipsoidMax = 1.5;   ///< Longest 3-sigma semi-axis still drawn as an ellipsoid [m]
    const Eigen::Vector3d camPos = cur.Tfc.translationVector;
    std::size_t nFar = 0, nBar = 0;
    // Dim off-view landmarks first so the ones in play sit on top.
    for (int pass = 0; pass < 2; ++pass)
    {
        for (const OofLandmarkView & lm : cur.oofLandmarks)
        {
            const bool inPlay = lm.status != SideDisambiguator::LANDMARK_NOT_IN_VIEW;
            if (inPlay != (pass == 1)) continue;   // Each landmark is visited exactly once across the passes
            if (lm.far) nFar++;

            const cv::Scalar col = landmarkStatusColour(lm.status);
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(lm.cov);
            if (es.info() != Eigen::Success) continue;
            const double ext = 3.0*std::sqrt(std::max(0.0, es.eigenvalues()(2)));

            cv::Point2d p;
            if (project(lm.pos, p)) cv::circle(panel, p, inPlay ? 2 : 1, col, cv::FILLED, cv::LINE_AA);

            if (ext <= kEllipsoidMax)
            {
                ellipsoid3(lm.pos, lm.cov, col, 1);
            }
            else if (lm.status == SideDisambiguator::LANDMARK_ASSOCIATED)
            {
                // Bars only for the landmarks carrying evidence this frame. Every
                // poorly-constrained landmark drawn at once (the honest picture,
                // but hundreds of overlapping rays) buries the scene; the ones
                // actually being matched make the same point legibly, and the HUD
                // reports the aggregate.
                //
                // The bar is the dominant 3-sigma axis, clipped to a plausible
                // range from the camera: unclipped, the radial interval on a 6 m
                // landmark reaches behind the camera -- a real artefact of putting
                // a symmetric Gaussian on a strictly positive depth.
                const Eigen::Vector3d axis = es.eigenvectors().col(2)*ext;
                auto clip = [&](const Eigen::Vector3d & q) -> Eigen::Vector3d {
                    const Eigen::Vector3d rel = q - camPos;
                    const double r = rel.norm();
                    if (r < 1e-6) return q;
                    return camPos + rel*(std::clamp(r, 0.5, 25.0)/r);
                };
                line3(clip(lm.pos - axis), clip(lm.pos + axis), col, 1);
                nBar++;
            }
        }
    }

    // ---- trails at their estimated / measured heights ----
    for (std::size_t i = 0; i <= idx; ++i)
    {
        cv::Point2d p;
        if (frames[i].hasTruth && project(frames[i].truthPos, p))
            cv::circle(panel, p, 1, cv::Scalar(80, 220, 80), cv::FILLED, cv::LINE_AA);
    }
    for (std::size_t i = 0; i <= idx; ++i)
    {
        cv::Point2d p;
        if (project(frames[i].estPos3, p))
        {
            const double frac = idx > 0 ? static_cast<double>(i)/idx : 0.0;
            cv::circle(panel, p, 1, cv::Scalar(255*(1 - frac), 128, 255*frac), cv::FILLED, cv::LINE_AA);
        }
    }

    // ---- NUbots baseline (2D estimate: shown on the ground) ----
    if (cur.hasBaseline)
    {
        const Eigen::Vector3d b(cur.basePos.x(), cur.basePos.y(), 0.0);
        polyline3([&](int i) -> Eigen::Vector3d {
            const double a = 2.0*M_PI*i/24.0;
            return b + 0.15*Eigen::Vector3d(std::cos(a), std::sin(a), 0.0);
        }, 24, cv::Scalar(180, 180, 180));
        line3(b, b + 0.4*Eigen::Vector3d(std::cos(cur.baseYaw), std::sin(cur.baseYaw), 0.0),
              cv::Scalar(180, 180, 180));
    }

    // ---- hypotheses (only when the bank is live) ----
    if (cur.hypotheses.size() > 1)
    {
        for (const HypothesisView & h : cur.hypotheses)
        {
            const int shade = static_cast<int>(90 + 120*std::clamp(h.weight, 0.0, 1.0));
            ellipsoid3(h.pos3, h.cov3, cv::Scalar(shade, shade/2, 255 - shade));
        }
    }

    // ---- pose estimate: 3-sigma ellipsoid, marker, heading, drop line ----
    ellipsoid3(cur.estPos3, cur.estCov3, cv::Scalar(255, 0, 200), 1);
    {
        cv::Point2d p;
        if (project(cur.estPos3, p))
            cv::circle(panel, p, 4, cv::Scalar(0, 128, 255), cv::FILLED, cv::LINE_AA);
        const Eigen::Vector3d head = cur.estPos3
            + 0.45*Eigen::Vector3d(std::cos(cur.estYaw), std::sin(cur.estYaw), 0.0);
        line3(cur.estPos3, head, cv::Scalar(0, 128, 255), 2);
        line3(cur.estPos3, {cur.estPos3.x(), cur.estPos3.y(), 0.0}, cv::Scalar(0, 90, 180));
    }

    // ---- mocap truth: marker at its measured height, drop line, heading ----
    if (cur.hasTruth)
    {
        cv::Point2d p;
        if (project(cur.truthPos, p))
        {
            cv::circle(panel, p, 4, cv::Scalar(80, 220, 80), 1, cv::LINE_AA);
            cv::drawMarker(panel, p, cv::Scalar(80, 220, 80), cv::MARKER_TILTED_CROSS, 8, 1, cv::LINE_AA);
        }
        const Eigen::Vector3d head = cur.truthPos
            + 0.45*Eigen::Vector3d(std::cos(cur.truthYaw), std::sin(cur.truthYaw), 0.0);
        line3(cur.truthPos, head, cv::Scalar(80, 220, 80), 2);
        line3(cur.truthPos, {cur.truthPos.x(), cur.truthPos.y(), 0.0}, cv::Scalar(60, 160, 60));
    }

    // ---- camera optical axis at the estimate ----
    line3(cur.Tfc.translationVector,
          cur.Tfc.translationVector + 0.5*(cur.Tfc.rotationMatrix*Eigen::Vector3d::UnitX()),
          cv::Scalar(0, 200, 255));

    // ---- 3D HUD ----
    hudText(panel, "3D map (3-sigma ellipsoids)   [3] top-down   drag: orbit   wheel or -/=: zoom",
            cv::Point(10, 22), 0.45);
    {
        std::string s = std::format("z: est {:.2f} m", cur.estPos3.z());
        if (cur.hasTruth)
            s += std::format("   truth marker {:.2f} m (markers sit ~6 cm above the torso origin)", cur.truthPos.z());
        hudText(panel, s, cv::Point(10, 44), 0.42, cv::Scalar(180, 255, 180));
    }
    if (!cur.oofLandmarks.empty())
    {
        hudText(panel, std::format("out-of-field map: {} landmarks, {} bearing-only (position is an ASSUMED "
                                   "range, not a triangulation)", cur.oofLandmarks.size(), nFar),
                cv::Point(10, 66), 0.42, cv::Scalar(255, 255, 180));
        hudText(panel, std::format("{} associated landmarks show a bar = 3-sigma along the dominant (usually "
                                   "depth) axis; ellipsoid only where 3-sigma < {:.1f} m", nBar, kEllipsoidMax),
                cv::Point(10, 88), 0.42, cv::Scalar(255, 255, 180));
    }

    // ---- legend ----
    if (showLegend_)
    {
        std::vector<LegendEntry> legend;
        legend.push_back({cv::Scalar(0, 128, 255), "our estimate (trail: blue start -> orange now)"});
        if (cur.hasTruth)
            legend.push_back({cv::Scalar(80, 220, 80), "mocap ground truth (at marker height)"});
        if (cur.hasBaseline)
            legend.push_back({cv::Scalar(180, 180, 180), "NUbots baseline (2D, on ground)"});
        legend.push_back({cv::Scalar(255, 0, 200), "3-sigma position ellipsoid", true});
        if (!cur.oofLandmarks.empty())
        {
            // Same association key as the camera panel, so a landmark reads the
            // same colour in both views.
            auto haveStatus = [&](int s) {
                return std::any_of(cur.oofLandmarks.begin(), cur.oofLandmarks.end(),
                                   [&](const OofLandmarkView & lm) { return lm.status == s; });
            };
            if (haveStatus(SideDisambiguator::LANDMARK_ASSOCIATED))
                legend.push_back({kColAssociated, "oof landmark: associated this frame"});
            if (haveStatus(SideDisambiguator::LANDMARK_MISSED))
                legend.push_back({kColMissed, "oof landmark: in FOV, nothing matched it"});
            if (haveStatus(SideDisambiguator::LANDMARK_AMBIGUOUS))
                legend.push_back({kColAmbiguous, "oof landmark: too smeared to discriminate"});
            if (haveStatus(SideDisambiguator::LANDMARK_EDGE))
                legend.push_back({kColInactive, "oof landmark: at the FOV margin"});
            legend.push_back({kColOffView, "oof landmark: mapped, not in view here"});
        }
        legend.push_back({cv::Scalar(0, 200, 255), "camera optical axis"});
        drawPanelLegend(panel, legend);
    }
    return panel;
}

cv::Mat LocalisationViewer::renderComposite(const std::vector<ViewerFrame> & frames, std::size_t idx, const cv::Mat & rawFrame) const
{
    const ViewerFrame & f = frames[idx];
    cv::Mat cam = renderCameraPanel(f, rawFrame, kPanelH);
    cv::Mat top = show3D_ ? render3DPanel(frames, idx, kPanelH)
                          : renderTopDownPanel(frames, idx, kPanelH);

    cv::Mat body;
    cv::hconcat(cam, top, body);

    cv::Mat header(kHeaderH, body.cols, CV_8UC3, cv::Scalar(25, 25, 25));
    hudText(header, std::format("sample {}/{}   t={:.2f}s   |   keys: [space] play/step  [n]/[p] next/prev  "
                                "[Home]/[End] jump  [3] 2D/3D  [k] key  [s] save  [q] quit",
                                idx + 1, frames.size(), f.t),
            cv::Point(10, 24), 0.44);

    cv::Mat composite;
    cv::vconcat(header, body, composite);
    return composite;
}

// ===========================================================================
// Headless video export
// ===========================================================================
void LocalisationViewer::exportVideo(const std::vector<ViewerFrame> & frames,
                                     const std::filesystem::path & outPath, double fps) const
{
    if (frames.empty())
    {
        std::println("Viewer: no frames to export.");
        return;
    }

    cv::VideoCapture cap(videoPath_.string());
    const bool haveVideo = cap.isOpened();
    if (!haveVideo)
        std::println("Viewer: could not open video {} (camera panel will be blank in the export).", videoPath_.string());

    // Sequential decode: frames are processed in order, so read forward and only
    // seek when the requested frame is not the next one (matches run()).
    int lastDecoded = -2;
    auto grabFrame = [&](int videoFrame) -> cv::Mat {
        cv::Mat raw;
        if (!haveVideo || videoFrame < 0) return raw;
        if (videoFrame != lastDecoded + 1)
            cap.set(cv::CAP_PROP_POS_FRAMES, static_cast<double>(videoFrame));
        cap.read(raw);
        lastDecoded = videoFrame;
        return raw;
    };

    if (fps <= 0.0) fps = playbackFps(frames);

    cv::VideoWriter writer;
    const int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
    for (std::size_t i = 0; i < frames.size(); ++i)
    {
        cv::Mat raw = grabFrame(frames[i].videoFrame);
        cv::Mat composite = renderComposite(frames, i, raw);
        if (!writer.isOpened())
        {
            if (!writer.open(outPath.string(), fourcc, fps, composite.size()))
            {
                std::println("Viewer: failed to open VideoWriter for {}", outPath.string());
                return;
            }
        }
        writer.write(composite);
    }
    writer.release();
    std::println("Exported {} ({} frames at {:.0f} fps)", outPath.string(), frames.size(), fps);
}

// ===========================================================================
// Interactive replay
// ===========================================================================
void LocalisationViewer::run(const std::vector<ViewerFrame> & frames, int mode, const std::filesystem::path & snapshotDir)
{
    if (frames.empty())
    {
        std::println("Viewer: no frames to display.");
        return;
    }

    // Headless snapshot mode (no display required): render a spread of frames to
    // PNG and return. Handy on servers/CI and for generating demo stills.
    if (std::getenv("VIEWER_DUMP") != nullptr)
    {
        std::filesystem::path dir = snapshotDir.empty() ? std::filesystem::path("viewer_dump") : snapshotDir;
        std::filesystem::create_directories(dir);
        cv::VideoCapture dcap(videoPath_.string());
        const double fracs[] = {0.0, 0.25, 0.5, 0.75, 1.0};
        for (double fr : fracs)
        {
            const std::size_t idx = std::min(frames.size() - 1,
                static_cast<std::size_t>(std::llround(fr*(frames.size() - 1))));
            cv::Mat raw;
            if (dcap.isOpened() && frames[idx].videoFrame >= 0)
            {
                dcap.set(cv::CAP_PROP_POS_FRAMES, static_cast<double>(frames[idx].videoFrame));
                dcap.read(raw);
            }
            const bool keep3D = show3D_;
            show3D_ = false;
            cv::Mat composite = renderComposite(frames, idx, raw);
            auto p = dir / std::format("viewer_dump_{:03d}.png", static_cast<int>(std::llround(fr*100)));
            cv::imwrite(p.string(), composite);
            show3D_ = true;
            cv::Mat composite3d = renderComposite(frames, idx, raw);
            auto p3 = dir / std::format("viewer_dump_{:03d}_3d.png", static_cast<int>(std::llround(fr*100)));
            cv::imwrite(p3.string(), composite3d);
            show3D_ = keep3D;
            std::println("Dumped {} and {} (frame {}, t={:.2f}s)", p.string(), p3.string(), idx, frames[idx].t);
        }
        return;
    }

    cv::VideoCapture cap(videoPath_.string());
    const bool haveVideo = cap.isOpened();
    if (!haveVideo)
        std::println("Viewer: could not open video {} (camera panel will be blank).", videoPath_.string());

    const std::string win = "RoboCup localisation";
    cv::namedWindow(win, cv::WINDOW_AUTOSIZE);

    std::println("Viewer controls: [space] play/step  [n]/[p] next/prev  [Home]/[End] first/last  "
                 "[3] 2D/3D pane  [k] colour key  drag orbits, wheel or -/= zooms (3D)  [s] save PNG  [q] quit");

    // Mouse orbit for the 3D pane: drag rotates, wheel zooms. The callback only
    // mutates the orbit state; while the 3D pane is up the render loop polls
    // (short waitKey timeout) so the next iteration picks the new view up.
    struct MouseState
    {
        LocalisationViewer * self;
        bool dragging = false;
        cv::Point last;
    } mouse{this};
    cv::setMouseCallback(win, [](int event, int x, int y, int flags, void * userdata) {
        MouseState & m = *static_cast<MouseState *>(userdata);
        if (event == cv::EVENT_LBUTTONDOWN)
        {
            m.dragging = true;
            m.last = {x, y};
        }
        else if (event == cv::EVENT_LBUTTONUP) m.dragging = false;
        else if (event == cv::EVENT_MOUSEMOVE && m.dragging)
        {
            m.self->orbitAz_ -= (x - m.last.x)*0.008;
            m.self->orbitEl_ = std::clamp(m.self->orbitEl_ + (y - m.last.y)*0.008,
                                          5.0*M_PI/180.0, 89.0*M_PI/180.0);
            m.last = {x, y};
        }
        else if (event == cv::EVENT_MOUSEWHEEL)
        {
            m.self->orbitDist_ = std::clamp(m.self->orbitDist_*(cv::getMouseWheelDelta(flags) > 0 ? 0.9 : 1.1),
                                            2.0, 40.0);
        }
    }, &mouse);

    int lastDecoded = -2;
    auto grabFrame = [&](int videoFrame) -> cv::Mat {
        cv::Mat raw;
        if (!haveVideo || videoFrame < 0) return raw;
        if (videoFrame != lastDecoded + 1)
            cap.set(cv::CAP_PROP_POS_FRAMES, static_cast<double>(videoFrame));
        cap.read(raw);
        lastDecoded = videoFrame;
        return raw;
    };

    std::size_t idx = 0;
    bool paused = (mode == 2);   // step mode starts paused; auto-play mode runs
    int snapCount = 0;
    const int playDelay = std::max(1, static_cast<int>(1000.0/playbackFps(frames)));

    while (true)
    {
        cv::Mat raw = grabFrame(frames[idx].videoFrame);
        cv::Mat composite = renderComposite(frames, idx, raw);
        cv::imshow(win, composite);

        // While the 3D pane is up, poll instead of blocking so mouse orbiting
        // re-renders; otherwise a pause blocks until a key arrives.
        const int delay = paused ? (show3D_ ? 30 : 0) : playDelay;
        const int key = cv::waitKey(delay);
        const int k = key & 0xFF;

        if (key < 0)   // timeout
        {
            if (!paused)   // playing: advance, pause at the end
            {
                if (idx + 1 < frames.size()) ++idx;
                else paused = true;
            }
            continue;      // paused 3D poll: re-render only (orbit may have moved)
        }
        if (k == 'q' || k == 27) break;                                  // quit
        else if (k == 'n' || k == '.' || k == 83 || k == 3)             // next
            idx = std::min(idx + 1, frames.size() - 1);
        else if (k == 'p' || k == ',' || k == 81 || k == 2)             // prev
            idx = idx > 0 ? idx - 1 : 0;
        else if (k == ' ')                                               // play/pause (auto) or step (paused)
        {
            if (mode == 2) idx = std::min(idx + 1, frames.size() - 1);
            else paused = !paused;
        }
        else if (k == 'h' || k == 'g')  idx = 0;                        // first
        else if (k == 'e')              idx = frames.size() - 1;        // last
        else if (k == '3')              show3D_ = !show3D_;             // 2D/3D right pane
        else if (k == 'k')              showLegend_ = !showLegend_;     // colour key on/off
        else if (k == '-')              orbitDist_ = std::min(40.0, orbitDist_*1.15);
        else if (k == '=' || k == '+')  orbitDist_ = std::max(2.0, orbitDist_/1.15);
        else if (k == 's' && !snapshotDir.empty())                      // save composite
        {
            std::filesystem::create_directories(snapshotDir);
            auto p = snapshotDir / std::format("viewer_frame_{:04d}.png", snapCount++);
            cv::imwrite(p.string(), composite);
            std::println("Saved {}", p.string());
        }
    }
    cv::destroyWindow(win);
}
