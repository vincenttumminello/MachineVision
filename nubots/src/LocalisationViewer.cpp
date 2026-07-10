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

namespace
{

constexpr int    kPanelH   = 760;    ///< Panel height [px]
constexpr int    kHeaderH  = 36;     ///< Header strip height [px]
constexpr double kHalfFov  = 75.0*M_PI/180.0;  ///< Half field-of-view for the top-down wedge
// True capture rate. The recording was made at 10 fps; the mp4 container metadata
// says 25 fps, but frame timing everywhere else comes from the timecode file, so
// this constant only sets the auto-play speed.
constexpr double kPlaybackFps = 10.0;

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

    // ---- YOLO detections (curved fisheye boxes) ----
    for (const DetectionView & d : f.detections)
    {
        const cv::Scalar col = classColour(d.name);
        const int thick = d.used ? 2 : 1;
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
        // Label at the top-left corner.
        cv::Point2d tl;
        if (projectPanel(d.corners.col(0).normalized(), tl))
        {
            std::string lbl = std::format("{} {:.0f}%", d.name, d.confidence*100.0);
            if (!d.used) lbl += " (unused)";
            hudText(panel, lbl, cv::Point(static_cast<int>(tl.x), static_cast<int>(tl.y) - 4), 0.36, col);
        }
        // Measurement point (what the estimator actually uses).
        Eigen::Vector3d mray; cv::Point2d mp;
        if (d.used && measurementRay(d, mray) && projectPanel(mray, mp))
        {
            cv::circle(panel, mp, 3, col, cv::FILLED, cv::LINE_AA);
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
    hudText(panel, std::format("camera  t={:.2f}s  video frame {}", f.t, f.videoFrame), cv::Point(10, 22), 0.45);
    hudText(panel, std::format("detections {}  associated {}/{}", f.detections.size(), f.nAssoc, f.nCand),
            cv::Point(10, 44), 0.42, cv::Scalar(200, 255, 200));
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

    // ---- trajectory trail ----
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
    hudText(panel, "top-down field", cv::Point(10, 22), 0.45);
    if (cur.hasError)
        hudText(panel, std::format("err vs baseline: {:.2f} m, {:.1f} deg", cur.errXY, cur.errYaw*180.0/M_PI),
                cv::Point(10, 44), 0.42, cv::Scalar(200, 255, 200));
    if (cur.hypotheses.size() > 1)
        hudText(panel, std::format("{} hypotheses", cur.hypotheses.size()),
                cv::Point(10, 66), 0.42, cv::Scalar(180, 220, 255));
    return panel;
}

cv::Mat LocalisationViewer::renderComposite(const std::vector<ViewerFrame> & frames, std::size_t idx, const cv::Mat & rawFrame) const
{
    const ViewerFrame & f = frames[idx];
    cv::Mat cam = renderCameraPanel(f, rawFrame, kPanelH);
    cv::Mat top = renderTopDownPanel(frames, idx, kPanelH);

    cv::Mat body;
    cv::hconcat(cam, top, body);

    cv::Mat header(kHeaderH, body.cols, CV_8UC3, cv::Scalar(25, 25, 25));
    hudText(header, std::format("sample {}/{}   t={:.2f}s   |   keys: [space] play/step  [n]/[p] next/prev  "
                                "[Home]/[End] jump  [s] save  [q] quit",
                                idx + 1, frames.size(), f.t),
            cv::Point(10, 24), 0.44);

    cv::Mat composite;
    cv::vconcat(header, body, composite);
    return composite;
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
            cv::Mat composite = renderComposite(frames, idx, raw);
            auto p = dir / std::format("viewer_dump_{:03d}.png", static_cast<int>(std::llround(fr*100)));
            cv::imwrite(p.string(), composite);
            std::println("Dumped {} (frame {}, t={:.2f}s)", p.string(), idx, frames[idx].t);
        }
        return;
    }

    cv::VideoCapture cap(videoPath_.string());
    const bool haveVideo = cap.isOpened();
    if (!haveVideo)
        std::println("Viewer: could not open video {} (camera panel will be blank).", videoPath_.string());

    const std::string win = "RoboCup localisation";
    cv::namedWindow(win, cv::WINDOW_AUTOSIZE);

    std::println("Viewer controls: [space] play/step  [n]/[p] next/prev  [Home]/[End] first/last  [s] save PNG  [q] quit");

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

    while (true)
    {
        cv::Mat raw = grabFrame(frames[idx].videoFrame);
        cv::Mat composite = renderComposite(frames, idx, raw);
        cv::imshow(win, composite);

        const int delay = paused ? 0 : std::max(1, static_cast<int>(1000.0/kPlaybackFps));
        const int key = cv::waitKey(delay);
        const int k = key & 0xFF;

        if (key < 0)   // timeout while playing: advance, pause at the end
        {
            if (idx + 1 < frames.size()) ++idx;
            else paused = true;
            continue;
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
