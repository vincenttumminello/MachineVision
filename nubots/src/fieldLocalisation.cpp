#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <format>
#include <fstream>
#include <limits>
#include <print>
#include <string>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "FieldMap.h"
#include "fieldLocalisation.h"
#include "GaussianInfo.hpp"
#include "MeasurementFieldLandmarks.h"
#include "MeasurementFieldLines.h"
#include "MeasurementGravity.h"
#include "MeasurementKinematicHeight.h"
#include "Pose.hpp"
#include "rotation.hpp"
#include "SensorLog.h"
#include "SystemLocalisation.h"

// Wrap angle to [-pi, pi]
static double wrapAngle(double a)
{
    return std::atan2(std::sin(a), std::cos(a));
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

// Colours (BGR) shared between the plotted marks and the legend so the two
// can never drift apart.
namespace plotcolour
{
    const cv::Scalar field(255, 255, 255);      ///< Field lines
    const cv::Scalar goalPost(0, 220, 220);     ///< Goal posts (yellow)
    const cv::Scalar baseline(180, 180, 180);   ///< NUbots baseline (grey)
    const cv::Scalar estimateStart(255, 128, 0);///< Our estimate at the start of the run (blue)
    const cv::Scalar estimateEnd(0, 128, 255);  ///< Our estimate at the end of the run (orange)
    const cv::Scalar covariance(255, 0, 200);   ///< 2-sigma position ellipse (magenta)

    // Interpolate the estimate trajectory colour by time fraction [0, 1].
    inline cv::Scalar estimate(double frac)
    {
        return cv::Scalar(255*(1 - frac), 128, 255*frac);
    }
}

// Simple top-down field renderer for trajectories
class FieldPlot
{
public:
    FieldPlot(const FieldDimensions & dims, double pixelsPerMetre = 100.0)
        : dims_(dims)
        , scale_(pixelsPerMetre)
    {
        const double borderm = dims.borderStripMinWidth + 0.3;
        fieldW_ = static_cast<int>((dims.fieldLength + 2*borderm)*scale_);
        fieldH_ = static_cast<int>((dims.fieldWidth  + 2*borderm)*scale_);
        width_  = fieldW_;
        height_ = topMargin_ + fieldH_ + bottomMargin_;

        // Dark surround, with the playing field painted green in the middle band
        img_ = cv::Mat(height_, width_, CV_8UC3, cv::Scalar(45, 45, 45));
        cv::rectangle(img_, cv::Point(0, topMargin_), cv::Point(fieldW_, topMargin_ + fieldH_),
                      cv::Scalar(30, 100, 30), cv::FILLED);
        drawField();
    }

    cv::Point2i toPixel(double xf, double yf) const
    {
        // Field x to the right, field y up in the image (offset below the title band)
        return {static_cast<int>(fieldW_/2.0 + xf*scale_),
                static_cast<int>(topMargin_ + fieldH_/2.0 - yf*scale_)};
    }

    void drawField()
    {
        const cv::Scalar & white = plotcolour::field;
        const int lw = std::max(1, static_cast<int>(dims_.lineWidth*scale_));
        const double hl = dims_.fieldLength/2, hw = dims_.fieldWidth/2;

        auto rect = [&](double x0, double y0, double x1, double y1)
        {
            cv::rectangle(img_, toPixel(x0, y0), toPixel(x1, y1), white, lw);
        };
        rect(-hl, -hw, hl, hw);                                                     // Boundary
        cv::line(img_, toPixel(0, -hw), toPixel(0, hw), white, lw);                 // Halfway line
        cv::circle(img_, toPixel(0, 0), static_cast<int>(dims_.centreCircleDiameter/2*scale_), white, lw);
        for (int s : {-1, 1})
        {
            rect(s*hl, -dims_.goalAreaWidth/2, s*(hl - dims_.goalAreaLength), dims_.goalAreaWidth/2);
            rect(s*hl, -dims_.penaltyAreaWidth/2, s*(hl - dims_.penaltyAreaLength), dims_.penaltyAreaWidth/2);
            cv::drawMarker(img_, toPixel(s*(hl - dims_.penaltyMarkDistance), 0), white, cv::MARKER_CROSS, 8, lw);
            for (int t : {-1, 1})
            {
                cv::circle(img_, toPixel(s*hl, t*dims_.goalWidth/2), std::max(2, static_cast<int>(dims_.goalpostWidth/2*scale_)), plotcolour::goalPost, -1);
            }
        }
    }

    /// @brief Draw the title (and optional subtitle) in the top band.
    void drawTitle(const std::string & title, const std::string & subtitle = "")
    {
        cv::putText(img_, title, cv::Point(14, 27), cv::FONT_HERSHEY_SIMPLEX, 0.62,
                    cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
        if (!subtitle.empty())
        {
            cv::putText(img_, subtitle, cv::Point(14, topMargin_ - 8), cv::FONT_HERSHEY_SIMPLEX, 0.40,
                        cv::Scalar(190, 190, 190), 1, cv::LINE_AA);
        }
    }

    /// @brief Draw the legend explaining every plotted mark in the bottom band.
    void drawLegend()
    {
        const int y0 = topMargin_ + fieldH_;
        const int col1 = 16;
        const int col2 = fieldW_/2 + 8;
        const int rowH = 30;
        const int textDx = 26;

        auto label = [&](int x, int row, const std::string & text)
        {
            cv::putText(img_, text, cv::Point(x + textDx, y0 + row*rowH + 5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.44, cv::Scalar(230, 230, 230), 1, cv::LINE_AA);
        };
        auto dot = [&](int x, int row, const cv::Scalar & colour)
        {
            cv::circle(img_, cv::Point(x + 8, y0 + row*rowH), 5, colour, cv::FILLED, cv::LINE_AA);
        };

        // NUbots baseline (grey dot)
        dot(col1, 1, plotcolour::baseline);
        label(col1, 1, "NUbots baseline pose (localisation.Field)");

        // Our estimate: blue->orange gradient swatch
        {
            const int cx = col1 + 8, cy = y0 + 2*rowH;
            for (int dx = -12; dx <= 12; ++dx)
            {
                double f = (dx + 12)/24.0;
                cv::line(img_, cv::Point(cx + dx, cy - 4), cv::Point(cx + dx, cy + 4), plotcolour::estimate(f), 1);
            }
        }
        label(col1, 2, "Our estimate (blue = start -> orange = end)");

        // 2-sigma covariance ring (magenta)
        cv::ellipse(img_, cv::Point(col2 + 8, y0 + rowH), cv::Size(9, 6), 0, 0, 360, plotcolour::covariance, 1, cv::LINE_AA);
        label(col2, 1, "2-sigma position uncertainty");

        // Goal posts (yellow) and field lines (white line)
        dot(col2, 2, plotcolour::goalPost);
        cv::line(img_, cv::Point(col2 + 2, y0 + 2*rowH + 12), cv::Point(col2 + 14, y0 + 2*rowH + 12), plotcolour::field, 1, cv::LINE_AA);
        label(col2, 2, "Goal posts / field lines (map)");
    }

    void drawPoint(double xf, double yf, const cv::Scalar & colour, int radius = 2)
    {
        cv::circle(img_, toPixel(xf, yf), radius, colour, -1, cv::LINE_AA);
    }

    void drawCovarianceEllipse(double xf, double yf, const Eigen::Matrix2d & P, const cv::Scalar & colour, double nSigma = 2.0)
    {
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> es(P);
        if (es.info() != Eigen::Success) return;
        const Eigen::Vector2d val = es.eigenvalues().cwiseMax(0.0);
        const Eigen::Vector2d v = es.eigenvectors().col(1);         // Major axis
        double angleDeg = std::atan2(-v.y(), v.x())*180.0/M_PI;     // Image y is flipped
        cv::Size axes(std::max(1, static_cast<int>(nSigma*std::sqrt(val(1))*scale_)),
                      std::max(1, static_cast<int>(nSigma*std::sqrt(val(0))*scale_)));
        cv::ellipse(img_, toPixel(xf, yf), axes, angleDeg, 0, 360, colour, 1, cv::LINE_AA);
    }

    const cv::Mat & image() const { return img_; }

private:
    FieldDimensions dims_;
    double scale_;
    int width_, height_;
    int fieldW_ = 0, fieldH_ = 0;
    const int topMargin_ = 44;      ///< Title band height [px]
    const int bottomMargin_ = 104;  ///< Legend band height [px]
    cv::Mat img_;
};

void runFieldLocalisation(const std::filesystem::path & dataDir, int interactive, const std::filesystem::path & outputDirectory)
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
    std::vector<BodyTwistSample> twists = SystemLocalisation::twistFromOdometry(log.sensors, t0);
    std::println("Derived {} body twist samples from odometry", twists.size());

    // Field landmark map
    FieldMap map;

    // Initialise from the first finite NUbots baseline estimate (offline convenience;
    // replaced by hypothesis-bank global initialisation later)
    std::size_t initIdx = 0;
    bool haveInit = false;
    Pose<double> Tfb0;
    for (std::size_t i = 0; i < log.fieldBaseline.size(); ++i)
    {
        const FieldBaselineSample & b = log.fieldBaseline[i];
        if (!b.Hfw.rotationMatrix.allFinite() || !b.Hfw.translationVector.allFinite() || !std::isfinite(b.cost))
        {
            continue;
        }
        std::size_t k = nearestIndex(log.sensors, b.t, [](const SensorsSample & s) { return s.t; });
        if (std::abs(log.sensors[k].t - b.t) > 0.1)
        {
            continue;
        }
        // Torso pose in field: Tfb = Hfw * Htw^{-1}
        Tfb0 = b.Hfw*log.sensors[k].Htw.inverse();
        initIdx = i;
        haveInit = true;
        break;
    }
    if (!haveInit)
    {
        std::println("ERROR: no valid baseline sample available to initialise from");
        return;
    }
    const double tInit = log.fieldBaseline[initIdx].t - t0;
    std::println("Initialising at t={:.3f} s from baseline (cost {:.3f})", tInit, log.fieldBaseline[initIdx].cost);

    Eigen::VectorXd eta0 = Eigen::VectorXd::Zero(SystemLocalisation::nx);
    eta0.head<3>() = Tfb0.translationVector;
    eta0.segment<3>(3) = rot2rpy(Tfb0.rotationMatrix);

    Eigen::MatrixXd S0 = Eigen::MatrixXd::Zero(SystemLocalisation::nx, SystemLocalisation::nx);
    S0.diagonal() << 0.3, 0.3, 0.05, 0.05, 0.05, 0.15, 0.05, 0.05;  // pose [m/rad] + camera bias [rad]
    auto p0 = GaussianInfo<double>::fromSqrtMoment(eta0, S0);

    SystemLocalisation system(p0, twists);
    system.resetTo(p0, tInit);

    // Multi-hypothesis field-symmetry handling. When enabled the belief is a
    // Gaussian mixture seeded with the initial pose and its 180 deg mirror; the
    // wrong mirror is down-weighted by the asymmetric landmark evidence and
    // pruned. Default off preserves the single-hypothesis shipping behaviour
    // (initialised here from a known-good baseline, so no ambiguity to resolve).
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
        std::size_t nAssoc, nCand;
        double baseX, baseY, baseYaw, baseCost;
        double errXY, errYaw;
        double updateMs;
    };
    std::vector<Record> records;
    records.reserve(log.vision.size());

    // The NUbots baseline is itself an estimate whose cost spikes when it is
    // lost; restrict the headline comparison to samples where it is trustworthy.
    const double trustedBaselineCost = 1.0;

    double sumSqErrXY = 0, sumSqErrYaw = 0, sumMs = 0, maxMs = 0;
    double sumSqErrXYTrusted = 0, sumSqErrYawTrusted = 0;
    std::size_t nCompared = 0, nTrusted = 0, nUpdates = 0, nSkipped = 0;

    for (const VisionSample & v : log.vision)
    {
        const double t = v.t - t0;
        if (t < tInit || v.detections.empty())
        {
            nSkipped++;
            continue;
        }
        if (!v.Hcw.rotationMatrix.allFinite() || !v.Hcw.translationVector.allFinite())
        {
            nSkipped++;
            continue;
        }

        // Camera pose w.r.t. torso from the kinematic chain (odometry cancels):
        // Tbc = Htw * Hcw^{-1}
        std::size_t k = nearestIndex(log.sensors, v.t, [](const SensorsSample & s) { return s.t; });
        Pose<double> Tbc = log.sensors[k].Htw*v.Hcw.inverse();

        // Measurement toggles (for ablation experiments)
        constexpr bool useGravity = true;
        constexpr bool useKinematicHeight = true;
        constexpr bool useFieldLines = false;

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
            }
        }
        if (useGravity && log.sensors[k].accelerometer.allFinite())
        {
            MeasurementGravity mg(t, log.sensors[k].accelerometer);
            system.process(mg);
        }
        if (useKinematicHeight)
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

        // Record state and comparison to NUbots baseline
        Record r;
        r.t = t;
        r.mean = system.density.mean();
        r.sigma = system.density.cov().diagonal().cwiseSqrt();
        r.nAssoc = meas.numAssociated();
        r.nCand = meas.numCandidates();
        r.updateMs = ms;
        r.baseX = r.baseY = r.baseYaw = r.baseCost = r.errXY = r.errYaw = std::numeric_limits<double>::quiet_NaN();

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
                r.errYaw = wrapAngle(r.mean(5) - r.baseYaw);
                sumSqErrXY += r.errXY*r.errXY;
                sumSqErrYaw += r.errYaw*r.errYaw;
                nCompared++;
                if (std::isfinite(b.cost) && b.cost < trustedBaselineCost)
                {
                    sumSqErrXYTrusted += r.errXY*r.errXY;
                    sumSqErrYawTrusted += r.errYaw*r.errYaw;
                    nTrusted++;
                }
            }
        }
        records.push_back(r);
    }

    // Summary
    std::println("");
    std::println("Processed {} vision updates ({} skipped)", nUpdates, nSkipped);
    {
        Eigen::VectorXd xFinal = system.density.mean();
        std::println("Final camera mount bias estimate: roll {:.2f} deg, pitch {:.2f} deg",
                     xFinal(6)*180.0/M_PI, xFinal(7)*180.0/M_PI);
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

    // Trajectory plot
    FieldPlot plot(map.dims);
    for (const Record & r : records)
    {
        if (std::isfinite(r.baseX))
        {
            plot.drawPoint(r.baseX, r.baseY, plotcolour::baseline, 2);
        }
    }
    double lastEllipse = -1e9;
    for (std::size_t i = 0; i < records.size(); ++i)
    {
        const Record & r = records[i];
        double frac = records.size() > 1 ? static_cast<double>(i)/(records.size() - 1) : 0.0;
        plot.drawPoint(r.mean(0), r.mean(1), plotcolour::estimate(frac), 2);
        if (r.t - lastEllipse >= 2.0)
        {
            Eigen::Matrix2d Pxy;
            Pxy << r.sigma(0)*r.sigma(0), 0, 0, r.sigma(1)*r.sigma(1);
            plot.drawCovarianceEllipse(r.mean(0), r.mean(1), Pxy, plotcolour::covariance);
            lastEllipse = r.t;
        }
    }

    plot.drawTitle("RoboCup field localisation (top-down)",
                   nTrusted > 0
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
        csv << "t,x,y,z,roll,pitch,yaw,sx,sy,sz,sroll,spitch,syaw,nAssoc,nCand,baseX,baseY,baseYaw,baseCost,errXY,errYaw,updateMs\n";
        for (const Record & r : records)
        {
            csv << r.t;
            for (int i = 0; i < 6; ++i) csv << ',' << r.mean(i);
            for (int i = 0; i < 6; ++i) csv << ',' << r.sigma(i);
            csv << ',' << r.nAssoc << ',' << r.nCand
                << ',' << r.baseX << ',' << r.baseY << ',' << r.baseYaw << ',' << r.baseCost
                << ',' << r.errXY << ',' << r.errYaw << ',' << r.updateMs << '\n';
        }

        cv::imwrite((outputDirectory / "field_trajectory.png").string(), plot.image());
        std::println("Exported field_localisation.csv and field_trajectory.png to {}", outputDirectory.string());
    }

    if (interactive > 0)
    {
        cv::imshow("Field localisation", plot.image());
        std::println("Press any key to exit.");
        cv::waitKey(0);
    }
}
