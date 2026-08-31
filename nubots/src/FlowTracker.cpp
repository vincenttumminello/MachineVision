#include <algorithm>
#include <cassert>
#include <vector>
#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include "FlowTracker.h"

FlowTracker::FlowTracker(const CameraLens & lens, const FieldDimensions & dims, const Options & opts)
    : options(opts)
    , lens_(lens)
    , farField_(lens, dims, opts.farField)
{}

FlowTracker::FlowTracker(const CameraLens & lens, const FieldDimensions & dims)
    : FlowTracker(lens, dims, Options{})
{}

void FlowTracker::reset()
{
    prevGray_.release();
    prevPts_.clear();
}

void FlowTracker::detect(const cv::Mat & gray)
{
    const int want = options.maxFeatures - static_cast<int>(prevPts_.size());
    if (want <= 0)
    {
        return;
    }

    // Keep new corners away from the ones already being tracked, so a top-up
    // does not pile duplicates onto features that are working.
    cv::Mat mask(gray.size(), CV_8UC1, cv::Scalar(255));
    const int b = options.imageBorder;
    cv::rectangle(mask, cv::Rect(0, 0, gray.cols, b), cv::Scalar(0), cv::FILLED);
    cv::rectangle(mask, cv::Rect(0, gray.rows - b, gray.cols, b), cv::Scalar(0), cv::FILLED);
    cv::rectangle(mask, cv::Rect(0, 0, b, gray.rows), cv::Scalar(0), cv::FILLED);
    cv::rectangle(mask, cv::Rect(gray.cols - b, 0, b, gray.rows), cv::Scalar(0), cv::FILLED);
    for (const cv::Point2f & p : prevPts_)
    {
        cv::circle(mask, p, static_cast<int>(options.minDistance), cv::Scalar(0), cv::FILLED);
    }

    std::vector<cv::Point2f> fresh;
    cv::goodFeaturesToTrack(gray, fresh, want, options.qualityLevel, options.minDistance, mask);
    prevPts_.insert(prevPts_.end(), fresh.begin(), fresh.end());
}

FlowFrame FlowTracker::track(const cv::Mat & gray, double t, const Pose<double> & Tbc, const Pose<double> & Tfc)
{
    assert(gray.type() == CV_8UC1);

    FlowFrame out;
    out.RbcPrev = prevRbc_;
    out.RbcCurr = Tbc.rotationMatrix;
    out.dt = t - prevT_;

    const bool haveHistory = !prevGray_.empty() && prevPts_.size() >= options.minMatches
                          && prevGray_.size() == gray.size();
    const bool dtUsable = out.dt > options.minDt && out.dt < options.maxDt;

    if (haveHistory && dtUsable)
    {
        std::vector<cv::Point2f> currPts;
        std::vector<unsigned char> status;
        std::vector<float> err;
        const cv::Size win(options.lkWindow, options.lkWindow);
        const cv::TermCriteria term(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01);
        cv::calcOpticalFlowPyrLK(prevGray_, gray, prevPts_, currPts, status, err,
                                 win, options.lkPyramidLevels, term);

        // Forward-backward check: track the result back and keep only the
        // features that land where they started. This is what removes the
        // failure LK does not report -- a window that locked onto the wrong
        // texture still comes back with status = 1 and a plausible error.
        std::vector<cv::Point2f> backPts;
        std::vector<unsigned char> backStatus;
        std::vector<float> backErr;
        cv::calcOpticalFlowPyrLK(gray, prevGray_, currPts, backPts, backStatus, backErr,
                                 win, options.lkPyramidLevels, term);

        const double fb2 = options.fbThreshold*options.fbThreshold;
        const double bx = options.imageBorder;
        std::vector<cv::Point2f> survivors;
        survivors.reserve(prevPts_.size());
        out.matches.reserve(prevPts_.size());

        for (std::size_t i = 0; i < prevPts_.size(); ++i)
        {
            if (!status[i] || !backStatus[i])
            {
                continue;
            }
            const cv::Point2f & p0 = prevPts_[i];
            const cv::Point2f & p1 = currPts[i];
            if (p1.x < bx || p1.x >= gray.cols - bx || p1.y < bx || p1.y >= gray.rows - bx)
            {
                continue;
            }
            const cv::Point2f d = backPts[i] - p0;
            if (d.x*d.x + d.y*d.y > fb2)
            {
                continue;
            }

            // Survived tracking: it stays in the feature set whatever its range.
            survivors.push_back(p1);
            out.nTracked++;

            FlowMatch m;
            m.pxPrev = Eigen::Vector2d(p0.x, p0.y);
            m.pxCurr = Eigen::Vector2d(p1.x, p1.y);
            m.uPrev = lens_.unproject(m.pxPrev);
            m.uCurr = lens_.unproject(m.pxCurr);

            // Only far-field features carry a depth-independent bearing shift.
            // Classified at the current frame's estimated pose; the previous
            // frame's classification would be the same to well inside the test's
            // own margins, and this way one pose serves both this and the ray.
            if (!farField_.isOutOfField(m.uCurr, Tfc))
            {
                out.nNear++;
                continue;
            }
            out.matches.push_back(std::move(m));
        }

        prevPts_ = std::move(survivors);
        out.valid = out.matches.size() >= options.minMatches;
    }
    else
    {
        // No usable history (first frame, a dropout, or a frame interval outside
        // the window): start the feature set again from this frame.
        prevPts_.clear();
    }

    const std::size_t before = prevPts_.size();
    if (prevPts_.size() < static_cast<std::size_t>(options.redetectBelow))
    {
        detect(gray);
    }
    out.nDetected = static_cast<int>(prevPts_.size() - before);

    prevGray_ = gray.clone();
    prevT_ = t;
    prevRbc_ = Tbc.rotationMatrix;
    return out;
}
