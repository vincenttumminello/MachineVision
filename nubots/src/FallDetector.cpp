#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <string>
#include <string_view>
#include <vector>
#include <Eigen/Core>
#include "FallDetector.h"
#include "SensorLog.h"

const char * to_string(Posture p)
{
    switch (p)
    {
        case Posture::UPRIGHT:    return "upright";
        case Posture::FALLING:    return "falling";
        case Posture::FALLEN:     return "fallen";
        case Posture::GETTING_UP: return "getting up";
    }
    return "?";
}

namespace
{

bool contains(const std::string & haystack, const char * needle)
{
    return haystack.find(needle) != std::string::npos;
}

} // namespace

Posture FallDetector::postureFromStateName(std::string_view state, bool & known)
{
    std::string s;
    s.reserve(state.size());
    for (char c : state)
    {
        s.push_back(static_cast<char>(std::toupper(static_cast<unsigned char>(c))));
    }

    known = true;
    // FALLING is tested before FALLEN only for readability; the two spellings are
    // distinct, so the order does not actually matter here.
    if (contains(s, "FALLING"))                                          return Posture::FALLING;
    if (contains(s, "FALLEN") || contains(s, "LYING")
        || contains(s, "PRONE") || contains(s, "SUPINE"))                return Posture::FALLEN;
    if (contains(s, "GETUP") || contains(s, "GETTING_UP")
        || contains(s, "GETTINGUP") || contains(s, "RECOVER"))           return Posture::GETTING_UP;
    if (contains(s, "STANDING") || contains(s, "UPRIGHT") || contains(s, "DYNAMIC")
        || contains(s, "WALKING") || contains(s, "READY") || contains(s, "CROUCH")
        || contains(s, "STANDBY") || contains(s, "RELAXED"))             return Posture::UPRIGHT;
    // UNKNOWN is a legitimate value the behaviour module publishes before it has
    // decided (notably at startup, standing still), so it is mapped to upright
    // rather than reported: treating it as a fall would suppress every update
    // until the first real classification arrives.
    if (contains(s, "UNKNOWN"))                                          return Posture::UPRIGHT;

    known = false;
    return Posture::UPRIGHT;
}

FallDetector::FallDetector(const std::vector<SensorsSample> & sensors,
                           const std::vector<StabilitySample> & stability,
                           double t0)
    : FallDetector(sensors, stability, t0, Options{})
{}

FallDetector::FallDetector(const std::vector<SensorsSample> & sensors,
                           const std::vector<StabilitySample> & stability,
                           double t0,
                           const Options & opts)
    : options(opts)
{
    if (!stability.empty())
    {
        usingFlags_ = true;
        buildFromFlags(stability, t0);
    }
    else
    {
        usingFlags_ = false;
        buildFromAccelerometer(sensors, t0);
    }

    // A getup script reaching its final stand is not the moment the walk engine's
    // odometry becomes meaningful again, and it is not the moment the robot's own
    // flags stop lagging reality. Both sources therefore get the same settle tail,
    // during which the posture reads GETTING_UP.
    if (options.settle > 0.0 && !timeline_.empty())
    {
        std::vector<Entry> out;
        out.reserve(timeline_.size() + 2);
        for (std::size_t i = 0; i < timeline_.size(); ++i)
        {
            const bool recovered = timeline_[i].posture == Posture::UPRIGHT
                                   && i > 0 && timeline_[i - 1].posture != Posture::UPRIGHT;
            if (!recovered)
            {
                out.push_back(timeline_[i]);
                continue;
            }
            const double tailEnd = timeline_[i].t + options.settle;
            const double next = (i + 1 < timeline_.size()) ? timeline_[i + 1].t
                                                           : std::numeric_limits<double>::infinity();
            out.push_back({timeline_[i].t, Posture::GETTING_UP});
            if (tailEnd < next)
            {
                out.push_back({tailEnd, Posture::UPRIGHT});
            }
            // Otherwise the next entry supersedes the tail before it expires.
        }
        timeline_.swap(out);
    }

    // Collapse runs of equal posture so `intervals()` and the diagnostics see one
    // breakpoint per genuine transition.
    std::vector<Entry> compressed;
    compressed.reserve(timeline_.size());
    for (const Entry & e : timeline_)
    {
        if (compressed.empty() || compressed.back().posture != e.posture)
        {
            compressed.push_back(e);
        }
    }
    timeline_.swap(compressed);

    buildIntervals();
}

void FallDetector::buildFromFlags(const std::vector<StabilitySample> & stability, double t0)
{
    timeline_.reserve(stability.size());
    for (const StabilitySample & s : stability)
    {
        if (!std::isfinite(s.t))
        {
            continue;
        }
        bool known = false;
        const Posture p = postureFromStateName(s.state, known);
        if (!known
            && std::find(unrecognised_.begin(), unrecognised_.end(), s.state) == unrecognised_.end())
        {
            unrecognised_.push_back(s.state);
        }
        timeline_.push_back({s.t - t0, p});
    }
}

void FallDetector::buildFromAccelerometer(const std::vector<SensorsSample> & sensors, double t0)
{
    // Usable samples only: the tilt estimate is an average over a window, so a
    // non-finite reading in the middle of one would poison the whole window.
    std::vector<double> ts;
    std::vector<Eigen::Vector3d> as;
    ts.reserve(sensors.size());
    as.reserve(sensors.size());
    for (const SensorsSample & s : sensors)
    {
        if (std::isfinite(s.t) && s.accelerometer.allFinite() && s.accelerometer.norm() > 1e-6)
        {
            ts.push_back(s.t - t0);
            as.push_back(s.accelerometer);
        }
    }
    const std::size_t n = ts.size();
    if (n == 0)
    {
        return;
    }

    // Smoothed torso tilt. Averaging the specific-force vectors over a window
    // (rather than averaging per-sample tilt angles) is what suppresses the
    // walking transients: the dynamic component is roughly zero-mean over a
    // footstep, the gravity component is not. Raw single samples reach 116 deg
    // of apparent tilt during ordinary walking in the recorded logs, so nothing
    // instantaneous is usable as a fall signal.
    std::vector<double> tilt(n);
    {
        std::size_t lo = 0, hi = 0;
        Eigen::Vector3d sum = Eigen::Vector3d::Zero();
        for (std::size_t i = 0; i < n; ++i)
        {
            while (hi < n && ts[hi] <= ts[i] + options.smoothWindow)
            {
                sum += as[hi];
                ++hi;
            }
            while (lo < hi && ts[lo] < ts[i] - options.smoothWindow)
            {
                sum -= as[lo];
                ++lo;
            }
            const std::size_t count = hi - lo;
            const Eigen::Vector3d mean = count > 0 ? Eigen::Vector3d(sum/static_cast<double>(count))
                                                   : as[i];
            const double norm = mean.norm();
            tilt[i] = norm > 1e-6 ? std::acos(std::clamp(mean.z()/norm, -1.0, 1.0)) : 0.0;
        }
    }

    // Mark the fallen windows. A core is a run past tiltEnter that lasts at least
    // minFallen; it is then grown outwards in both directions to wherever the tilt
    // last crossed tiltExit, which captures the topple on the way in and the time
    // spent on the ground on the way out.
    std::vector<Posture> posture(n, Posture::UPRIGHT);
    std::size_t i = 0;
    while (i < n)
    {
        if (tilt[i] <= options.tiltEnter)
        {
            ++i;
            continue;
        }
        std::size_t core0 = i;
        std::size_t core1 = i;
        while (core1 + 1 < n && tilt[core1 + 1] > options.tiltEnter)
        {
            ++core1;
        }
        i = core1 + 1;

        if (ts[core1] - ts[core0] < options.minFallen)
        {
            continue;       // Walking transient, not a fall
        }

        std::size_t down0 = core0;
        while (down0 > 0 && tilt[down0 - 1] > options.tiltExit)
        {
            --down0;
        }
        std::size_t down1 = core1;
        while (down1 + 1 < n && tilt[down1 + 1] > options.tiltExit)
        {
            ++down1;
        }

        for (std::size_t j = down0; j < core0; ++j)
        {
            posture[j] = Posture::FALLING;
        }
        for (std::size_t j = core0; j <= down1; ++j)
        {
            posture[j] = Posture::FALLEN;
        }
    }

    timeline_.reserve(n);
    for (std::size_t j = 0; j < n; ++j)
    {
        timeline_.push_back({ts[j], posture[j]});
    }
}

void FallDetector::buildIntervals()
{
    intervals_.clear();
    for (std::size_t i = 0; i < timeline_.size(); ++i)
    {
        if (timeline_[i].posture == Posture::UPRIGHT)
        {
            continue;
        }
        if (!intervals_.empty() && intervals_.back().end >= timeline_[i].t)
        {
            continue;       // Already inside a reported window
        }
        Interval iv;
        iv.start = timeline_[i].t;
        std::size_t j = i;
        while (j + 1 < timeline_.size() && timeline_[j + 1].posture != Posture::UPRIGHT)
        {
            ++j;
        }
        iv.end = (j + 1 < timeline_.size()) ? timeline_[j + 1].t
                                            : std::numeric_limits<double>::infinity();
        intervals_.push_back(iv);
        i = j;
    }
}

Posture FallDetector::at(double t) const
{
    if (timeline_.empty() || t < timeline_.front().t)
    {
        return Posture::UPRIGHT;
    }
    // Last breakpoint at or before t; the posture holds until the next one.
    auto it = std::upper_bound(timeline_.begin(), timeline_.end(), t,
        [](double time, const Entry & e) { return time < e.t; });
    return std::prev(it)->posture;
}
