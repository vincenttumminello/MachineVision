#include "SensorLog.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

#include <simdjson.h>

namespace
{

constexpr double NaN = std::numeric_limits<double>::quiet_NaN();

// ---------------------------------------------------------------------------------------------
// Tolerant JSON scalar extraction
//
// A handful of fields in the recorded log were never initialised by the logger and therefore
// contain either a garbage double (a stray ~1e-310 style denormal) or the literal JSON string
// "NaN" in place of a genuine number. Neither is a hard error: we simply want NaN out of it, so
// downstream consumers that only look at the fields they actually need (and check for NaN/finite
// where it matters) keep working.
// ---------------------------------------------------------------------------------------------

/**
 * @brief Extract a double from a (possibly missing / possibly stringly-typed) JSON field
 * @param field The result of an object field lookup, e.g. obj["x"]
 * @return The numeric value, or NaN if the field is missing, not a number, or the string "NaN"
 */
double getDoubleTolerant(simdjson::simdjson_result<simdjson::ondemand::value> field)
{
    simdjson::ondemand::value val;
    if (field.get(val) != simdjson::SUCCESS)
    {
        return NaN;
    }
    simdjson::ondemand::json_type type;
    if (val.type().get(type) != simdjson::SUCCESS || type != simdjson::ondemand::json_type::number)
    {
        // Covers the "NaN" string case, null, bool, object, array, and any lookup error
        return NaN;
    }
    double d;
    if (val.get_double().get(d) != simdjson::SUCCESS)
    {
        return NaN;
    }
    return d;
}

/**
 * @brief Extract a std::string_view from a (possibly missing) JSON string field
 */
std::string_view getStringTolerant(simdjson::simdjson_result<simdjson::ondemand::value> field)
{
    std::string_view sv;
    if (field.get_string().get(sv) != simdjson::SUCCESS)
    {
        return {};
    }
    return sv;
}

/**
 * @brief Parse a {"x":.., "y":.., "z":..} JSON object into an Eigen::Vector3d (tolerant of NaN)
 */
Eigen::Vector3d parseVec3(simdjson::simdjson_result<simdjson::ondemand::value> field)
{
    simdjson::ondemand::object obj;
    if (field.get_object().get(obj) != simdjson::SUCCESS)
    {
        return Eigen::Vector3d(NaN, NaN, NaN);
    }
    double x = getDoubleTolerant(obj["x"]);
    double y = getDoubleTolerant(obj["y"]);
    double z = getDoubleTolerant(obj["z"]);
    return Eigen::Vector3d(x, y, z);
}

/**
 * @brief Parse a serialised Isometry3 (4 columns x, y, z, t; each column is {x,y,z,t}) into a Pose
 *
 * The keys x, y, z, t are the four COLUMNS of a 4x4 homogeneous transform. The inner x, y, z, t
 * keys of each column object are the ROWS. Only the top 3x4 block is meaningful.
 */
Pose<double> parseIso3(simdjson::simdjson_result<simdjson::ondemand::value> field)
{
    Pose<double> T;
    simdjson::ondemand::object obj;
    if (field.get_object().get(obj) != simdjson::SUCCESS)
    {
        T.rotationMatrix.setConstant(NaN);
        T.translationVector.setConstant(NaN);
        return T;
    }
    T.rotationMatrix.col(0) = parseVec3(obj["x"]);
    T.rotationMatrix.col(1) = parseVec3(obj["y"]);
    T.rotationMatrix.col(2) = parseVec3(obj["z"]);
    T.translationVector     = parseVec3(obj["t"]);
    return T;
}

/**
 * @brief Parse an ISO-8601 UTC timestamp with nanosecond precision, e.g.
 *        "2026-07-03T01:53:37.736980592Z", into seconds since the Unix epoch.
 *
 * Parsed manually (fixed-width fields) rather than via locale-dependent strptime/sscanf on a
 * possibly-non-null-terminated simdjson string_view. std::chrono::sys_days (C++20) is used to
 * convert the calendar date component to days since the epoch without a timegm() dependency.
 */
double parseIso8601(std::string_view s)
{
    // Fixed layout: YYYY-MM-DDTHH:MM:SS[.fraction]Z
    if (s.size() < 19)
    {
        return NaN;
    }

    auto digits = [&](std::size_t pos, std::size_t len) -> long {
        long v = 0;
        for (std::size_t i = 0; i < len; ++i)
        {
            char c = s[pos + i];
            if (c < '0' || c > '9')
            {
                return -1;
            }
            v = v * 10 + (c - '0');
        }
        return v;
    };

    long year = digits(0, 4);
    long month = digits(5, 2);
    long day = digits(8, 2);
    long hour = digits(11, 2);
    long minute = digits(14, 2);
    long second = digits(17, 2);
    if (year < 0 || month < 0 || day < 0 || hour < 0 || minute < 0 || second < 0)
    {
        return NaN;
    }

    double frac = 0.0;
    std::size_t pos = 19;
    if (pos < s.size() && s[pos] == '.')
    {
        std::size_t start = ++pos;
        while (pos < s.size() && s[pos] >= '0' && s[pos] <= '9')
        {
            ++pos;
        }
        std::size_t len = pos - start;
        if (len > 0)
        {
            long fracDigits = digits(start, len);
            if (fracDigits >= 0)
            {
                frac = static_cast<double>(fracDigits) / std::pow(10.0, static_cast<double>(len));
            }
        }
    }

    using namespace std::chrono;
    year_month_day ymd{std::chrono::year{static_cast<int>(year)},
                        std::chrono::month{static_cast<unsigned>(month)},
                        std::chrono::day{static_cast<unsigned>(day)}};
    sys_days sd{ymd};
    double daysEpochSeconds = static_cast<double>(sd.time_since_epoch().count()) * 86400.0;

    return daysEpochSeconds + hour * 3600.0 + minute * 60.0 + second + frac;
}

/**
 * @brief Binary-search frameTimes (sorted ascending) for the closest entry to a capture time
 * @param t          Capture time to match [s since epoch]
 * @param frameTimes Video frame times, sorted ascending [s since epoch]
 * @param maxErr     Maximum allowed |frameTimes[idx] - t| for a match to be accepted [s]
 * @return The matching frame index, or -1 if t is not finite or no frame is within maxErr
 */
int matchVideoFrame(double t, const std::vector<double> & frameTimes, double maxErr)
{
    if (std::isnan(t))
    {
        return -1;
    }
    auto it = std::lower_bound(frameTimes.begin(), frameTimes.end(), t);
    std::size_t bestIdx = 0;
    double bestErr = std::numeric_limits<double>::infinity();
    if (it != frameTimes.end())
    {
        std::size_t idx = static_cast<std::size_t>(it - frameTimes.begin());
        double err = std::abs(frameTimes[idx] - t);
        if (err < bestErr)
        {
            bestErr = err;
            bestIdx = idx;
        }
    }
    if (it != frameTimes.begin())
    {
        std::size_t idx = static_cast<std::size_t>(it - frameTimes.begin()) - 1;
        double err = std::abs(frameTimes[idx] - t);
        if (err < bestErr)
        {
            bestErr = err;
            bestIdx = idx;
        }
    }
    return (bestErr <= maxErr) ? static_cast<int>(bestIdx) : -1;
}

/**
 * @brief Load the video frame timecodes from file
 *
 * One value per line, one line per video frame, in frame order. Each value is the absolute
 * capture time of that frame as a Unix timestamp in seconds (fractional seconds allowed).
 * Because the frames are captured in order the values are expected to be monotonically
 * non-decreasing; the caller verifies this.
 */
std::vector<double> loadTimecodes(const std::filesystem::path & timecodePath)
{
    std::ifstream file(timecodePath);
    if (!file)
    {
        throw std::runtime_error("SensorLog: failed to open timecode file " + timecodePath.string());
    }
    std::vector<double> tc;
    std::string line;
    while (std::getline(file, line))
    {
        if (line.empty())
        {
            continue;
        }
        tc.push_back(std::stod(line));
    }
    return tc;
}

} // namespace

SensorLog::SensorLog(const std::filesystem::path & jsonPath, const std::filesystem::path & timecodePath)
{
    // frameTimes[i] is the absolute capture time (Unix seconds) of video frame i, in frame order.
    frameTimes = loadTimecodes(timecodePath);

    simdjson::ondemand::parser parser;
    simdjson::padded_string json = simdjson::padded_string::load(jsonPath.string());
    simdjson::ondemand::document_stream docs;
    if (parser.iterate_many(json).get(docs) != simdjson::SUCCESS)
    {
        throw std::runtime_error("SensorLog: failed to open " + jsonPath.string());
    }

    for (auto docResult : docs)
    {
        // docResult is a simdjson_result<ondemand::document_reference>: fields are accessed
        // directly on it (forward-only, with automatic out-of-order fallback), so there is no
        // need to materialise an intermediate ondemand::document/object for the root value.
        std::string_view type = getStringTolerant(docResult["type"]);

        if (type == "message.input.Sensors")
        {
            simdjson::ondemand::object data;
            if (docResult["data"].get_object().get(data) != simdjson::SUCCESS)
            {
                continue;
            }
            SensorsSample sample;
            sample.t = parseIso8601(getStringTolerant(data["timestamp"]));
            sample.Htw = parseIso3(data["Htw"]);
            sample.accelerometer = parseVec3(data["accelerometer"]);
            sample.gyroscope = parseVec3(data["gyroscope"]);
            sensors.push_back(std::move(sample));
        }
        else if (type == "message.vision.BoundingBoxes")
        {
            simdjson::ondemand::object data;
            if (docResult["data"].get_object().get(data) != simdjson::SUCCESS)
            {
                continue;
            }
            VisionSample sample;
            sample.t = parseIso8601(getStringTolerant(data["timestamp"]));
            sample.videoFrame = -1;
            sample.Hcw = parseIso3(data["Hcw"]);

            simdjson::ondemand::array boxes;
            if (data["boundingBoxes"].get_array().get(boxes) == simdjson::SUCCESS)
            {
                for (auto boxResult : boxes)
                {
                    simdjson::ondemand::object boxObj;
                    if (boxResult.get_object().get(boxObj) != simdjson::SUCCESS)
                    {
                        continue;
                    }
                    Detection det;
                    det.name = std::string(getStringTolerant(boxObj["name"]));
                    det.confidence = getDoubleTolerant(boxObj["confidence"]);
                    det.corners.setConstant(NaN);

                    simdjson::ondemand::array corners;
                    if (boxObj["corners"].get_array().get(corners) == simdjson::SUCCESS)
                    {
                        int col = 0;
                        for (auto cornerResult : corners)
                        {
                            if (col >= 4)
                            {
                                break;
                            }
                            det.corners.col(col) = parseVec3(cornerResult);
                            ++col;
                        }
                    }
                    sample.detections.push_back(std::move(det));
                }
            }

            vision.push_back(std::move(sample));
        }
        else if (type == "message.behaviour.state.WalkState")
        {
            int64_t timestampUs = 0;
            bool haveTimestamp = docResult["timestamp"].get_int64().get(timestampUs) == simdjson::SUCCESS;

            simdjson::ondemand::object data;
            if (docResult["data"].get_object().get(data) != simdjson::SUCCESS)
            {
                continue;
            }
            WalkStateSample sample;
            sample.t = haveTimestamp ? static_cast<double>(timestampUs) * 1e-6 : NaN;
            sample.state = std::string(getStringTolerant(data["state"]));
            sample.velocityTarget = parseVec3(data["velocityTarget"]);
            walk.push_back(std::move(sample));
        }
        else if (type == "message.localisation.Field")
        {
            int64_t timestampUs = 0;
            bool haveTimestamp = docResult["timestamp"].get_int64().get(timestampUs) == simdjson::SUCCESS;

            simdjson::ondemand::object data;
            if (docResult["data"].get_object().get(data) != simdjson::SUCCESS)
            {
                continue;
            }
            FieldBaselineSample sample;
            sample.t = haveTimestamp ? static_cast<double>(timestampUs) * 1e-6 : NaN;
            sample.Hfw = parseIso3(data["Hfw"]);
            sample.cost = getDoubleTolerant(data["cost"]);
            fieldBaseline.push_back(std::move(sample));
        }
        else if (type == "message.input.MotionCapture")
        {
            int64_t timestampUs = 0;
            bool haveTimestamp = docResult["timestamp"].get_int64().get(timestampUs) == simdjson::SUCCESS;

            simdjson::ondemand::object data;
            if (docResult["data"].get_object().get(data) != simdjson::SUCCESS)
            {
                continue;
            }
            // The NatNet per-frame timestamps in the payload are uninitialised garbage in this
            // log; the envelope timestamp (receive time, same clock as everything else) is used.
            // Only the first rigid body is taken: the capture volume tracks the one robot.
            simdjson::ondemand::array bodies;
            if (data["rigidBodies"].get_array().get(bodies) != simdjson::SUCCESS)
            {
                continue;
            }
            for (auto bodyResult : bodies)
            {
                simdjson::ondemand::object body;
                if (bodyResult.get_object().get(body) != simdjson::SUCCESS)
                {
                    break;
                }
                MocapSample sample;
                sample.t = haveTimestamp ? static_cast<double>(timestampUs) * 1e-6 : NaN;
                sample.position = parseVec3(body["position"]);

                // Quaternion streamed as {x, y, z, t} with t the scalar (w) part.
                simdjson::ondemand::object rot;
                double qx = NaN, qy = NaN, qz = NaN, qw = NaN;
                if (body["rotation"].get_object().get(rot) == simdjson::SUCCESS)
                {
                    qx = getDoubleTolerant(rot["x"]);
                    qy = getDoubleTolerant(rot["y"]);
                    qz = getDoubleTolerant(rot["z"]);
                    qw = getDoubleTolerant(rot["t"]);
                }
                const double n2 = qx*qx + qy*qy + qz*qz + qw*qw;
                bool trackingValid = false;
                (void) body["trackingValid"].get_bool().get(trackingValid);
                sample.valid = trackingValid && sample.position.allFinite()
                               && std::isfinite(n2) && n2 > 1e-12;
                if (sample.valid)
                {
                    const double s = 1.0 / std::sqrt(n2);
                    qx *= s; qy *= s; qz *= s; qw *= s;
                    sample.R << 1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw),
                                2*(qx*qy + qz*qw),     1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw),
                                2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw),     1 - 2*(qx*qx + qy*qy);
                }
                else
                {
                    sample.R.setConstant(NaN);
                }
                mocap.push_back(std::move(sample));
                break;
            }
        }
        else if (type == "message.vision.FieldLines")
        {
            simdjson::ondemand::object data;
            if (docResult["data"].get_object().get(data) != simdjson::SUCCESS)
            {
                continue;
            }
            LinePointsSample sample;
            sample.t = parseIso8601(getStringTolerant(data["timestamp"]));
            sample.videoFrame = -1;
            sample.Hcw = parseIso3(data["Hcw"]);

            // rPWw: field-line points on the ground plane in world space, produced upstream by
            // intersecting camera rays with the ground plane using this same Hcw. Recover the
            // odometry-independent measurement (a unit ray in the camera frame) by re-applying
            // Hcw to each point and normalising.
            std::vector<Eigen::Vector3d> rayList;
            simdjson::ondemand::array pts;
            if (data["rPWw"].get_array().get(pts) == simdjson::SUCCESS)
            {
                for (auto ptResult : pts)
                {
                    Eigen::Vector3d rPWw = parseVec3(ptResult);
                    if (!rPWw.array().isFinite().all())
                    {
                        continue;
                    }
                    Eigen::Vector3d rPCc = sample.Hcw * rPWw;
                    if (!rPCc.array().isFinite().all())
                    {
                        continue;
                    }
                    double norm = rPCc.norm();
                    if (!(norm >= 1e-9))
                    {
                        continue;
                    }
                    rayList.push_back(rPCc / norm);
                }
            }
            sample.rays.resize(3, static_cast<Eigen::Index>(rayList.size()));
            for (std::size_t col = 0; col < rayList.size(); ++col)
            {
                sample.rays.col(static_cast<Eigen::Index>(col)) = rayList[col];
            }

            linePoints.push_back(std::move(sample));
        }
        // else: not one of the five wanted types; the document_stream iterator skips the
        // remainder of this document cheaply (no further parsing/materialisation occurs since
        // on-demand only parses fields that are actually accessed).
    }

    // ------------------------------------------------------------------------------------------
    // Video frame alignment
    //
    // frameTimes already holds each frame's absolute capture time (Unix seconds), on the same
    // clock as the message capture timestamps, so a vision sample is aligned to a frame purely
    // by matching capture times. No per-run offset estimation is needed. Dropped video frames,
    // dropped message packets, and asynchronously logged messages all fall out gracefully: a
    // sample with no frame within the match tolerance simply keeps videoFrame == -1.
    // ------------------------------------------------------------------------------------------
    if (frameTimes.empty())
    {
        throw std::runtime_error("SensorLog: timecode file is empty");
    }

    // Frames are captured in order, so their timestamps must be non-decreasing; matchVideoFrame
    // also relies on frameTimes being sorted ascending. A violation means the file is not in
    // frame order (or not Unix timestamps at all), which would silently corrupt alignment.
    for (std::size_t j = 1; j < frameTimes.size(); ++j)
    {
        if (frameTimes[j] < frameTimes[j - 1])
        {
            throw std::runtime_error(
                "SensorLog: timecode file is not monotonically non-decreasing at line "
                + std::to_string(j + 1) + "; expected Unix timestamps (seconds) in frame order");
        }
    }

    constexpr double kMaxFrameMatchError = 5e-3; // 5 ms
    int nMatched = 0;
    for (VisionSample & v : vision)
    {
        v.videoFrame = matchVideoFrame(v.t, frameTimes, kMaxFrameMatchError);
        if (v.videoFrame != -1)
        {
            ++nMatched;
        }
    }

    // Sanity check that the two clocks are genuinely aligned. Individual unmatched samples are
    // expected (dropped frames / async messages), but a wholesale mismatch means the timecode
    // file and the log are on different clocks or otherwise incompatible.
    if (!vision.empty() && static_cast<double>(nMatched) < 0.9 * static_cast<double>(vision.size()))
    {
        throw std::runtime_error("SensorLog: fewer than 90% of vision samples matched a video frame; "
                                 "timecode and log clocks appear unaligned");
    }

    for (LinePointsSample & lp : linePoints)
    {
        lp.videoFrame = matchVideoFrame(lp.t, frameTimes, kMaxFrameMatchError);
    }

    // ------------------------------------------------------------------------------------------
    // Sort each stream by time (the log is expected to already be close to time-ordered per
    // stream, but sort defensively) and compute t0.
    // ------------------------------------------------------------------------------------------
    auto byTime = [](const auto & a, const auto & b) { return a.t < b.t; };
    std::stable_sort(sensors.begin(), sensors.end(), byTime);
    std::stable_sort(vision.begin(), vision.end(), byTime);
    std::stable_sort(walk.begin(), walk.end(), byTime);
    std::stable_sort(fieldBaseline.begin(), fieldBaseline.end(), byTime);
    std::stable_sort(linePoints.begin(), linePoints.end(), byTime);
    std::stable_sort(mocap.begin(), mocap.end(), byTime);

    t0 = std::numeric_limits<double>::infinity();
    if (!sensors.empty())
    {
        t0 = std::min(t0, sensors.front().t);
    }
    if (!vision.empty())
    {
        t0 = std::min(t0, vision.front().t);
    }
    if (!walk.empty())
    {
        t0 = std::min(t0, walk.front().t);
    }
    if (!fieldBaseline.empty())
    {
        t0 = std::min(t0, fieldBaseline.front().t);
    }
    if (!linePoints.empty())
    {
        t0 = std::min(t0, linePoints.front().t);
    }
    if (!std::isfinite(t0))
    {
        t0 = 0.0;
    }
}
