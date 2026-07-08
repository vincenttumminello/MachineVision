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
 * @brief Load the video frame timecodes (ms since video start, one per line) from file
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
    std::vector<double> timecodesMs = loadTimecodes(timecodePath);

    simdjson::ondemand::parser parser;
    simdjson::padded_string json = simdjson::padded_string::load(jsonPath.string());
    simdjson::ondemand::document_stream docs;
    if (parser.iterate_many(json).get(docs) != simdjson::SUCCESS)
    {
        throw std::runtime_error("SensorLog: failed to open " + jsonPath.string());
    }

    bool haveFirstVisionCapture = false;
    double firstVisionCapture = 0.0;

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

            if (!haveFirstVisionCapture && !std::isnan(sample.t))
            {
                haveFirstVisionCapture = true;
                firstVisionCapture = sample.t;
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
        // else: not one of the four wanted types; the document_stream iterator skips the
        // remainder of this document cheaply (no further parsing/materialisation occurs since
        // on-demand only parses fields that are actually accessed).
    }

    // ------------------------------------------------------------------------------------------
    // Video frame alignment
    //
    // T0 = (first BoundingBoxes capture time) - tc[2]/1000, since the first two video frames
    // precede the first vision message (empirically verified to sub-millisecond accuracy).
    // ------------------------------------------------------------------------------------------
    if (!haveFirstVisionCapture)
    {
        throw std::runtime_error("SensorLog: no message.vision.BoundingBoxes samples with a valid capture time");
    }
    if (timecodesMs.size() < 3)
    {
        throw std::runtime_error("SensorLog: timecode file has fewer than 3 entries");
    }

    double T0 = firstVisionCapture - timecodesMs[2] / 1000.0;
    frameTimes.resize(timecodesMs.size());
    for (std::size_t j = 0; j < timecodesMs.size(); ++j)
    {
        frameTimes[j] = T0 + timecodesMs[j] / 1000.0;
    }

    constexpr double kMaxFrameMatchError = 5e-3; // 5 ms
    int nMatched = 0;
    for (VisionSample & v : vision)
    {
        if (std::isnan(v.t))
        {
            v.videoFrame = -1;
            continue;
        }
        // Binary search frameTimes (sorted ascending) for the closest entry to v.t
        auto it = std::lower_bound(frameTimes.begin(), frameTimes.end(), v.t);
        std::size_t bestIdx = 0;
        double bestErr = std::numeric_limits<double>::infinity();
        if (it != frameTimes.end())
        {
            std::size_t idx = static_cast<std::size_t>(it - frameTimes.begin());
            double err = std::abs(frameTimes[idx] - v.t);
            if (err < bestErr)
            {
                bestErr = err;
                bestIdx = idx;
            }
        }
        if (it != frameTimes.begin())
        {
            std::size_t idx = static_cast<std::size_t>(it - frameTimes.begin()) - 1;
            double err = std::abs(frameTimes[idx] - v.t);
            if (err < bestErr)
            {
                bestErr = err;
                bestIdx = idx;
            }
        }
        if (bestErr <= kMaxFrameMatchError)
        {
            v.videoFrame = static_cast<int>(bestIdx);
            ++nMatched;
        }
        else
        {
            v.videoFrame = -1;
        }
    }

    if (!vision.empty() && static_cast<double>(nMatched) < 0.99 * static_cast<double>(vision.size()))
    {
        throw std::runtime_error("SensorLog: fewer than 99% of vision samples matched a video frame");
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
    if (!std::isfinite(t0))
    {
        t0 = 0.0;
    }
}
