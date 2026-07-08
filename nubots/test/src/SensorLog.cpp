#include <doctest/doctest.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <string>
#include <vector>
#include "../../src/SensorLog.h"

namespace
{

// The recorded log lives in nubots/data, but the test binary may be invoked either from the
// project source directory (nubots/) or from the build directory (nubots/build/), so try a
// couple of candidate relative locations before giving up.
std::filesystem::path findDataFile(const std::string & filename)
{
    const std::vector<std::filesystem::path> candidates = {
        std::filesystem::path("data") / filename,
        std::filesystem::path("..") / "data" / filename,
        std::filesystem::path("nubots") / "data" / filename,
    };
    for (const std::filesystem::path & p : candidates)
    {
        if (std::filesystem::exists(p))
        {
            return p;
        }
    }
    return std::filesystem::path("data") / filename; // does not exist; caller checks
}

template <typename Sample>
bool isTimeOrdered(const std::vector<Sample> & v)
{
    for (std::size_t i = 1; i < v.size(); ++i)
    {
        if (v[i].t < v[i - 1].t)
        {
            return false;
        }
    }
    return true;
}

} // namespace

SCENARIO("SensorLog: parse recorded_data.json and align to Left_timecode.txt")
{
    GIVEN("A recorded sensor log and matching video timecode file")
    {
        std::filesystem::path jsonPath = findDataFile("recorded_data.json");
        std::filesystem::path timecodePath = findDataFile("Left_timecode.txt");

        if (!std::filesystem::exists(jsonPath) || !std::filesystem::exists(timecodePath))
        {
            MESSAGE("SensorLog test SKIPPED: data files not found (looked for data/recorded_data.json "
                    "and data/Left_timecode.txt relative to the working directory)");
            return;
        }

        WHEN("Constructing a SensorLog")
        {
            auto start = std::chrono::steady_clock::now();
            SensorLog log(jsonPath, timecodePath);
            auto end = std::chrono::steady_clock::now();
            double loadSeconds = std::chrono::duration<double>(end - start).count();
            MESSAGE("SensorLog: loaded " << jsonPath.string() << " in " << loadSeconds << " s");

            THEN("Stream sizes match the known message counts in the log")
            {
                CHECK(log.sensors.size() == 7809);
                CHECK(log.vision.size() == 861);
                CHECK(log.walk.size() == 7607);
                CHECK(log.fieldBaseline.size() == 846);
                CHECK(log.linePoints.size() == 848);
                CHECK(log.frameTimes.size() == 863);
            }

            THEN("All five streams are strictly (weakly) time-ordered")
            {
                CHECK(isTimeOrdered(log.sensors));
                CHECK(isTimeOrdered(log.vision));
                CHECK(isTimeOrdered(log.walk));
                CHECK(isTimeOrdered(log.fieldBaseline));
                CHECK(isTimeOrdered(log.linePoints));
            }

            THEN("Exactly one vision sample has no matching video frame")
            {
                int nUnmatched = 0;
                for (const VisionSample & v : log.vision)
                {
                    if (v.videoFrame == -1)
                    {
                        ++nUnmatched;
                    }
                }
                CHECK(nUnmatched == 1);
            }

            THEN("Detection corner rays are unit-norm for the first vision sample with detections")
            {
                bool checked = false;
                for (const VisionSample & v : log.vision)
                {
                    if (v.detections.empty())
                    {
                        continue;
                    }
                    for (const Detection & d : v.detections)
                    {
                        for (int col = 0; col < 4; ++col)
                        {
                            Eigen::Vector3d corner = d.corners.col(col);
                            if (corner.array().isFinite().all())
                            {
                                CHECK(corner.norm() == doctest::Approx(1.0).epsilon(1e-6));
                                checked = true;
                            }
                        }
                    }
                    if (checked)
                    {
                        break;
                    }
                }
                CHECK(checked);
            }

            THEN("A mid-log sensors sample has a finite, gravity-magnitude accelerometer reading")
            {
                REQUIRE(log.sensors.size() > 4000);
                const Eigen::Vector3d & a = log.sensors[4000].accelerometer;
                REQUIRE(a.array().isFinite().all());
                double norm = a.norm();
                CHECK(norm > 8.0);
                CHECK(norm < 12.0);
            }

            THEN("Every LinePoints sample has at least one ray, and rays are unit-norm for the first "
                 "non-empty sample")
            {
                bool checked = false;
                for (const LinePointsSample & lp : log.linePoints)
                {
                    CHECK(lp.rays.cols() >= 1);
                    if (!checked && lp.rays.cols() > 0)
                    {
                        for (Eigen::Index col = 0; col < lp.rays.cols(); ++col)
                        {
                            Eigen::Vector3d ray = lp.rays.col(col);
                            CHECK(ray.norm() == doctest::Approx(1.0).epsilon(1e-6));
                        }
                        checked = true;
                    }
                }
                CHECK(checked);
            }
        }
    }
}
