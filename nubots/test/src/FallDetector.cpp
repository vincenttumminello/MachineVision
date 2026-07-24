#include <doctest/doctest.h>
#include <cmath>
#include <string>
#include <vector>
#include <Eigen/Core>
#include "../../src/FallDetector.h"
#include "../../src/SensorLog.h"

namespace
{

constexpr double kG = 9.80665;

/// A sensor sample whose accelerometer reads gravity at the given torso pitch.
SensorsSample tiltedSample(double t, double pitch)
{
    SensorsSample s;
    s.t = t;
    // Torso tilted by `pitch` about y: the specific force in the torso frame is
    // Rfb^T * [0 0 g], which for a pitch-only rotation is [-g sin(pitch), 0, g cos(pitch)].
    s.accelerometer = Eigen::Vector3d(-kG*std::sin(pitch), 0.0, kG*std::cos(pitch));
    s.gyroscope = Eigen::Vector3d::Zero();
    return s;
}

/// Sensor stream at 100 Hz that is upright, falls flat, and stays down.
std::vector<SensorsSample> fallStream(double uprightUntil, double downFrom, double until)
{
    std::vector<SensorsSample> out;
    for (double t = 0.0; t < until; t += 0.01)
    {
        double pitch = 0.0;
        if (t >= downFrom)
        {
            pitch = 0.5*M_PI;                       // Flat on its face
        }
        else if (t > uprightUntil)
        {
            // Linear topple between the two
            pitch = 0.5*M_PI*(t - uprightUntil)/(downFrom - uprightUntil);
        }
        out.push_back(tiltedSample(t, pitch));
    }
    return out;
}

StabilitySample flag(double t, const std::string & state)
{
    return StabilitySample{t, state};
}

} // namespace

SCENARIO("FallDetector maps stability state names")
{
    GIVEN("Names a NUbots log may carry")
    {
        bool known = false;
        THEN("upright postures map to UPRIGHT")
        {
            CHECK(FallDetector::postureFromStateName("STANDING", known) == Posture::UPRIGHT);
            CHECK(known);
            CHECK(FallDetector::postureFromStateName("DYNAMIC", known) == Posture::UPRIGHT);
            CHECK(known);
            CHECK(FallDetector::postureFromStateName("UNKNOWN", known) == Posture::UPRIGHT);
            CHECK(known);
        }
        THEN("fall postures are distinguished from each other")
        {
            CHECK(FallDetector::postureFromStateName("FALLING", known) == Posture::FALLING);
            CHECK(FallDetector::postureFromStateName("FALLEN", known) == Posture::FALLEN);
            CHECK(FallDetector::postureFromStateName("LYING", known) == Posture::FALLEN);
            CHECK(FallDetector::postureFromStateName("GETTING_UP", known) == Posture::GETTING_UP);
        }
        THEN("matching is case insensitive")
        {
            CHECK(FallDetector::postureFromStateName("fallen", known) == Posture::FALLEN);
            CHECK(known);
        }
        THEN("an unknown name is reported rather than silently accepted")
        {
            // Silently treating this as upright would disable every fall
            // protection downstream, so `known` is the caller's only warning.
            CHECK(FallDetector::postureFromStateName("SOMETHING_NEW", known) == Posture::UPRIGHT);
            CHECK_FALSE(known);
        }
    }
}

SCENARIO("FallDetector prefers the robot's own flags")
{
    GIVEN("A log with both a stability stream and an accelerometer that never tilts")
    {
        std::vector<SensorsSample> sensors;
        for (double t = 0.0; t < 10.0; t += 0.01)
        {
            sensors.push_back(tiltedSample(t, 0.0));    // Perfectly upright throughout
        }
        std::vector<StabilitySample> stability{
            flag(0.0, "STANDING"), flag(3.0, "FALLING"), flag(3.5, "FALLEN"),
            flag(5.0, "GETTING_UP"), flag(7.0, "STANDING")};

        WHEN("the timeline is built")
        {
            FallDetector::Options opts;
            opts.settle = 1.0;
            FallDetector falls(sensors, stability, 0.0, opts);

            THEN("the flags are the source")
            {
                CHECK(falls.usingFlags());
                CHECK(falls.unrecognisedStates().empty());
            }
            THEN("the flagged window is reported even though the IMU saw nothing")
            {
                REQUIRE(falls.intervals().size() == 1);
                CHECK(falls.intervals()[0].start == doctest::Approx(3.0));
                CHECK(falls.at(2.9) == Posture::UPRIGHT);
                CHECK(falls.at(3.2) == Posture::FALLING);
                CHECK(falls.at(4.0) == Posture::FALLEN);
                CHECK(falls.at(6.0) == Posture::GETTING_UP);
            }
            THEN("the robot is still recovering for the settle tail after the flag clears")
            {
                // A getup script reaching its final stand is not the moment the
                // walk engine's odometry becomes meaningful again.
                CHECK(falls.at(7.5) == Posture::GETTING_UP);
                CHECK(falls.at(8.5) == Posture::UPRIGHT);
                CHECK(falls.intervals()[0].end == doctest::Approx(8.0));
            }
        }
    }
}

SCENARIO("FallDetector derives posture from the accelerometer without flags")
{
    GIVEN("A stream that topples flat at t=3 s and stays down")
    {
        std::vector<SensorsSample> sensors = fallStream(3.0, 3.6, 10.0);

        WHEN("the timeline is built with no stability stream")
        {
            FallDetector falls(sensors, {}, 0.0);

            THEN("the derived detector is in use and finds the fall")
            {
                CHECK_FALSE(falls.usingFlags());
                REQUIRE(falls.intervals().size() == 1);
                CHECK(falls.at(1.0) == Posture::UPRIGHT);
                CHECK(falls.at(5.0) == Posture::FALLEN);
            }
            THEN("the topple itself is covered, not just the time spent flat")
            {
                // The window is grown back to where the tilt last crossed tiltExit,
                // so the frames during the topple are suppressed too.
                CHECK(falls.intervals()[0].start < 3.6);
                CHECK(falls.at(3.5) != Posture::UPRIGHT);
            }
        }
    }

    GIVEN("An upright stream carrying the accelerometer transients of ordinary walking")
    {
        // Walking in the recorded logs spikes the raw accelerometer past 90 deg of
        // apparent tilt for single samples and swings |a| between 3.8 and 19 m/s^2.
        // None of that is a fall, and mistaking it for one would suppress the
        // landmark updates through every footstep.
        std::vector<SensorsSample> sensors;
        int i = 0;
        for (double t = 0.0; t < 20.0; t += 0.01, ++i)
        {
            SensorsSample s = tiltedSample(t, 0.0);
            if (i % 25 == 0)
            {
                s.accelerometer = Eigen::Vector3d(14.0, -6.0, -2.0);   // Impact transient
            }
            else
            {
                s.accelerometer += Eigen::Vector3d(3.0*std::sin(20.0*t), 2.0*std::cos(17.0*t), 4.0*std::sin(13.0*t));
            }
            sensors.push_back(s);
        }

        WHEN("the timeline is built")
        {
            FallDetector falls(sensors, {}, 0.0);

            THEN("no fall is reported")
            {
                CHECK(falls.intervals().empty());
                CHECK(falls.at(10.0) == Posture::UPRIGHT);
            }
        }
    }

    GIVEN("A brief stumble that recovers faster than minFallen")
    {
        std::vector<SensorsSample> sensors;
        for (double t = 0.0; t < 10.0; t += 0.01)
        {
            // Lurches past tiltEnter for only 0.2 s
            const double pitch = (t > 4.0 && t < 4.2) ? 70.0*M_PI/180.0 : 0.0;
            sensors.push_back(tiltedSample(t, pitch));
        }

        WHEN("the timeline is built")
        {
            FallDetector falls(sensors, {}, 0.0);

            THEN("it is not treated as a fall")
            {
                CHECK(falls.intervals().empty());
            }
        }
    }
}

SCENARIO("FallDetector handles logs with no usable data")
{
    GIVEN("No sensors and no flags")
    {
        FallDetector falls({}, {}, 0.0);
        THEN("everything reads upright rather than trapping the filter")
        {
            CHECK(falls.intervals().empty());
            CHECK(falls.at(0.0) == Posture::UPRIGHT);
            CHECK(falls.at(1e6) == Posture::UPRIGHT);
        }
    }
}
