#include <doctest/doctest.h>
#include <algorithm>
#include <cmath>
#include <vector>
#include <Eigen/Core>

#include "../../src/FieldMap.h"

namespace
{
    // Returns true if a landmark with position (-x, -y, z) exists in landmarks
    bool hasMirror(const std::vector<Eigen::Vector3d> & landmarks, const Eigen::Vector3d & p)
    {
        const Eigen::Vector3d mirrored(-p.x(), -p.y(), p.z());
        return std::any_of(landmarks.begin(), landmarks.end(), [&](const Eigen::Vector3d & q)
        {
            return (q - mirrored).norm() < 1e-9;
        });
    }
}

SCENARIO("FieldMap default dimensions")
{
    GIVEN("A FieldMap constructed with default field dimensions")
    {
        FieldMap map;

        WHEN("Querying the L_INTERSECTION landmarks")
        {
            const std::vector<Eigen::Vector3d> & lLandmarks = map.landmarks(LandmarkType::L_INTERSECTION);
            THEN("There are exactly 12 landmarks")
            {
                REQUIRE(lLandmarks.size() == 12);
            }
        }

        WHEN("Querying the T_INTERSECTION landmarks")
        {
            const std::vector<Eigen::Vector3d> & tLandmarks = map.landmarks(LandmarkType::T_INTERSECTION);
            THEN("There are exactly 10 landmarks")
            {
                REQUIRE(tLandmarks.size() == 10);
            }
        }

        WHEN("Querying the X_INTERSECTION landmarks")
        {
            const std::vector<Eigen::Vector3d> & xLandmarks = map.landmarks(LandmarkType::X_INTERSECTION);
            THEN("There are exactly 5 landmarks")
            {
                REQUIRE(xLandmarks.size() == 5);
            }
        }

        WHEN("Querying the GOAL_POST landmarks")
        {
            const std::vector<Eigen::Vector3d> & goalLandmarks = map.landmarks(LandmarkType::GOAL_POST);
            THEN("There are exactly 4 landmarks")
            {
                REQUIRE(goalLandmarks.size() == 4);

                AND_THEN("Each goal post is at x = +-fieldLength/2 and y = +-goalWidth/2")
                {
                    for (const Eigen::Vector3d & p : goalLandmarks)
                    {
                        CHECK(std::abs(std::abs(p.x()) - map.dims.fieldLength/2) < 1e-9);
                        CHECK(std::abs(std::abs(p.y()) - map.dims.goalWidth/2) < 1e-9);
                    }
                }
            }
        }

        WHEN("Considering every landmark type")
        {
            const std::vector<LandmarkType> types = {
                LandmarkType::L_INTERSECTION,
                LandmarkType::T_INTERSECTION,
                LandmarkType::X_INTERSECTION,
                LandmarkType::GOAL_POST
            };

            THEN("Every landmark lies on the ground plane (z == 0)")
            {
                for (const LandmarkType & type : types)
                {
                    for (const Eigen::Vector3d & p : map.landmarks(type))
                    {
                        CHECK(p.z() == 0.0);
                    }
                }
            }

            THEN("Every landmark's 180 degree mirror is present in the same type set")
            {
                for (const LandmarkType & type : types)
                {
                    const std::vector<Eigen::Vector3d> & landmarks = map.landmarks(type);
                    for (const Eigen::Vector3d & p : landmarks)
                    {
                        CAPTURE(static_cast<int>(type));
                        CAPTURE(p.x());
                        CAPTURE(p.y());
                        CHECK(hasMirror(landmarks, p));
                    }
                }
            }

            THEN("Every landmark lies within the border strip bounds")
            {
                const double xBound = map.dims.fieldLength/2 + map.dims.borderStripMinWidth;
                const double yBound = map.dims.fieldWidth/2 + map.dims.borderStripMinWidth;
                for (const LandmarkType & type : types)
                {
                    for (const Eigen::Vector3d & p : map.landmarks(type))
                    {
                        CHECK(std::abs(p.x()) <= xBound);
                        CHECK(std::abs(p.y()) <= yBound);
                    }
                }
            }
        }
    }
}
