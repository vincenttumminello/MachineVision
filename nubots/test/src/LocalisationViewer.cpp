#include <doctest/doctest.h>
#include <vector>
#include <Eigen/Core>
#include <opencv2/core.hpp>
#include "../../src/FieldMap.h"
#include "../../src/FisheyeLens.h"
#include "../../src/LocalisationViewer.h"
#include "../../src/Pose.hpp"

// Build one plausible frame: robot 1 m behind centre facing +x, camera level,
// seeing a +x goal post and a field-line point.
static ViewerFrame makeFrame(const FieldMap & map)
{
    ViewerFrame vf;
    vf.videoFrame = 0;
    vf.t = 1.0;

    // Camera at (-1, 0, 0.45), optical axis along field +x (Rfc = identity)
    vf.Tfc.rotationMatrix = Eigen::Matrix3d::Identity();
    vf.Tfc.translationVector = Eigen::Vector3d(-1.0, 0.0, 0.45);
    const Eigen::Matrix3d Rcf = vf.Tfc.rotationMatrix.transpose();
    const Eigen::Vector3d rCFf = vf.Tfc.translationVector;

    vf.estPos = Eigen::Vector2d(-1.0, 0.0);
    vf.estYaw = 0.0;
    vf.estCov = (Eigen::Matrix2d() << 0.04, 0.0, 0.0, 0.02).finished();

    HypothesisView h;
    h.pos = vf.estPos; h.yaw = vf.estYaw; h.cov = vf.estCov; h.weight = 1.0;
    vf.hypotheses.push_back(h);

    // Goal post detection (all four corners set to the ray to the post)
    const Eigen::Vector3d post = map.landmarks(LandmarkType::GOAL_POST).front();
    const Eigen::Vector3d ray = (Rcf*(post - rCFf)).normalized();
    DetectionView d;
    d.name = "goal post"; d.confidence = 0.9; d.used = true;
    d.corners << ray, ray, ray, ray;
    vf.detections.push_back(d);

    AssociationView a;
    a.measRay = ray; a.landmark = post;
    vf.associations.push_back(a);

    // One downward field-line ray
    vf.lineRays.resize(3, 1);
    vf.lineRays.col(0) = Eigen::Vector3d(3.0, 0.0, -0.6).normalized();

    vf.hasBaseline = true;
    vf.basePos = Eigen::Vector2d(-1.05, 0.02);
    vf.baseYaw = 0.03;
    vf.hasError = true; vf.errXY = 0.05; vf.errYaw = 0.03;
    vf.nAssoc = 1; vf.nCand = 1; vf.updateMs = 0.3;
    return vf;
}

SCENARIO("LocalisationViewer renders a composite without a display")
{
    FieldMap map;
    FisheyeLens lens;
    LocalisationViewer viewer(map, lens, "unused.mp4");

    GIVEN("A recorded frame and a blank camera image")
    {
        std::vector<ViewerFrame> frames{makeFrame(map)};

        WHEN("the composite is rendered")
        {
            cv::Mat composite = viewer.renderComposite(frames, 0, cv::Mat());

            THEN("a non-empty 3-channel image large enough for both panels is produced")
            {
                REQUIRE_FALSE(composite.empty());
                CHECK(composite.type() == CV_8UC3);
                CHECK(composite.rows > 700);    // header + panel height
                CHECK(composite.cols > 1200);   // camera panel + top-down panel
            }
        }

        WHEN("a real camera image is supplied")
        {
            cv::Mat raw(1024, 1280, CV_8UC3, cv::Scalar(60, 60, 60));
            cv::Mat composite = viewer.renderComposite(frames, 0, raw);

            THEN("the render succeeds")
            {
                REQUIRE_FALSE(composite.empty());
                CHECK(composite.rows > 700);
            }
        }
    }

    GIVEN("A multi-hypothesis frame")
    {
        std::vector<ViewerFrame> frames{makeFrame(map)};
        // Add the 180 deg mirror as a second, lower-weight hypothesis
        HypothesisView mirror;
        mirror.pos = Eigen::Vector2d(1.0, 0.0);
        mirror.yaw = M_PI;
        mirror.cov = frames[0].estCov;
        mirror.weight = 0.2;
        frames[0].hypotheses[0].weight = 0.8;
        frames[0].hypotheses.push_back(mirror);

        THEN("both hypotheses render without error")
        {
            cv::Mat composite = viewer.renderComposite(frames, 0, cv::Mat());
            REQUIRE_FALSE(composite.empty());
        }
    }
}
