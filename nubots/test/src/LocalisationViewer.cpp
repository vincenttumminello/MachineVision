#include <doctest/doctest.h>
#include <vector>
#include <Eigen/Core>
#include <opencv2/core.hpp>
#include "../../src/FieldMap.h"
#include "../../src/FisheyeLens.h"
#include "../../src/LocalisationViewer.h"
#include "../../src/Pose.hpp"
#include "../../src/SideDisambiguator.h"

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
    d.name = "goal post"; d.confidence = 0.9; d.status = DetectionStatus::ASSOCIATED;
    d.corners << ray, ray, ray, ray;
    vf.detections.push_back(d);

    // A second detection the estimator could not associate, so the camera panel
    // exercises both branches of the association colour key.
    DetectionView unassoc;
    unassoc.name = "L-intersection"; unassoc.confidence = 0.8;
    unassoc.status = DetectionStatus::UNASSOCIATED;
    const Eigen::Vector3d stray = Eigen::Vector3d(3.0, 1.0, -0.5).normalized();
    unassoc.corners << stray, stray, stray, stray;
    vf.detections.push_back(unassoc);

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

    // 3D-panel data: full position/covariance, mocap truth and one mapped
    // out-of-field landmark
    vf.estPos3 = Eigen::Vector3d(-1.0, 0.0, 0.44);
    vf.estCov3 = (Eigen::Matrix3d() << 0.04, 0.0, 0.0,
                                       0.0, 0.02, 0.0,
                                       0.0, 0.0, 0.001).finished();
    vf.hypotheses[0].pos3 = vf.estPos3;
    vf.hypotheses[0].cov3 = vf.estCov3;
    vf.hasTruth = true;
    vf.truthPos = Eigen::Vector3d(-0.95, 0.05, 0.50);
    vf.truthYaw = 0.02;
    vf.oofLandmarks.push_back({Eigen::Vector3d(6.0, 2.0, 1.5),
                               Eigen::Matrix3d::Identity()*0.01, false});
    vf.oofLandmarks.push_back({Eigen::Vector3d(-7.0, -3.0, 2.0),
                               Eigen::Matrix3d::Identity()*4.0, true});

    // Out-of-field corners and projected map landmarks, one per status, so the
    // camera panel's association colour key is exercised end to end.
    vf.oofVisibleMargin = 30.0;
    for (int s = SideDisambiguator::FEATURE_ON_CARPET; s <= SideDisambiguator::FEATURE_ASSOCIATED; ++s)
    {
        vf.oofFeatures.push_back({Eigen::Vector2d(200.0 + 40.0*s, 300.0), s});
    }
    for (int s = SideDisambiguator::LANDMARK_EDGE; s <= SideDisambiguator::LANDMARK_CULLED_OUTLIER; ++s)
    {
        vf.oofLandmarkProj.push_back({Eigen::Vector2d(200.0 + 40.0*s, 380.0),
                                      Eigen::Vector2d(210.0 + 40.0*s, 300.0), s, s % 2 == 0});
    }
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

        WHEN("the right pane is switched to the 3D map")
        {
            viewer.setRightPane3D(true);
            cv::Mat composite3d = viewer.renderComposite(frames, 0, cv::Mat());
            viewer.setRightPane3D(false);
            cv::Mat composite2d = viewer.renderComposite(frames, 0, cv::Mat());

            THEN("the 3D composite renders at the same size as the top-down one")
            {
                REQUIRE_FALSE(composite3d.empty());
                CHECK(composite3d.rows == composite2d.rows);
                CHECK(composite3d.cols == composite2d.cols);
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
