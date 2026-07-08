#include <doctest/doctest.h>
#include <cmath>
#include <vector>
#include <Eigen/Core>
#include "../../src/FieldMap.h"
#include "../../src/GaussianInfo.hpp"
#include "../../src/MeasurementFieldLandmarks.h"
#include "../../src/Pose.hpp"
#include "../../src/rotation.hpp"
#include "../../src/SensorLog.h"
#include "../../src/SystemLocalisation.h"

// Build a synthetic detection whose relevant rays equal the true ray to a landmark
// seen from camera pose Tfc (all four corners set to the ray; goal posts use BR/BL,
// intersections use the box centre, so both reduce to the same ray).
static Detection makeDetection(const std::string & name, const Eigen::Vector3d & rLFf, const Pose<double> & Tfc)
{
    Eigen::Vector3d rLCc = Tfc.rotationMatrix.transpose()*(rLFf - Tfc.translationVector);
    Eigen::Vector3d u = rLCc.normalized();
    Detection det;
    det.name = name;
    det.confidence = 0.9;
    det.corners << u, u, u, u;
    return det;
}

SCENARIO("MeasurementFieldLandmarks association and MAP update")
{
    GIVEN("A robot near the centre of the field looking towards the +x goal")
    {
        FieldMap map;

        // True torso pose: 1 m behind centre, 0.45 m torso height, facing +x
        // (camera mount bias states zero)
        Eigen::VectorXd etaTrue = Eigen::VectorXd::Zero(SystemLocalisation::nx);
        etaTrue.head<6>() << -1.0, 0.2, 0.45, 0.0, 0.0, 0.1;

        // Camera 0.1 m above torso origin, aligned with torso
        Pose<double> Tbc;
        Tbc.translationVector = Eigen::Vector3d(0.05, 0, 0.1);

        Pose<double> Tfb = SystemLocalisation::fieldPose<double>(Eigen::VectorXd(etaTrue));
        Pose<double> Tfc = Tfb*Tbc;

        // Detections of the two +x goal posts and the +x penalty mark
        VisionSample sample;
        sample.t = 0;
        sample.videoFrame = -1;
        const auto & posts = map.landmarks(LandmarkType::GOAL_POST);
        const auto & xmarks = map.landmarks(LandmarkType::X_INTERSECTION);
        for (const auto & post : posts)
        {
            if (post.x() > 0)
            {
                sample.detections.push_back(makeDetection("goal post", post, Tfc));
            }
        }
        for (const auto & xm : xmarks)
        {
            if (xm.x() > 1.0)
            {
                sample.detections.push_back(makeDetection("X-intersection", xm, Tfc));
            }
        }
        // Penalty-area and goal-area corners on the +x half: wide bearing spread
        for (const auto & lm : map.landmarks(LandmarkType::L_INTERSECTION))
        {
            if (lm.x() > 1.5 && lm.x() < 3.0)
            {
                sample.detections.push_back(makeDetection("L-intersection", lm, Tfc));
            }
        }
        REQUIRE(sample.detections.size() == 7);

        std::vector<BodyTwistSample> twists;    // Empty: no motion

        WHEN("the prior is centred at the true pose")
        {
            auto p0 = GaussianInfo<double>::fromSqrtMoment(etaTrue, Eigen::MatrixXd(Eigen::MatrixXd::Identity(SystemLocalisation::nx, SystemLocalisation::nx)*0.1));
            SystemLocalisation system(p0, twists);
            MeasurementFieldLandmarks meas(0.0, sample, Tbc, map, system);

            THEN("all detections associate")
            {
                CHECK(meas.numCandidates() == 7);
                CHECK(meas.numAssociated() == 7);
            }

            THEN("the log likelihood peaks at the true pose")
            {
                double llTrue = meas.logLikelihood(etaTrue, system);
                Eigen::VectorXd etaPerturbed = etaTrue;
                etaPerturbed(0) += 0.2;
                etaPerturbed(5) += 0.05;
                CHECK(llTrue > meas.logLikelihood(etaPerturbed, system));
            }

            THEN("the gradient matches finite differences")
            {
                Eigen::VectorXd x = etaTrue;
                x(0) += 0.05;
                x(4) += 0.02;
                Eigen::VectorXd g;
                meas.logLikelihood(x, system, g);
                const double h = 1e-6;
                for (int j = 0; j < SystemLocalisation::nx; ++j)
                {
                    Eigen::VectorXd xp = x, xm = x;
                    xp(j) += h;
                    xm(j) -= h;
                    double dfd = (meas.logLikelihood(xp, system) - meas.logLikelihood(xm, system))/(2*h);
                    CHECK(g(j) == doctest::Approx(dfd).epsilon(1e-3));
                }
            }
        }

        WHEN("the prior is offset from the true pose")
        {
            Eigen::VectorXd etaPrior = etaTrue;
            etaPrior(0) += 0.15;
            etaPrior(1) -= 0.1;
            etaPrior(5) += 0.05;

            Eigen::MatrixXd Sp = Eigen::MatrixXd::Identity(SystemLocalisation::nx, SystemLocalisation::nx)*0.2;
            Sp.diagonal().tail<2>().setConstant(0.02);  // Tight camera-bias prior: rays here are exact
            auto p0 = GaussianInfo<double>::fromSqrtMoment(etaPrior, Sp);
            SystemLocalisation system(p0, twists);
            // The synthetic rays are exact, so use a correspondingly tight noise model
            MeasurementFieldLandmarks::Options options;
            options.sigmaAngular = 0.005;
            MeasurementFieldLandmarks meas(0.0, sample, Tbc, map, system, options);
            REQUIRE(meas.numAssociated() == 7);

            THEN("the MAP update recovers the true pose and reports evidence")
            {
                double errBefore = (system.density.mean() - etaTrue).head(2).norm();
                meas.process(system);
                Eigen::VectorXd err = system.density.mean() - etaTrue;
                CHECK(err.head(2).norm() < errBefore);
                CHECK(err.head(2).norm() < 0.05);  // Wide bearing spread pins the pose
                CHECK(std::abs(err(5)) < 0.02);    // Yaw corrected
                CHECK(std::isfinite(meas.logEvidence()));

                // Posterior covariance shrinks relative to prior in observed directions
                CHECK(system.density.cov()(5, 5) < 0.2*0.2);
            }
        }
    }
}
