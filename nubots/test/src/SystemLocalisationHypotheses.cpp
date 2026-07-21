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

// Same synthetic-detection helper as the MeasurementFieldLandmarks test: the
// four corner rays all equal the true unit ray to the landmark from pose Tfc.
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

SCENARIO("SystemLocalisation mirror transform")
{
    GIVEN("An arbitrary 8-DOF state")
    {
        Eigen::VectorXd x(SystemLocalisation::nx);
        x << 1.3, -0.7, 0.45, 0.02, -0.03, 0.6, 0.01, -0.02;

        WHEN("it is mirrored across the field centre")
        {
            Eigen::VectorXd y = SystemLocalisation::mirrorState(x);

            THEN("horizontal position is negated and yaw offset by pi")
            {
                CHECK(y(0) == doctest::Approx(-1.3));
                CHECK(y(1) == doctest::Approx(0.7));
                CHECK(y(2) == doctest::Approx(0.45));           // height unchanged
                CHECK(y(3) == doctest::Approx(0.02));           // roll unchanged
                CHECK(y(4) == doctest::Approx(-0.03));          // pitch unchanged
                CHECK(std::abs(std::remainder(y(5) - (0.6 + M_PI), 2.0*M_PI)) < 1e-9);
                CHECK(y(6) == doctest::Approx(0.01));           // camera bias unchanged
                CHECK(y(7) == doctest::Approx(-0.02));
            }

            THEN("mirroring twice returns the original pose")
            {
                Eigen::VectorXd z = SystemLocalisation::mirrorState(y);
                CHECK((z.head<5>() - x.head<5>()).norm() < 1e-9);
                CHECK(std::abs(std::remainder(z(5) - x(5), 2.0*M_PI)) < 1e-9);
                CHECK((z.tail<2>() - x.tail<2>()).norm() < 1e-9);
            }
        }

        WHEN("a density is mirrored")
        {
            Eigen::MatrixXd S = Eigen::MatrixXd::Zero(SystemLocalisation::nx, SystemLocalisation::nx);
            S.diagonal() << 0.3, 0.2, 0.05, 0.05, 0.05, 0.15, 0.05, 0.05;
            S(0, 5) = 0.1;    // Cross-covariance between x-position and yaw
            auto g = GaussianInfo<double>::fromSqrtMoment(Eigen::VectorXd(x), S);

            auto gm = SystemLocalisation::mirrorDensity(g);

            THEN("the mean is the mirrored state")
            {
                Eigen::VectorXd mu = gm.mean();
                CHECK((mu - SystemLocalisation::mirrorState(x)).head<5>().norm() < 1e-9);
            }

            THEN("variances are preserved and the x-yaw covariance flips sign")
            {
                Eigen::MatrixXd P = g.cov();
                Eigen::MatrixXd Pm = gm.cov();
                CHECK(Pm(0, 0) == doctest::Approx(P(0, 0)).epsilon(1e-6));   // var(x) unchanged
                CHECK(Pm(5, 5) == doctest::Approx(P(5, 5)).epsilon(1e-6));   // var(yaw) unchanged
                CHECK(Pm(0, 5) == doctest::Approx(-P(0, 5)).epsilon(1e-6));  // sign flip (x negated, yaw not)
            }
        }
    }
}

SCENARIO("Hypothesis bank + out-of-field evidence resolve the field symmetry")
{
    GIVEN("A robot behind the centre looking towards the +x goal with a mirror hypothesis")
    {
        FieldMap map;

        Eigen::VectorXd etaTrue = Eigen::VectorXd::Zero(SystemLocalisation::nx);
        etaTrue.head<6>() << -1.0, 0.2, 0.45, 0.0, 0.0, 0.1;
        const Eigen::VectorXd etaMirror = SystemLocalisation::mirrorState(etaTrue);

        Pose<double> Tbc;
        Tbc.translationVector = Eigen::Vector3d(0.05, 0, 0.1);
        Pose<double> Tfc = SystemLocalisation::fieldPose<double>(Eigen::VectorXd(etaTrue))*Tbc;

        // Seven asymmetric +x-half detections (goal posts, penalty mark, corners)
        VisionSample sample;
        sample.t = 0;
        sample.videoFrame = -1;
        for (const auto & post : map.landmarks(LandmarkType::GOAL_POST))
            if (post.x() > 0) sample.detections.push_back(makeDetection("goal post", post, Tfc));
        for (const auto & xm : map.landmarks(LandmarkType::X_INTERSECTION))
            if (xm.x() > 1.0) sample.detections.push_back(makeDetection("X-intersection", xm, Tfc));
        for (const auto & lm : map.landmarks(LandmarkType::L_INTERSECTION))
            if (lm.x() > 1.5 && lm.x() < 3.0) sample.detections.push_back(makeDetection("L-intersection", lm, Tfc));
        REQUIRE(sample.detections.size() == 7);

        std::vector<BodyTwistSample> twists;

        // Prior centred on the true pose (as if from a trusted initial baseline)
        Eigen::MatrixXd S0 = Eigen::MatrixXd::Identity(SystemLocalisation::nx, SystemLocalisation::nx)*0.2;
        S0.diagonal().tail<2>().setConstant(0.02);
        auto p0 = GaussianInfo<double>::fromSqrtMoment(etaTrue, S0);
        SystemLocalisation system(p0, twists);

        MeasurementFieldLandmarks::Options options;
        options.sigmaAngular = 0.02;

        WHEN("two mirror hypotheses are seeded and a landmark measurement is applied")
        {
            system.initialiseHypotheses();
            REQUIRE(system.numHypotheses() == 2);

            MeasurementFieldLandmarks meas(0.0, sample, Tbc, map, system, options);
            system.process(meas);

            THEN("on-field landmarks leave the two hypotheses equally weighted")
            {
                // The field is symmetric: re-associated at its own pose the mirror
                // fits the mirror-partner landmarks exactly as well as the true pose
                // fits the originals. Landmark evidence therefore CANNOT separate
                // them -- the belief must stay a genuine two-mode 50/50 mixture.
                REQUIRE(system.numHypotheses() == 2);
                auto w = system.hypothesisWeights();
                CHECK(w[0] == doctest::Approx(0.5).epsilon(0.05));
                CHECK(w[1] == doctest::Approx(0.5).epsilon(0.05));

                // Both the true pose and its mirror are present among the components
                const auto & comps = system.hypotheses();
                double dTrue = std::min((comps[0].mean() - etaTrue).head<2>().norm(),
                                        (comps[1].mean() - etaTrue).head<2>().norm());
                double dMirror = std::min((comps[0].mean() - etaMirror).head<2>().norm(),
                                          (comps[1].mean() - etaMirror).head<2>().norm());
                CHECK(dTrue < 0.15);
                CHECK(dMirror < 0.15);
            }
        }

        WHEN("out-of-field evidence favouring the representative is then folded in")
        {
            system.initialiseHypotheses();
            MeasurementFieldLandmarks meas(0.0, sample, Tbc, map, system, options);
            system.process(meas);
            REQUIRE(system.numHypotheses() == 2);

            // The representative is the true pose (equal weights => the first
            // component, which initialiseHypotheses seeds as the current density).
            REQUIRE((system.density.mean() - etaTrue).head<2>().norm() < 0.15);

            // A few frames of decisive own-over-mirror background evidence.
            for (int k = 0; k < 5; ++k) system.addSideLogEvidence(1.5);

            THEN("the true hypothesis dominates and the mirror is pruned")
            {
                auto w = system.hypothesisWeights();
                CHECK(*std::max_element(w.begin(), w.end()) > 0.9);
                Eigen::VectorXd mu = system.density.mean();
                CHECK((mu.head<2>() - etaTrue.head<2>()).norm() < 0.15);
                CHECK(std::abs(std::remainder(mu(5) - etaTrue(5), 2.0*M_PI)) < 0.15);
            }
        }

        WHEN("out-of-field evidence contradicts the representative")
        {
            system.initialiseHypotheses();
            MeasurementFieldLandmarks meas(0.0, sample, Tbc, map, system, options);
            system.process(meas);
            REQUIRE((system.density.mean() - etaTrue).head<2>().norm() < 0.15);

            // Decisive evidence AGAINST the current representative hands leadership
            // to the mirror pose -- the smooth, weight-driven equivalent of a flip.
            system.addSideLogEvidence(-8.0);

            THEN("leadership passes to the mirror pose")
            {
                Eigen::VectorXd mu = system.density.mean();
                CHECK((mu.head<2>() - etaMirror.head<2>()).norm() < 0.15);
                CHECK(std::abs(std::remainder(mu(5) - etaMirror(5), 2.0*M_PI)) < 0.15);
            }
        }

        WHEN("no hypotheses are active")
        {
            MeasurementFieldLandmarks meas(0.0, sample, Tbc, map, system, options);

            THEN("system.process matches a direct single-Gaussian update")
            {
                // Reference: direct Event::process on an identical system
                SystemLocalisation ref(p0, twists);
                MeasurementFieldLandmarks measRef(0.0, sample, Tbc, map, ref, options);
                measRef.process(ref);

                REQUIRE(system.numHypotheses() == 1);
                system.process(meas);   // Should be identical to the direct path

                Eigen::VectorXd a = system.density.mean();
                Eigen::VectorXd b = ref.density.mean();
                CHECK((a - b).norm() < 1e-9);
            }
        }
    }
}
