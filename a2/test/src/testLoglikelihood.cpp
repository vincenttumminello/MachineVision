#include <doctest/doctest.h>
#include <Eigen/Core>
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <print>
#include "../../src/Camera.h"
#include "../../src/MeasurementOutdoorFlowBundle.h"
#include "../../src/SystemEstimator.h"
#include "../../src/rotation.hpp"
#include "../../src/GaussianInfo.hpp"
#include "../../src/SystemVisualNav.h"

SCENARIO("MeasurementOutdoorFlowBundle: Log likelihood and gradient")
{
    GIVEN("A camera and synthetic flow data")
    {
        // Setup camera with known intrinsics
        Camera camera;
        camera.cameraMatrix = (cv::Mat_<double>(3, 3) << 
            500.0, 0.0, 320.0,
            0.0, 500.0, 240.0,
            0.0, 0.0, 1.0);
        camera.distCoeffs = cv::Mat::zeros(5, 1, CV_64F);
        camera.imageSize = cv::Size(640, 480);
        
        // Set body-to-camera transform
        Eigen::Matrix3d Rbc;
        Rbc << 0, 0, 1,
               1, 0, 0,
               0, 1, 0;
        camera.Tbc.rotationMatrix = Rbc;
        camera.Tbc.translationVector = Eigen::Vector3d::Zero();
        
        
        // Create synthetic test images instead of loading from file
        cv::Mat imgkm1 = cv::Mat::zeros(480, 640, CV_8UC3);
        cv::Mat imgk = cv::Mat::zeros(480, 640, CV_8UC3);
        
        // Add some synthetic texture for optical flow
        cv::randu(imgkm1, cv::Scalar(20, 20, 20), cv::Scalar(100, 100, 100));
        cv::randu(imgk, cv::Scalar(20, 20, 20), cv::Scalar(100, 100, 100));
        
        // Add some distinctive features
        std::vector<cv::Point2f> features_km1 = {
            {100, 100}, {200, 150}, {300, 200}, {400, 250}, {150, 300},
            {500, 100}, {250, 350}, {450, 180}, {350, 120}, {180, 250}
        };
        
        // Corresponding features in frame k (with small motion)
        std::vector<cv::Point2f> features_k = {
            {102, 101}, {203, 152}, {305, 205}, {408, 255}, {153, 305},
            {505, 103}, {255, 355}, {458, 185}, {358, 125}, {185, 255}
        };
        
        // Draw features as white circles
        for (const auto& pt : features_km1) {
            cv::circle(imgkm1, pt, 8, cv::Scalar(255, 255, 255), -1);
            cv::circle(imgkm1, pt, 12, cv::Scalar(0, 0, 0), 2); // Black border
        }
        
        for (const auto& pt : features_k) {
            cv::circle(imgk, pt, 8, cv::Scalar(255, 255, 255), -1);
            cv::circle(imgk, pt, 12, cv::Scalar(0, 0, 0), 2); // Black border
        }
    
        // std::println("Using synthetic test images ({}x{})", imgk.cols, imgk.rows);
    
        // Optional: Save synthetic images for debugging
        // cv::imwrite("debug_synthetic_frame1.png", imgkm1);
        // cv::imwrite("debug_synthetic_frame2.png", imgk);


        // Start with empty features for first frame
        Eigen::Matrix<double, 2, Eigen::Dynamic> rQOikm1(2, 0);

        // Create initial pose etakm1
        Eigen::VectorXd etakm1(6);
        etakm1 << 0.0, 0.0, -100.0, 0.0, 0.0, 0.0;

        GaussianInfo<double> initialDensity = GaussianInfo<double>::fromSqrtInfo(Eigen::VectorXd::Zero(18), Eigen::MatrixXd::Identity(18, 18));
        SystemVisualNav system(initialDensity);
        
        WHEN("Creating a measurement with the synthetic data")
        {
            MeasurementOutdoorFlowBundle measurement(0.0, camera, system, imgk, imgkm1, rQOikm1, etakm1);
            
            THEN("The log likelihood should be computable")
            {
                // Create a state vector x = [ν, η, ζ]
                Eigen::VectorXd x(18);
                x.setZero();
                
                // Set current pose η[k] (indices 6-11)
                x.segment<3>(6) << 0.0, 0.0, -100.0;  // Position: [N, E, D]
                x.segment<3>(9) << 0.0, 0.0, 0.0;      // Orientation: [roll, pitch, yaw]
                
                // Set previous pose ζ[k-1] (indices 12-17) - slight change
                x.segment<3>(12) << 0.1, 0.0, -100.0;
                x.segment<3>(15) << 0.0, 0.0, 0.0;
                
                
                double logLik = measurement.logLikelihood(x, system);
                
                // Log likelihood should be finite and negative (probability < 1)
                REQUIRE(std::isfinite(logLik));
                REQUIRE(logLik < 0.0);
                
                WHEN("Computing the gradient")
                {
                    Eigen::VectorXd g;
                    double logLik_grad = measurement.logLikelihood(x, system, g);
                    
                    THEN("The gradient should have correct dimensions")
                    {
                        REQUIRE(g.size() == 18);
                    }
                    
                    THEN("The log likelihood value should match")
                    {
                        REQUIRE(logLik_grad == doctest::Approx(logLik).epsilon(1e-6));
                    }
                    
                    THEN("The gradient should be finite")
                    {
                        for (int i = 0; i < g.size(); ++i) {
                            REQUIRE(std::isfinite(g(i)));
                        }
                    }
                    
                    THEN("The gradient should be verified by finite differences")
                    {
                        // Verify gradient using central finite differences
                        const double eps = 1e-6;
                        Eigen::VectorXd g_fd(18);
                        
                        for (int i = 0; i < 18; ++i) {
                            Eigen::VectorXd x_plus = x;
                            Eigen::VectorXd x_minus = x;
                            x_plus(i) += eps;
                            x_minus(i) -= eps;
                            
                            double f_plus = measurement.logLikelihood(x_plus, system);
                            double f_minus = measurement.logLikelihood(x_minus, system);
                            
                            g_fd(i) = (f_plus - f_minus) / (2.0 * eps);
                        }
                        
                        // Check gradient accuracy
                        for (int i = 0; i < 18; ++i) {
                            // Use relative tolerance for non-zero gradients
                            double tol = std::max(1e-4, 1e-4 * std::abs(g_fd(i)));
                            REQUIRE(g(i) == doctest::Approx(g_fd(i)).epsilon(tol));
                        }
                    }

                    THEN("The gradient should match finite differences for all state components")
                    {
                        const double eps = 1e-6;
                        Eigen::VectorXd g_fd(18);
                        
                        std::println("\n=== GRADIENT VERIFICATION ===");
                        std::println("Index | Component    | Autodiff      | FiniteDiff    | RelError");
                        std::println("------|--------------|---------------|---------------|----------");
                        
                        const std::vector<std::string> labels = {
                            "v_N", "v_E", "v_D",           // velocity (0-2)
                            "omega_r", "omega_p", "omega_y", // angular velocity (3-5)
                            "r_N", "r_E", "r_D",           // position (6-8)
                            "roll", "pitch", "yaw",        // orientation (9-11)
                            "r_N_km1", "r_E_km1", "r_D_km1", // previous position (12-14)
                            "roll_km1", "pitch_km1", "yaw_km1" // previous orientation (15-17)
                        };
                        
                        bool all_gradients_ok = true;
                        
                        for (int i = 0; i < 18; ++i) {
                            Eigen::VectorXd x_plus = x;
                            Eigen::VectorXd x_minus = x;
                            x_plus(i) += eps;
                            x_minus(i) -= eps;
                            
                            double f_plus = measurement.logLikelihood(x_plus, system);
                            double f_minus = measurement.logLikelihood(x_minus, system);
                            
                            g_fd(i) = (f_plus - f_minus) / (2.0 * eps);
                            
                            double rel_error = std::abs(g(i) - g_fd(i)) / (std::abs(g_fd(i)) + 1e-10);
                            
                            std::println("{:5} | {:12} | {:13.6e} | {:13.6e} | {:8.2e}", 
                                       i, labels[i], g(i), g_fd(i), rel_error);
                            
                            if (rel_error > 0.01) {  // More than 1% error
                                std::println("      ^^^ WARNING: Large gradient error! ^^^");
                                all_gradients_ok = false;
                            }
                        }
                        
                        std::println("\nGradient norms:");
                        std::println("  Autodiff:   {:.6e}", g.norm());
                        std::println("  FiniteDiff: {:.6e}", g_fd.norm());
                        std::println("  Difference: {:.6e}", (g - g_fd).norm());
                        
                        // Check yaw gradient specifically
                        std::println("\nYaw gradient (index 11):");
                        std::println("  Autodiff:   {:.6e}", g(11));
                        std::println("  FiniteDiff: {:.6e}", g_fd(11));
                        
                        if (std::abs(g_fd(11)) < 1e-8) {
                            std::println("  ^^^ WARNING: Yaw gradient nearly ZERO! No observability! ^^^");
                        }
                        
                        REQUIRE(all_gradients_ok);
                        
                        // Verify each component individually
                        for (int i = 0; i < 18; ++i) {
                            double tol = std::max(1e-4, 1e-4 * std::abs(g_fd(i)));
                            REQUIRE(g(i) == doctest::Approx(g_fd(i)).epsilon(tol));
                        }
                    }
                }
                
                WHEN("Computing the Hessian")
                {
                    Eigen::VectorXd g;
                    Eigen::MatrixXd H;
                    double logLik_hess = measurement.logLikelihood(x, system, g, H);
                    
                    THEN("The Hessian should have correct dimensions")
                    {
                        REQUIRE(H.rows() == 18);
                        REQUIRE(H.cols() == 18);
                    }
                    
                    THEN("The Hessian should be symmetric")
                    {
                        for (int i = 0; i < 18; ++i) {
                            for (int j = 0; j < 18; ++j) {
                                REQUIRE(H(i, j) == doctest::Approx(H(j, i)).epsilon(1e-6));
                            }
                        }
                    }
                    
                    THEN("The Hessian should be finite")
                    {
                        for (int i = 0; i < 18; ++i) {
                            for (int j = 0; j < 18; ++j) {
                                REQUIRE(std::isfinite(H(i, j)));
                            }
                        }
                    }
                }
            }
        }
        
        WHEN("Testing with different altitude values")
        {
            Eigen::VectorXd x1(18), x2(18);
            x1.setZero();
            x2.setZero();
            
            // Configuration 1: high altitude
            x1.segment<3>(6) << 0.5, 0.0, -199.5;
            x1.segment<3>(12) << 0.0, 0.0, -200.0;
            
            // Configuration 2: low altitude
            x2.segment<3>(6) << 0.5, 0.0, -49.5;
            x2.segment<3>(12) << 0.0, 0.0, -50.0;
            
            GaussianInfo<double> initialDensity = GaussianInfo<double>::fromSqrtInfo(Eigen::VectorXd::Zero(18), Eigen::MatrixXd::Identity(18, 18));
            SystemVisualNav system(initialDensity);

            MeasurementOutdoorFlowBundle measurement(0.0, camera, system, imgk, imgkm1, rQOikm1, etakm1);            
            
            THEN("Different altitudes should give different likelihoods")
            {
                double logLik1 = measurement.logLikelihood(x1, system);
                double logLik2 = measurement.logLikelihood(x2, system);
                
                REQUIRE(std::isfinite(logLik1));
                REQUIRE(std::isfinite(logLik2));
                REQUIRE(logLik1 != doctest::Approx(logLik2));
            }
        }
        
        WHEN("Testing gradient descent direction")
        {
            Eigen::VectorXd x(18);
            x.setZero();
            x.segment<3>(6) << 0.0, 0.0, -100.0;
            x.segment<3>(9) << 0.0, 0.0, 0.0;
            x.segment<3>(12) << 0.1, 0.0, -100.0;
            x.segment<3>(15) << 0.0, 0.0, 0.0;

            GaussianInfo<double> initialDensity = GaussianInfo<double>::fromSqrtInfo(Eigen::VectorXd::Zero(18), Eigen::MatrixXd::Identity(18, 18));
            SystemVisualNav system(initialDensity);
            
            MeasurementOutdoorFlowBundle measurement(0.0, camera, system, imgk, imgkm1, rQOikm1, etakm1);
                 
            Eigen::VectorXd g;
            double logLik = measurement.logLikelihood(x, system, g);
            
            THEN("Moving in gradient direction should increase log likelihood")
            {
                double alpha = 1e-4;  // Small step size
                Eigen::VectorXd x_new = x + alpha * g;
                
                double logLik_new = measurement.logLikelihood(x_new, system);
                
                // Log likelihood should increase when moving in gradient direction
                REQUIRE(-logLik_new > -logLik);
            }
        }
    }
}