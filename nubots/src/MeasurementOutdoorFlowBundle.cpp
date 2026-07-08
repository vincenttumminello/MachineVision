
#include <algorithm>
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>
#include <autodiff/forward/utils/derivative.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <cstddef>
#include <print>
#include <iostream>
#include <numeric>
#include <vector>
#include <Eigen/Core>

#include "MeasurementOutdoorFlowBundle.h"
#include "funcmin.hpp"
#include "GaussianInfo.hpp"
#include "rotation.hpp"
#include "SystemEstimator.h"
#include "SystemVisualNav.h"

MeasurementOutdoorFlowBundle::MeasurementOutdoorFlowBundle(double time, const Camera & camera, SystemVisualNav & system, const cv::Mat & imgk_raw, const cv::Mat & imgkm1_raw, const Eigen::Matrix<double, 2, Eigen::Dynamic> & rQOikm1, const Eigen::VectorXd & etakm1)
    : Measurement(time)
    , camera_(camera)
    , rQOikm1_(rQOikm1)
    , rQOik_()
    , rQbarOikm1_()
    , rQbarOik_()
    , mask_()
    , pkm1_()
    , pk_()
    , etakm1_(etakm1)
    , sigma_(1.0) 
{
    system.setFlowEvent();
    
    // updateMethod_ = UpdateMethod::NEWTONTRUSTEIG;
    updateMethod_ = UpdateMethod::BFGSTRUSTSQRT;

    const int divisor               = 2;                // Image scaling factor
    const int maxNumFeatures        = 1000 * 1.5;                // Maximum number of features per frame
    const int minNumFeatures        = 900 * 1.5;                // Minimum number of feature per frame

    cv::TermCriteria termcrit(cv::TermCriteria::COUNT|cv::TermCriteria::EPS,30,0.01);
    cv::Size subPixWinSize(11, 11);     // Window size for subpixel refinement
    cv::Size winSize(21, 21);           // Window size for optical flow

    // Convert images to grayscale
    cv::Mat imgk_gray;
    cv::Mat imgkm1_gray;
    cv::cvtColor(imgk_raw, imgk_gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(imgkm1_raw, imgkm1_gray, cv::COLOR_BGR2GRAY);

    // Scale images
    cv::Mat imgk_scaled;
    cv::Mat imgkm1_scaled;
    cv::resize(imgk_gray, imgk_scaled, cv::Size(), 1.0/divisor, 1.0/divisor);
    cv::resize(imgkm1_gray, imgkm1_scaled, cv::Size(), 1.0/divisor, 1.0/divisor);

    std::vector<cv::Point2f> rQOikm1_scaled;
    if (rQOikm1_.cols() < minNumFeatures)
    {
        // Initialise new features
        rQOikm1_scaled.resize(rQOikm1_.cols());
        for (int j = 0; j < rQOikm1_.cols(); ++j)
        {
            rQOikm1_scaled[j].x = rQOikm1_(0, j)/divisor;
            rQOikm1_scaled[j].y = rQOikm1_(1, j)/divisor;
        }

        // Detect new features in current frame
        std::vector<cv::Point2f> newFeatures;
        int maxNewCorners = maxNumFeatures - rQOikm1_.cols();
        double qualityLevel = 0.001;
        double minDistance = 40.0/divisor;

        // Create a mask to avoid detecting features near existing ones
        cv::Mat mask = cv::Mat::ones(imgk_scaled.size(), CV_8U) * 255;
        for (const auto& pt : rQOikm1_scaled) {
            cv::circle(mask, pt, static_cast<int>(minDistance), cv::Scalar(0), -1);
        }

        cv::goodFeaturesToTrack(imgk_scaled, newFeatures, maxNewCorners, qualityLevel, minDistance, mask);

        if (!newFeatures.empty())
        {
            // Refine new feature locations
            cv::cornerSubPix(imgk_scaled, newFeatures, cv::Size(5,5), cv::Size(-1,-1), termcrit);

            // Add new features to the previous set
            rQOikm1_scaled.insert(rQOikm1_scaled.end(), newFeatures.begin(), newFeatures.end());

            // std::println("Added {} new features (total now: {}).", newFeatures.size(), rQOik.size());
        }
    }
    else
    {
        rQOikm1_scaled.resize(rQOikm1_.cols());
        for (int j = 0; j < rQOikm1_.cols(); ++j)
        {
            rQOikm1_scaled[j].x = rQOikm1_(0, j)/divisor;
            rQOikm1_scaled[j].y = rQOikm1_(1, j)/divisor;
        }
    }

    // Track features from frame k-1 to frame k using the scaled images and previous points
    std::vector<cv::Point2f> rQOik_scaled;
    // TODO: Lab 11
    std::vector<uchar> status;
    std::vector<float> err;

    cv::calcOpticalFlowPyrLK(imgkm1_scaled, imgk_scaled, rQOikm1_scaled, rQOik_scaled, 
                            status, err, winSize, 3, termcrit);

    // Keep points that have been matched between both frames
    int np = 0;
    std::vector<cv::Point2f> rQOik_scaled_filtered;
    std::vector<cv::Point2f> rQOikm1_scaled_filtered;

    for (size_t i = 0; i < status.size(); ++i)
    {
        if (status[i])
        {
            rQOik_scaled_filtered.push_back(rQOik_scaled[i]);
            rQOikm1_scaled_filtered.push_back(rQOikm1_scaled[i]);
            np++;
        }
    }

    rQOik_scaled = rQOik_scaled_filtered;
    rQOikm1_scaled = rQOikm1_scaled_filtered;
    // std::println("After filtering by status, there are {} associations.", np);

    // Calculate where flow points would be in the unscaled image
    rQOik_.resize(2, np);
    rQOikm1_.resize(2, np);
    for (int j = 0; j < np; ++j)
    {
        rQOik_(0, j) = rQOik_scaled[j].x*divisor;
        rQOik_(1, j) = rQOik_scaled[j].y*divisor;

        rQOikm1_(0, j) = rQOikm1_scaled[j].x*divisor;
        rQOikm1_(1, j) = rQOikm1_scaled[j].y*divisor;
    }

    // Calculate the undistorted location of features
    rQbarOik_.resize(2, np);
    rQbarOikm1_.resize(2, np);

    rQbarOik_ = camera_.undistort(rQOik_);
    rQbarOikm1_ = camera_.undistort(rQOikm1_);

    // Use RANSAC to find fundamental matrix and determine inliers (mask_)
    //
    // Note: We don't actually use the fundamental matrix computed here, it is just used to
    //       determine which undistorted flow vectors are consistent with the epipolar constraint.

    // Convert Eigen matrices to std::vector<cv::Point2f> for cv::findFundamentalMat
    std::vector<cv::Point2f> undistortedQOik, undistortedQOikm1;
    for (int i = 0; i < rQbarOik_.cols(); ++i) {
        undistortedQOik.emplace_back(
            static_cast<float>(rQbarOik_(0, i)),
            static_cast<float>(rQbarOik_(1, i))
        );
    }
    for (int i = 0; i < rQbarOikm1_.cols(); ++i) {
        undistortedQOikm1.emplace_back(
            static_cast<float>(rQbarOikm1_(0, i)),
            static_cast<float>(rQbarOikm1_(1, i))
        );
    }

    mask_.resize(np);
    cv::findFundamentalMat(undistortedQOikm1, undistortedQOik, cv::FM_RANSAC, 1.0, 0.99, mask_);

    int nInliers = std::count(mask_.begin(), mask_.end(), true);
    // std::println("No. inliers = {}, No. outliers  = {}", nInliers, mask_.size() - nInliers);

    // Inlier undistorted homogeneous points (pkm1_ and pk_)
    pk_     = Eigen::MatrixXd::Ones(3, nInliers);
    pkm1_   = Eigen::MatrixXd::Ones(3, nInliers);
    int inlierIdx = 0;
    for (int i = 0; i < np; ++i)
    {
        if (mask_[i])
        {
            pkm1_(0, inlierIdx) = rQbarOikm1_(0, i);
            pkm1_(1, inlierIdx) = rQbarOikm1_(1, i);
            pkm1_(2, inlierIdx) = 1.0;
            
            pk_(0, inlierIdx) = rQbarOik_(0, i);
            pk_(1, inlierIdx) = rQbarOik_(1, i);
            pk_(2, inlierIdx) = 1.0;
            
            inlierIdx++;
        }
    }
}

void MeasurementOutdoorFlowBundle::update(SystemBase & system_)
{
    // Downcast since we know that Measurement events only occur to SystemEstimator objects
    SystemEstimator & system = dynamic_cast<SystemEstimator &>(system_);

    const Eigen::Index & nx = system.density.dim();

    // Second-order iterated update
    Eigen::VectorXd g(nx);
    Eigen::VectorXd x = system.density.mean(); // Set initial decision variable to prior mean
    Eigen::MatrixXd Xi = system.density.sqrtInfoMat();

    switch (updateMethod_)
    {
        case UpdateMethod::BFGSTRUSTSQRT: 
        {
            // Create cost function with prototype V = costFunc(x, g)
            auto costFunc = [&](const Eigen::VectorXd & x, Eigen::VectorXd & g){ return costJointDensity(x, system, g); };

            // Minimise cost
            int ret = funcmin::BFGSTrustSqrt(costFunc, x, g, Xi, verbosity_);
            assert(ret == 0);
            break;
        }
        case UpdateMethod::BFGSLMSQRT:
        {
            // Create cost function with prototype V = costFunc(x, g)
            auto costFunc = [&](const Eigen::VectorXd & x, Eigen::VectorXd & g){ return costJointDensity(x, system, g); };

            // Minimise cost
            int ret = funcmin::BFGSLMSqrt(costFunc, x, g, Xi, verbosity_);
            assert(ret == 0);
            break;
        }
        case UpdateMethod::SR1TRUSTEIG:
        {
            // Generate eigendecomposition of initial Hessian (prior information matrix)
            // via an SVD of Xi = U*D*V.', i.e., Xi.'*Xi = V*D*U.'*U*D*V.' = V*D^2*V.'
            // This avoids the loss of precision associated with directly computing the eigendecomposition of Xi.'*Xi
            Eigen::JacobiSVD<Eigen::MatrixXd> svd(Xi, Eigen::ComputeFullV);
            Eigen::MatrixXd Q = svd.matrixV();
            Eigen::VectorXd v = svd.singularValues().array().square();

            assert(Q.rows() == nx);
            assert(Q.cols() == nx);
            assert(v.size() == nx);

            // Foreshadowing: If we were doing landmark SLAM with a quasi-Newton method,
            //                we can purposely introduce negative eigenvalues for newly
            //                initialised landmarks to force the Hessian and hence
            //                posterior sqrt information matrix to be approximated correctly.

            // Create cost function with prototype V = costFunc(x, g)
            auto costFunc = [&](const Eigen::VectorXd & x, Eigen::VectorXd & g){ return costJointDensity(x, system, g); };

            // Minimise cost
            int ret = funcmin::SR1TrustEig(costFunc, x, g, Q, v, verbosity_);
            assert(ret == 0);

            // Post-calculate posterior square-root information matrix from Hessian eigendecomposition
            Xi = v.array().sqrt().matrix().asDiagonal()*Q.transpose();
            Eigen::HouseholderQR<Eigen::Ref<Eigen::MatrixXd>> qr(Xi);           // In-place QR decomposition
            Xi = Xi.triangularView<Eigen::Upper>();                             // Safe aliasing
            break;
        }
        case UpdateMethod::NEWTONTRUSTEIG:
        {
            // Create cost function with prototype V = costFunc(x, g, H)
            auto costFunc = [&](const Eigen::VectorXd & x, Eigen::VectorXd & g, Eigen::MatrixXd & H){ return costJointDensity(x, system, g, H); };

            // Minimise cost
            Eigen::MatrixXd Q(nx, nx);
            Eigen::VectorXd v(nx);
            int ret = funcmin::NewtonTrustEig(costFunc, x, g, Q, v, verbosity_);
            assert(ret == 0);

            // Post-calculate posterior square-root information matrix from Hessian eigendecomposition
            Xi = v.array().sqrt().matrix().asDiagonal()*Q.transpose();
            Eigen::HouseholderQR<Eigen::Ref<Eigen::MatrixXd>> qr(Xi);           // In-place QR decomposition
            Xi = Xi.triangularView<Eigen::Upper>();                             // Safe aliasing
            break;
        }
        default:
            throw std::invalid_argument("Invalid update method");
    }

    // Set posterior mean to maximum a posteriori (MAP) estimate
    Eigen::VectorXd mu = x;
    system.density = GaussianInfo<double>::fromSqrtInfo(Xi*mu, Xi);
}

Eigen::VectorXd MeasurementOutdoorFlowBundle::simulate(const Eigen::VectorXd & x, const SystemEstimator & system) const
{
    Eigen::VectorXd y;
    throw std::runtime_error("Not implemented");
    return y;
}

const Eigen::Matrix<double, 2, Eigen::Dynamic> & MeasurementOutdoorFlowBundle::trackedPreviousFeatures() const
{
    return rQOikm1_;
}

const Eigen::Matrix<double, 2, Eigen::Dynamic> & MeasurementOutdoorFlowBundle::trackedCurrentFeatures() const
{
    return rQOik_;
}

const std::vector<unsigned char> & MeasurementOutdoorFlowBundle::inlierMask() const
{
    return mask_;
}

double MeasurementOutdoorFlowBundle::logLikelihood(const Eigen::VectorXd & x, const SystemEstimator & system) const
{
    return logLikelihoodImpl(x);
}

double MeasurementOutdoorFlowBundle::logLikelihood(const Eigen::VectorXd & x, const SystemEstimator & system, Eigen::VectorXd & g) const
{
    // std::cout << "Body rotation passed into logLikelihood: \n" << x.segment<3>(9).transpose() << std::endl;

    // Evaluate gradient for Newton and quasi-Newton methods
    g.resize(x.size());
    
    using autodiff::dual;
    using autodiff::gradient;
    using autodiff::wrt;
    using autodiff::at;

    // convert x to dual (no casting back to double anywhere!)
    Eigen::Matrix<dual, Eigen::Dynamic, 1> xdual = x.cast<dual>();

    // result (autodiff::dual will hold the function value)
    dual fdual;

    // Lambda must call the templated member function instantiated for 'dual'.
    // Use 'this->template logLikelihoodImpl<dual>' to avoid C++ parsing issues.
    auto func = [this](const Eigen::Matrix<dual, Eigen::Dynamic, 1>& xd) -> dual {
        return this->template logLikelihoodImpl<dual>(xd);
    };

    // Compute gradient. gradient(...) returns an Eigen::VectorXd (double) of derivatives.
    // We capture the returned Eigen vector and then move it into g.
    Eigen::VectorXd grad = gradient(func, wrt(xdual), at(xdual), fdual);

    // fill output gradient
    g = grad;

    // return scalar log-likelihood value as double
    return static_cast<double>(fdual);
}

double MeasurementOutdoorFlowBundle::logLikelihood(const Eigen::VectorXd & x, const SystemEstimator & system, Eigen::VectorXd & g, Eigen::MatrixXd & H) const
{
    // Evaluate Hessian for Newton method
    g.resize(x.size());
    H.resize(x.size(), x.size());
    
    using autodiff::dual2nd;
    using autodiff::hessian;
    using autodiff::wrt;
    using autodiff::at;

    // Convert x to dual2nd for second-order derivatives
    Eigen::Matrix<dual2nd, Eigen::Dynamic, 1> xdual = x.cast<dual2nd>();
    
    // Result (dual2nd will hold the function value)
    dual2nd fdual;

    // Lambda calls the templated logLikelihoodImpl for dual2nd
    auto func = [this](const Eigen::Matrix<dual2nd, Eigen::Dynamic, 1>& xd) -> dual2nd {
        return this->template logLikelihoodImpl<dual2nd>(xd);
    };

    // Compute Hessian and gradient simultaneously
    // hessian(...) returns an Eigen::MatrixXd for H and fills g with gradient
    H = hessian(func, wrt(xdual), at(xdual), fdual, g);

    // Return scalar log-likelihood value as double
    return static_cast<double>(fdual);
}

Eigen::Matrix<double, 2, Eigen::Dynamic> MeasurementOutdoorFlowBundle::predictedFeatures(const Eigen::VectorXd & x, const SystemEstimator & system) const
{
    std::size_t np = rQOik_.cols();

    // Predict undistorted homogeneous image points
    Eigen::Matrix<double, 3, Eigen::Dynamic> pkm1(3, np);
    Eigen::Matrix<double, 3, Eigen::Dynamic> pk(3, np);
    pkm1.topRows<2>() = rQbarOikm1_;
    pkm1.row(2).setOnes();
    pk.topRows<2>() = rQbarOik_;
    pk.row(2).setOnes();

    Eigen::Matrix<double, 3, Eigen::Dynamic> pk_hat = predictFlowImpl(x, pkm1, pk);
    
    // Convert to Cartesian with division protection
    Eigen::Matrix<double, 2, Eigen::Dynamic> rQbarOik_hat(2, np);
    const double eps = 1e-8;
    for (int j = 0; j < np; ++j) {
        double w = pk_hat(2, j);
        if (std::abs(w) < eps) {
            w = (w >= 0) ? eps : -eps;
        }
        rQbarOik_hat(0, j) = pk_hat(0, j) / w;
        rQbarOik_hat(1, j) = pk_hat(1, j) / w;
    }
    
    // Apply lens distortion using camera's distort method
    Eigen::Matrix<double, 2, Eigen::Dynamic> rQOik_hat = camera_.distort(rQbarOik_hat);
    
    return rQOik_hat;
}

