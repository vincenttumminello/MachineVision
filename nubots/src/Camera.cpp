#include <cassert>
#include <cstddef>
#include <cmath>
#include <stdexcept>
#include <limits>
#include <format>
#include <vector>
#include <filesystem>
#include <regex>
#include <print>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/persistence.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/calib3d.hpp>
#include "to_string.hpp"
#include "rotation.hpp"
#include "Pose.hpp"
#include "Camera.h"

void Chessboard::write(cv::FileStorage & fs) const
{
    fs << "{"
       << "grid_width"  << boardSize.width
       << "grid_height" << boardSize.height
       << "square_size" << squareSize
       << "}";
}

void Chessboard::read(const cv::FileNode & node)
{
    node["grid_width"]  >> boardSize.width;
    node["grid_height"] >> boardSize.height;
    node["square_size"] >> squareSize;
}

std::vector<cv::Point3f> Chessboard::gridPoints() const
{
    std::vector<cv::Point3f> rPNn_all;
    rPNn_all.reserve(boardSize.height*boardSize.width);
    for (int i = 0; i < boardSize.height; ++i)
        for (int j = 0; j < boardSize.width; ++j)
            rPNn_all.push_back(cv::Point3f(j*squareSize, i*squareSize, 0));   
    return rPNn_all; 
}

std::ostream & operator<<(std::ostream & os, const Chessboard & chessboard)
{
    return os << "boardSize: " << chessboard.boardSize << ", squareSize: " << chessboard.squareSize;
}

ChessboardImage::ChessboardImage(const cv::Mat & image_, const Chessboard & chessboard, const std::filesystem::path & filename_)
    : image(image_)
    , filename(filename_)
    , isFound(false)
{
    //  - Detect chessboard corners in image and set the corners member
    //  - (optional) Do subpixel refinement of detected corners
    // cv::findChessboardCorners(image, chessboard.boardSize, corners); 

    cv::Mat gray;
    if (image.channels() == 3 || image.channels() == 4) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    }
    else {
        gray = image;
    }

    bool found = cv::findChessboardCorners(
        gray, // If input image is not grayscale, it is converted internally
        chessboard.boardSize, // cv::Size(cols, rows)
        corners,
        cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE
    );

    if (found)
    {
        // Optional: refine corner positions to subpixel accuracy
        cv::cornerSubPix(
            gray,
            corners,
            cv::Size(11, 11),
            cv::Size(-1, -1),
            cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.001)
        );
        isFound = true;
    }
    else
    {
        isFound = false;
    }

}

void ChessboardImage::drawCorners(const Chessboard & chessboard)
{
    cv::drawChessboardCorners(image, chessboard.boardSize, corners, isFound);
}

void ChessboardImage::drawBox(const Chessboard & chessboard, const Camera & camera)
{
    float squareSize = chessboard.squareSize;
    cv::Size boardSize = chessboard.boardSize;

    const float boxHeight = -0.23f; // meters (given)

    // Generate 8 vertices of the box in world coordinates
    std::vector<cv::Point3f> vertices;

    // Base 
    vertices.push_back({0, 0, 0});                                                                      // top-left
    vertices.push_back({(boardSize.width -1)*squareSize, 0, 0});                                        // top-right
    vertices.push_back({(boardSize.width -1)*squareSize, (boardSize.height -1)*squareSize, 0});         // bottom-right
    vertices.push_back({0, (boardSize.height -1)*squareSize, 0});                                       // bottom-left
    // Top
    vertices.push_back({0, 0, boxHeight});                                                              // top-left
    vertices.push_back({(boardSize.width -1)*squareSize, 0, boxHeight});                                // top-right
    vertices.push_back({(boardSize.width -1)*squareSize, (boardSize.height -1)*squareSize, boxHeight}); // bottom-right
    vertices.push_back({0, (boardSize.height -1)*squareSize, boxHeight});                               // bottom-left

    // Define edges as pairs of indices
    std::vector<std::pair<cv::Point3f, cv::Point3f>> edges = {
        // Base
        {vertices[0], vertices[1]}, // top edge
        {vertices[1], vertices[2]}, // right edge
        {vertices[2], vertices[3]}, // bottom edge
        {vertices[3], vertices[0]}, // left edge
        // Top
        {vertices[4], vertices[5]}, // top edge
        {vertices[5], vertices[6]}, // right edge
        {vertices[6], vertices[7]}, // bottom edge
        {vertices[7], vertices[4]}, // left edge
        // Vertical edges
        {vertices[0], vertices[4]}, // left upper
        {vertices[1], vertices[5]}, // right upper
        {vertices[2], vertices[6]}, // right lower
        {vertices[3], vertices[7]}  // left lower
    };

    // Draw each edge
    int segmentsPerEdge = 50; // Number of segments per edge for smoothness
    const int thickness = 2;
    // Edge colors
    std::vector<cv::Scalar> edgeColours = {
        cv::Scalar(255, 0, 0),
        cv::Scalar(255, 0, 0),
        cv::Scalar(255, 0, 0),
        cv::Scalar(255, 0, 0),
        cv::Scalar(0, 255, 0),
        cv::Scalar(0, 255, 0),
        cv::Scalar(0, 255, 0),
        cv::Scalar(0, 255, 0),
        cv::Scalar(0, 0, 255),
        cv::Scalar(0, 0, 255),
        cv::Scalar(0, 0, 255),
        cv::Scalar(0, 0, 255)
    };

    for (size_t e = 0; e < edges.size(); ++e) 
    {
        const auto& edge = edges[e];
        const cv::Scalar& colour = edgeColours[e];
        for (int i = 0; i < segmentsPerEdge; ++i) 
        {
            float alpha0 = float(i) / segmentsPerEdge;
            float alpha1 = float(i + 1) / segmentsPerEdge;
            cv::Point3f p0 = edge.first * (1 - alpha0) + edge.second * alpha0;
            cv::Point3f p1 = edge.first * (1 - alpha1) + edge.second * alpha1;
            cv::Vec3d p0d(p0.x, p0.y, p0.z);
            cv::Vec3d p1d(p1.x, p1.y, p1.z);

            // Check if within FOV
            if (camera.isWorldWithinFOV(p0d, Tnc) && camera.isWorldWithinFOV(p1d, Tnc)) 
            {
                cv::Vec2d q0 = camera.worldToPixel(p0d, Tnc);
                cv::Vec2d q1 = camera.worldToPixel(p1d, Tnc);

                cv::line(image, cv::Point(cvRound(q0[0]), cvRound(q0[1])),
                        cv::Point(cvRound(q1[0]), cvRound(q1[1])),
                        colour, thickness);
            }
        }
    }
}

void ChessboardImage::recoverPose(const Chessboard & chessboard, const Camera & camera)
{
    std::vector<cv::Point3f> rPNn_all = chessboard.gridPoints();

    cv::Mat Thetacn, rNCc;
    cv::solvePnP(rPNn_all, corners, camera.cameraMatrix, camera.distCoeffs, Thetacn, rNCc);

    Pose<double> Tcn(Thetacn, rNCc);
    Tnc = Tcn.inverse();
}

ChessboardData::ChessboardData(const std::filesystem::path & configPath)
{
    // Ensure the config file exists
    if (!std::filesystem::exists(configPath))
    {
        throw std::runtime_error("Config file does not exist: " + configPath.string());
    }

    // Open the config file
    cv::FileStorage fs(configPath.string(), cv::FileStorage::READ);
    if (!fs.isOpened())
    {
        throw std::runtime_error("Failed to open config file: " + configPath.string());
    }

    // Read chessboard configuration
    cv::FileNode node = fs["chessboard_data"];
    node["chessboard"] >> chessboard;
    std::println("Chessboard: {}", to_string(chessboard));

    // Read file pattern for chessboard images
    std::string pattern;
    node["file_regex"] >> pattern;
    fs.release();

    // Create regex object from pattern
    std::regex re(pattern, std::regex_constants::basic | std::regex_constants::icase);
    
    // Get the directory containing the config file
    std::filesystem::path root = configPath.parent_path();
    std::println("Scanning directory {} for file pattern \"{}\"", root.string(), pattern);

    // Populate chessboard images from regex
    chessboardImages.clear();
    if (std::filesystem::exists(root) && std::filesystem::is_directory(root))
    {
        // Iterate through all files in the directory and its subdirectories
        for (const auto & p : std::filesystem::recursive_directory_iterator(root))
        {
            if (std::filesystem::is_regular_file(p))
            {
                // Check if the file matches the regex pattern
                if (std::regex_match(p.path().filename().string(), re))
                {
                    std::print("Loading {}...", p.path().filename().string());

                    // Try to load the file as an image
                    cv::Mat image = cv::imread(p.path().string(), cv::IMREAD_COLOR);

                    bool isImage = !image.empty();
                    if (isImage)
                    {
                        // If it's an image, detect chessboard
                        std::print(" done, detecting chessboard...");
                        ChessboardImage ci(image, chessboard, p.path().filename());
                        std::println("{}", ci.isFound ? " found" : " not found");
                        if (ci.isFound)
                        {
                            chessboardImages.push_back(ci);
                        }
                    }
                    else
                    {
                        // If it's not an image, try to load it as a video
                        cv::VideoCapture cap(p.path().string());
                        bool isVideo = cap.isOpened();
                        if (isVideo)
                        {
                            // Get number of video frames
                            int nFrames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
                            std::println(" done, found {} frames", nFrames);

                            // Loop through selected frames
                            for (int idxFrame = 0; idxFrame < nFrames; idxFrame += 50/*Use every 50th frame*/)
                            {
                                // Read frame
                                std::print("Reading {} frame {}...", p.path().filename().string(), idxFrame);
                                cv::Mat frame;
                                cap.set(cv::CAP_PROP_POS_FRAMES, idxFrame);
                                cap.read(frame);

                                if (frame.empty())
                                {
                                    std::println(" end of file found");
                                    break;
                                }

                                // Detect chessboard in frame
                                std::print(" done, detecting chessboard...");
                                std::string baseName = p.path().stem().string();
                                std::string frameFilename = std::format("{}_{:05d}.jpg", baseName, idxFrame);
                                ChessboardImage ci(frame, chessboard, frameFilename);
                                std::println("{}", ci.isFound ? " found" : " not found");
                                if (ci.isFound)
                                {
                                    chessboardImages.push_back(ci);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void ChessboardData::drawCorners()
{
    for (auto & chessboardImage : chessboardImages)
    {
        chessboardImage.drawCorners(chessboard);
    }
}

void ChessboardData::drawBoxes(const Camera & camera)
{
    for (auto & chessboardImage : chessboardImages)
    {
        chessboardImage.drawBox(chessboard, camera);
    }
}

void ChessboardData::recoverPoses(const Camera & camera)
{
    for (auto & chessboardImage : chessboardImages)
    {
        chessboardImage.recoverPose(chessboard, camera);
    }
}

void Camera::calibrate(ChessboardData & chessboardData)
{
    std::vector<cv::Point3f> rPNn_all = chessboardData.chessboard.gridPoints();

    std::vector<std::vector<cv::Point2f>> rQOi_all;
    for (const auto & chessboardImage : chessboardData.chessboardImages)
    {
        rQOi_all.push_back(chessboardImage.corners);
    }
    assert(!rQOi_all.empty());

    imageSize = chessboardData.chessboardImages[0].image.size();
    
    flags = cv::CALIB_RATIONAL_MODEL | cv::CALIB_THIN_PRISM_MODEL;

    std::vector<std::vector<cv::Point3f>> objectPoints(chessboardData.chessboardImages.size(), rPNn_all);

    // Find intrinsic and extrinsic camera parameters
    cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
    distCoeffs = cv::Mat::zeros(12, 1, CV_64F);
    // std::vector<cv::Mat> Thetacn_all, rNCc_all;
    std::vector<cv::Mat> rvecs, tvecs;
    std::print("Calibrating camera...");
    double rms = cv::calibrateCamera(
        objectPoints,   // vector<vector<Point3f>>
        rQOi_all,       // vector<vector<Point2f>>
        imageSize,      // cv::Size
        cameraMatrix,   // CV_64F
        distCoeffs,     // CV_64F (size matches flags)
        rvecs,          // output rvecs (size = nimages)
        tvecs,          // output tvecs (size = nimages)
        flags
    );
    std::println(" done");
    
    // Pre-compute constants used in isVectorWithinFOV
    calcFieldOfView();

    // Write extrinsic camera parameters for each chessboard image
    assert(chessboardData.chessboardImages.size() == tvecs.size());
    assert(chessboardData.chessboardImages.size() == rvecs.size());
    for (std::size_t k = 0; k < chessboardData.chessboardImages.size(); ++k)
    {
        // Set the camera orientation and position (extrinsic camera parameters)
        Pose<double> & Tnc = chessboardData.chessboardImages[k].Tnc;

        // Convert rotation vector (Rodrigues) -> rotation matrix (cv::Mat)
        cv::Mat Rcn_mat;
        cv::Rodrigues(rvecs[k], Rcn_mat);

        // Convert cv::Mat -> Eigen::Matrix3d
        Eigen::Matrix3d Rcn_eig;
        cv::cv2eigen(Rcn_mat, Rcn_eig);

        // Rnc is transpose of Rcn
        Eigen::Matrix3d Rnc_eig = Rcn_eig.transpose();

        // Convert translation vector tvecs[k] (camera->chessboard in camera coords) to Eigen
        Eigen::Vector3d rNCc_eig;
        rNCc_eig << tvecs[k].at<double>(0), tvecs[k].at<double>(1), tvecs[k].at<double>(2);

        // rCNn = -Rnc * rNCc  (camera position in chessboard/world frame depending on conventions)
        Eigen::Vector3d rCNn_eig = -Rnc_eig * rNCc_eig;

        // Store into Pose<double> which uses Eigen types
        Tnc.rotationMatrix    = Rnc_eig;
        Tnc.translationVector = rCNn_eig;
    }

    printCalibration();
    std::println("{:>30} {}", "RMS reprojection error:", rms);

    assert(cv::checkRange(cameraMatrix));
    assert(cv::checkRange(distCoeffs));
}

void Camera::printCalibration() const
{
    std::bitset<8*sizeof(flags)> bitflag(flags);
    std::println("\nCalibration data:");
    std::println("{:>30} {}", "Bit flags:", bitflag.to_string());
    std::println("{:>30}\n{}", "cameraMatrix:", to_string(cameraMatrix));
    std::println("{:>30}\n{}", "distCoeffs:", to_string(distCoeffs.t()));
    std::println("{:>30} (fx, fy) = ({}, {})", "Focal lengths:",
              cameraMatrix.at<double>(0, 0), cameraMatrix.at<double>(1, 1));       
    std::println("{:>30} (cx, cy) = ({}, {})", "Principal point:",
              cameraMatrix.at<double>(0, 2), cameraMatrix.at<double>(1, 2));     
    std::println("{:>30} {} deg", "Field of view (horizontal):", 180.0/CV_PI*hFOV);
    std::println("{:>30} {} deg", "Field of view (vertical):", 180.0/CV_PI*vFOV);
    std::println("{:>30} {} deg", "Field of view (diagonal):", 180.0/CV_PI*dFOV);
}

void Camera::calcFieldOfView()
{
    assert(cameraMatrix.rows == 3);
    assert(cameraMatrix.cols == 3);
    assert(cameraMatrix.type() == CV_64F);

    float imageHeight = static_cast<float>(imageSize.height);
    float imageWidth = static_cast<float>(imageSize.width);

    // Horizontal FOV
    cv::Vec3d leftVec  = pixelToVector(cv::Vec2d(0, imageHeight / 2.0));
    cv::Vec3d rightVec = pixelToVector(cv::Vec2d(imageWidth - 1, imageHeight / 2.0));
    hFOV = std::acos(leftVec.dot(rightVec));

    // Vertical FOV
    cv::Vec3d topVec    = pixelToVector(cv::Vec2d(imageWidth / 2.0, 0));
    cv::Vec3d bottomVec = pixelToVector(cv::Vec2d(imageWidth / 2.0, imageHeight - 1));
    vFOV = std::acos(topVec.dot(bottomVec));

    // Diagonal FOV
    cv::Vec3d topLeft     = pixelToVector(cv::Vec2d(0, 0));
    cv::Vec3d bottomRight = pixelToVector(cv::Vec2d(imageWidth - 1, imageHeight - 1));
    dFOV = std::acos(topLeft.dot(bottomRight));
}


cv::Vec3d Camera::worldToVector(const cv::Vec3d & rPNn, const Pose<double> & Tnb) const
{
    // Camera pose Tnc (i.e., Rnc, rCNn)
    Pose<double> Tnc = bodyToCamera(Tnb); // Tnb*Tbc

    // Compute the unit vector uPCc from the world position rPNn and camera pose Tnc

    // Convert inputs to Eigen for math (Pose uses Eigen types)
    Eigen::Vector3d rPNn_eig(rPNn[0], rPNn[1], rPNn[2]);
    const Eigen::Matrix3d & Rnc_eig = Tnc.rotationMatrix;      // Rnc
    const Eigen::Vector3d & rCNn_eig = Tnc.translationVector;  // rCNn
    
    // Compute point in camera frame: rPCc = Rnc^T * (rPNn - rCNn)
    Eigen::Vector3d rPCc_eig = Rnc_eig.transpose() * (rPNn_eig - rCNn_eig);

    double n = rPCc_eig.norm();
    if (n <= std::numeric_limits<double>::epsilon()) {
        throw std::runtime_error("worldToVector: zero-length vector encountered");
    }

    Eigen::Vector3d uPCc_eig = rPCc_eig / n;

    // Convert back to cv::Vec3d
    return cv::Vec3d(uPCc_eig[0], uPCc_eig[1], uPCc_eig[2]);
}

cv::Vec2d Camera::worldToPixel(const cv::Vec3d & rPNn, const Pose<double> & Tnb) const
{
    return vectorToPixel(worldToVector(rPNn, Tnb));
}

cv::Vec2d Camera::vectorToPixel(const cv::Vec3d & rPCc) const
{
    cv::Vec2d rQOi;
    std::vector<cv::Point2d> imagePoints;
    cv::projectPoints(
        std::vector<cv::Vec3d>{rPCc}, 
        cv::Vec3d::zeros(), // no rotation
        cv::Vec3d::zeros(), // no translation
        cameraMatrix, 
        distCoeffs, 
        imagePoints
    );
    // Check if projection was successful
    if (imagePoints.empty()) {
        throw std::runtime_error("vectorToPixel: projection returned no points");
    }

    // Assign the pixel coordinates
    rQOi[0] = imagePoints[0].x;
    rQOi[1] = imagePoints[0].y;

    return rQOi;
}

Eigen::Vector2d Camera::vectorToPixel(const Eigen::Vector3d & rPCc, Eigen::Matrix23d & J) const
{
    Eigen::Vector2d rQOi;
    double fx_ = cameraMatrix.at<double>(0, 0);
    double fy_ = cameraMatrix.at<double>(1, 1);
    double cx_ = cameraMatrix.at<double>(0, 2);
    double cy_ = cameraMatrix.at<double>(1, 2);

    // Extract camera location
    double x = rPCc[0];
    double y = rPCc[1];
    double z = rPCc[2];

    // Compute u, v, & r
    double u = x / z;
    double v = y / z; 
    double r2 = u*u + v*v;
    double r4 = r2*r2;
    double r6 = r2*r2*r2;
    double r = std::sqrt(r2);

    // Get distortion coefficients (order set by CV flags when calibrating)
    double k1 = distCoeffs.at<double>(0);
    double k2 = distCoeffs.at<double>(1);
    double p1 = distCoeffs.at<double>(2);
    double p2 = distCoeffs.at<double>(3);
    double k3 = distCoeffs.at<double>(4);
    double k4 = distCoeffs.at<double>(5);
    double k5 = distCoeffs.at<double>(6);
    double k6 = distCoeffs.at<double>(7);
    double s1 = distCoeffs.at<double>(8);
    double s2 = distCoeffs.at<double>(9);
    double s3 = distCoeffs.at<double>(10);
    double s4 = distCoeffs.at<double>(11);

    // Compute alpha and beta
    double alpha = k1*r2 + k2*r4 + k3*r6;
    double beta = k4*r2 + k5*r4 + k6*r6;

    // Compute radial distortion factor c
    double c = (1 + alpha) / (1 + beta);

    // Compute distorted coordinates
    double u_prime = c * u + 2*p1*u*v + p2*(r2+2*u*u) + s1*r2 + s2*r4;
    double v_prime = c * v + p1*(r2+2*v*v) + 2*p2*u*v + s3*r2 + s4*r4;

    // Convert to pixel coordinates
    rQOi[0] = fx_ * u_prime + cx_;
    rQOi[1] = fy_ * v_prime + cy_;

    // Compute jacobian analytically 
    Eigen::MatrixXd dudrPCc, dvdrPCc;
    dudrPCc.resize(1, 3);
    dudrPCc << 1.0/z, 0.0, -x/(z*z);
    dvdrPCc.resize(1, 3);
    dvdrPCc << 0.0, 1.0/z, -y/(z*z);

    double drdu = (r > 0) ? u/r : 0.0;
    double drdv = (r > 0) ? v/r : 0.0;

    double dalphadr = 2*k1*r + 4*k2*r*r2 + 6*k3*r*r4;
    double dbetadr = 2*k4*r + 4*k5*r*r2 + 6*k6*r*r4;

    double dcdr = (dalphadr*(1+beta) - (1+alpha)*dbetadr) / ((1+beta)*(1+beta)); //

    double duprime_du = dcdr*drdu*u + c + 2*p1*v + p2*(2*drdu*r + 4*u) + 2*s1*drdu*r + 4*s2*drdu*r*r2;
    double duprime_dv = dcdr*drdv*u + 2*p1*u + p2*(2*r*drdv) + 2*s1*r*drdv + 4*s2*drdv*r*r2;

    double dvprime_du = dcdr*drdu*v + 2*p2*v + p1*(2*r*drdu) + 2*s3*r*drdu + 4*s4*drdu*r*r2;
    double dvprime_dv = dcdr*drdv*v + c + 2*p2*u + p1*(2*r*drdv+4*v) + 2*s3*r*drdv + 4*s4*drdv*r*r2;

    J.row(0) = fx_ * (duprime_du*dudrPCc + duprime_dv*dvdrPCc);
    J.row(1) = fy_ * (dvprime_du*dudrPCc + dvprime_dv*dvdrPCc);

    return rQOi;
}

cv::Vec3d Camera::pixelToVector(const cv::Vec2d & rQOi) const
{
    // Compute unit vector (uPCc) for the given pixel location (rQOi)
    
    std::vector<cv::Point2f> undistorted;
    // Convert to float vector for OpenCV
    std::vector<cv::Point2f> pxLocation = {cv::Point2f(rQOi[0], rQOi[1])};
    // pxLocation.push_back(cv::Point2f(static_cast<float>(rQOi[0]), static_cast<float>(rQOi[1])));

    cv::undistortPoints(
        pxLocation,
        undistorted,
        cameraMatrix,
        distCoeffs
    );

    if (undistorted.empty())
    {
        throw std::runtime_error("Undistortion failed: no points returned.");
    }

    // Convert undistorted 2D point to 3D unit vector
    cv::Vec3d uPCc(undistorted[0].x, undistorted[0].y, 1.0);
    return uPCc / cv::norm(uPCc);  // normalize
}

bool Camera::isVectorWithinFOV(const cv::Vec3d & rPCc) const
{
    // Check if rPCc lies in the image
    // Pyramid method:
    // 1. Reject vectors behind the camera (negative z)
    // 2. Compute dot product of each face normal vector (computed during calibration) with input vector
    // 3. If all dot products are positive, the vector is within the FOV

    if (rPCc[2] <= 0) {
        return false; // Behind camera
    }
    for (const auto& normal : faceNormals) {
        if (normal.dot(rPCc) < 0) {
            return false; // Outside FOV
        }
    }

    return true;
}

bool Camera::isWorldWithinFOV(const cv::Vec3d & rPNn, const Pose<double> & Tnb) const
{
    return isVectorWithinFOV(worldToVector(rPNn, Tnb));
}

Eigen::Matrix<double, 2, Eigen::Dynamic> Camera::undistort(const Eigen::Matrix<double, 2, Eigen::Dynamic> & rQOi) const
{
    // Convert from Eigen matrix to std::vector of cv::Point2d
    std::vector<cv::Point2d> rQOi_cv(rQOi.cols());
    for (int i = 0; i < rQOi.cols(); ++i)
    {
        rQOi_cv[i] = cv::Point2d(rQOi(0, i), rQOi(1, i));
    }

    // Undistort points
    std::vector<cv::Point2d> rQbarOi_cv;

    cv::undistortPoints(rQOi_cv, rQbarOi_cv, cameraMatrix, distCoeffs, cv::noArray(), cameraMatrix);

    // Convert from std::vector of cv::Point2d to Eigen matrix
    Eigen::Matrix<double, 2, Eigen::Dynamic> rQbarOi(2, rQOi.cols());
    for (int i = 0; i < rQbarOi_cv.size(); ++i)
    {
        rQbarOi(0, i) = rQbarOi_cv[i].x;
        rQbarOi(1, i) = rQbarOi_cv[i].y;
    }

    return rQbarOi;
}

Eigen::Matrix<double, 3, Eigen::Dynamic> Camera::undistort(const Eigen::Matrix<double, 3, Eigen::Dynamic> & pQOi) const
{
    // Extract the Euclidean points from the homogeneous coordinates
    Eigen::Matrix<double, 2, Eigen::Dynamic> rQOi = pQOi.topRows<2>().array().rowwise() / pQOi.row(2).array();

    // Call the Euclidean undistort function
    Eigen::Matrix<double, 2, Eigen::Dynamic> rQbarOi = undistort(rQOi);

    // Create the output matrix with homogeneous coordinates
    Eigen::Matrix<double, 3, Eigen::Dynamic> pQbarOi(3, pQOi.cols());
    pQbarOi.topRows<2>() = rQbarOi;
    pQbarOi.row(2).setOnes();

    return pQbarOi;
}

Eigen::Matrix<double, 2, Eigen::Dynamic> Camera::distort(const Eigen::Matrix<double, 2, Eigen::Dynamic> & rQbarOi) const
{
    double fx = cameraMatrix.at<double>( 0,  0);
    double fy = cameraMatrix.at<double>( 1,  1);
    double cx = cameraMatrix.at<double>( 0,  2);
    double cy = cameraMatrix.at<double>( 1,  2);

    // Convert from Euclidean coordinates to homogeneous coordinates
    Eigen::Matrix<double, 3, Eigen::Dynamic> pQbarOi(3, rQbarOi.cols());
    pQbarOi.topRows<2>() = rQbarOi;
    pQbarOi.row(2).setOnes();

    // Solve K*rPCc = pQbarOi for rPCc
    Eigen::Matrix<double, 3, Eigen::Dynamic> rPCc;

    Eigen::Matrix3d K;
    K << fx, 0, cx,
         0, fy, cy,
         0, 0, 1;

    rPCc.resize(3, pQbarOi.cols());

    // Eigen::Matrix<double, 3, Eigen::Dynamic> rPCc(3, pQbarOi.cols());
    for (int i = 0; i < pQbarOi.cols(); ++i)
    {
        rPCc.col(i) = K.inverse() * pQbarOi.col(i);
    }

    // Use camera model (with lens distortion) to get pixel coordinates
    Eigen::Matrix<double, 2, Eigen::Dynamic> rQOi(2, rQbarOi.cols());

    for (int i = 0; i < rPCc.cols(); ++i)
    {
        rQOi.col(i) = vectorToPixel(Eigen::Vector3d(rPCc.col(i)));
    }
    return rQOi;
}

Eigen::Matrix<double, 3, Eigen::Dynamic> Camera::distort(const Eigen::Matrix<double, 3, Eigen::Dynamic> & pQbarOi) const
{
    // Convert from homogeneous coordinates to Euclidean coordinates
    Eigen::Matrix<double, 2, Eigen::Dynamic> rQbarOi = pQbarOi.topRows<2>().array().rowwise() / pQbarOi.row(2).array();

    // Call the Euclidean distort function
    Eigen::Matrix<double, 2, Eigen::Dynamic> rQOi = distort(rQbarOi);

    // Convert from Euclidean coordinates to homogeneous coordinates
    Eigen::Matrix<double, 3, Eigen::Dynamic> pQOi(3, pQbarOi.cols());
    pQOi.topRows<2>() = rQOi;
    pQOi.row(2).setOnes();

    return pQOi;
}

void Camera::computePyramid() 
{
    // Get image params
    const double height = static_cast<double>(imageSize.height);
    const double width = static_cast<double>(imageSize.width);
    const double cx = cameraMatrix.at<double>(0, 2); // x coordinate of principal point (pixel coords)
    const double cy = cameraMatrix.at<double>(1, 2); // y coordinate of principal point (pixel coords)

    // Margin in pixels
    const double pxMargin = std::max(0.0, (1.5 - 1.0)) * 50.0;

    // Sample just outside the image (half pixel)
    double LowerBound = height - 0.5 + pxMargin;
    double UpperBound = - 0.5 + pxMargin;
    double RightBound = width - 0.5 + pxMargin;
    double LeftBound = - 0.5 - pxMargin;

    // Define optical axis from principal point
    cv::Vec3d opticalAxis = pixelToVector(cv::Vec2d(cx, cy));

    // Define corner rays
    cv::Vec3d topLeft = pixelToVector(cv::Vec2d(LeftBound, UpperBound));
    cv::Vec3d topRight = pixelToVector(cv::Vec2d(RightBound, UpperBound));
    cv::Vec3d bottomLeft = pixelToVector(cv::Vec2d(LeftBound, LowerBound));
    cv::Vec3d bottomRight = pixelToVector(cv::Vec2d(RightBound, LowerBound));

    // Create helper function
    auto face = [&](const cv::Vec3d & a, const cv::Vec3d & b) {
        // Computes the normal vector of a plane defined by a and b and orients it in the direction of the camera's optical axis
        cv::Vec3d normal = a.cross(b);
        // Check if normal is inside pyramid (dot product with optical axis should be positive)
        if (normal.dot(opticalAxis) < 0) {
            normal = -normal; // Flip normal to point in the direction of the optical axis
        }

        double norm = cv::norm(normal); // Magnitude of vector

        // Ensure don't divide by zero
        if (norm == 0) {
            // Define normal as zero vector
            return cv::Vec3d(0, 0, 0);
        }
        return normal / norm; // Return unit normal vector
    };

    // Compute face normals and store
    faceNormals.resize(4); // 4 faces
    faceNormals[0] = face(topLeft, topRight);       // Top face
    faceNormals[1] = face(topRight, bottomRight);   // Right face
    faceNormals[2] = face(bottomRight, bottomLeft); // Bottom face
    faceNormals[3] = face(bottomLeft, topLeft);     // Left face
}

void Camera::write(cv::FileStorage & fs) const
{
    fs << "{"
       << "camera_matrix"           << cameraMatrix
       << "distortion_coefficients" << distCoeffs
       << "flags"                   << flags
       << "imageSize"               << imageSize
       << "}";
}

void Camera::read(const cv::FileNode & node)
{
    node["camera_matrix"]           >> cameraMatrix;
    node["distortion_coefficients"] >> distCoeffs;
    node["flags"]                   >> flags;
    node["imageSize"]               >> imageSize;

    // Pre-compute constants used in isVectorWithinFOV
    calcFieldOfView();

    assert(cameraMatrix.cols == 3);
    assert(cameraMatrix.rows == 3);
    assert(cameraMatrix.type() == CV_64F);
    assert(distCoeffs.cols == 1);
    assert(distCoeffs.type() == CV_64F);
}

