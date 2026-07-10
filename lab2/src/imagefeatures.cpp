#include <string>  
#include <print>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/aruco.hpp>
#include "imagefeatures.h"
#include <opencv2/objdetect/aruco_dictionary.hpp>

cv::Mat detectAndDrawHarris(const cv::Mat & img, int maxNumFeatures)
{
    cv::Mat imgout = img.clone();
    // Convert to grayscale 
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY); // Convert img to grayscale and store in gray

    cv::Mat harrisDst; // Initialise output image for Harris corner detection
    cv::cornerHarris(gray, harrisDst, 2, 3, 0.04); // Perform Harris corner detection

    // Collect all candidate points above a threshold
    struct HarrisPoint 
    {
        cv::Point pt; 
        float response;
    };
    std::vector<HarrisPoint> candidates;
    float threshold = 4e-4; // Threshold for corner detection TODO
    for (int y = 0; y < harrisDst.rows; ++y)
    {
        for (int x = 0; x < harrisDst.cols; ++x)
        {
            if (harrisDst.at<float>(y, x) > threshold)
            {
                candidates.push_back({cv::Point(x, y), harrisDst.at<float>(y, x)});
            }
        }
    }

    // Print number of candidates above threshold
    std::print("Number of candidates above threshold: {}\n", candidates.size());

    // Sort by response (descending order)
    std::sort(candidates.begin(), candidates.end(), [](const HarrisPoint & a, const HarrisPoint & b) 
    {
        return a.response > b.response;
    }); 

    // Print sorted list of texture values and indices
    std::print("Top {} Harris features: \n", maxNumFeatures);
    for (int i = 0; i <std::min(maxNumFeatures, (int)candidates.size()); ++i)
    {
        std::print("Index: ({}, {}), Response: {:.6f}\n", candidates[i].pt.x, candidates[i].pt.y, candidates[i].response);
    }

    

    // Draw all features above threshold
    for (const auto & candidate : candidates)
    {
        if (candidate.response > threshold)
        {
            cv::circle(imgout, candidate.pt, 3, cv::Scalar(0, 0, 255), -1);
        }
    }

    // Draw the top N features with indices
    for (int i = 0; i < std::min(maxNumFeatures, (int)candidates.size()); ++i)
    {
        cv::circle(imgout, candidates[i].pt, 5, cv::Scalar(0, 255, 0), -1);
        cv::putText(imgout, "id=" + std::to_string(i), candidates[i].pt + cv::Point(5, -5), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 1);
    };
    return imgout;
}

cv::Mat detectAndDrawShiAndTomasi(const cv::Mat & img, int maxNumFeatures)
{
    // Define structure for points
    struct ShiTomasiPoint 
    {
        cv::Point pt; 
        float eigenVal;
    };

    cv::Mat imgout = img.clone();

    // TODO
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY); // Convert img to grayscale and store in gray

    cv::Mat eigenVal; // Initialise output image for Shi-Tomasi corner detection
    cv::cornerMinEigenVal(gray, eigenVal, 3, 3); // Perform Shi-Tomasi corner detection

    // Collect all candidate points above a threshold
    float threshold = 1e-2; 
    std::vector<ShiTomasiPoint> candidates;
    for (int y = 0; y < eigenVal.rows; ++y)
    {
        for (int x = 0; x < eigenVal.cols; ++x)
        {
            if (eigenVal.at<float>(y, x) > threshold)
            {
                candidates.push_back({cv::Point(x, y), eigenVal.at<float>(y, x)});
            }
        }
    }

    // Print number of candidates above threshold
    std::print("Number of candidates above threshold: {}\n", candidates.size());

    // Sort by response (descending order)
    std::sort(candidates.begin(), candidates.end(), [](const ShiTomasiPoint & a, const ShiTomasiPoint & b) 
    {
        return a.eigenVal > b.eigenVal;
    }); 

    // Print sorted list of texture values and indices
    std::print("Top {} Shi-Tomasi features: \n", maxNumFeatures);
    for (int i = 0; i <std::min(maxNumFeatures, (int)candidates.size()); ++i)
    {
        std::print("Index: ({}, {}), Eigenvalue: {:.6f}\n", candidates[i].pt.x, candidates[i].pt.y, candidates[i].eigenVal);
    }

    

    // Draw all features above threshold
    for (const auto & candidate : candidates)
    {
        if (candidate.eigenVal > threshold)
        {
            cv::circle(imgout, candidate.pt, 3, cv::Scalar(0, 0, 255), -1);
        }
    }

    // Draw the top N features with indices
    for (int i = 0; i < std::min(maxNumFeatures, (int)candidates.size()); ++i)
    {
        cv::circle(imgout, candidates[i].pt, 5, cv::Scalar(0, 255, 0), -1);
        cv::putText(imgout, "id=" + std::to_string(i), candidates[i].pt + cv::Point(5, -5), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 1);
    }


    return imgout;
}

cv::Mat detectAndDrawFAST(const cv::Mat & img, int maxNumFeatures)
{
    cv::Mat imgout = img.clone();

    // TODO
    // Convert to grayscale
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

    // Detect FAST keypoints
    std::vector<cv::KeyPoint> keypoints;
    int threshold = 80;
    bool nonmaxSuppression = true;
    cv::FAST(gray, keypoints, threshold, nonmaxSuppression);

    // Store keypoints and their scores in a vector
    struct FastPoint {
        cv::Point2f pt;
        float score;
    };
    std::vector<FastPoint> candidates;
    for (const auto& kp : keypoints) {
        candidates.push_back({kp.pt, kp.response});
    }

    // Sort by score (descending)
    std::sort(candidates.begin(), candidates.end(), [](const FastPoint& a, const FastPoint& b) {
        return a.score > b.score;
    });

    // Print number of candidates above threshold
    std::print("Number of candidates above threshold: {}\n", candidates.size());

    // Sort by response (descending order)
    std::sort(candidates.begin(), candidates.end(), [](const FastPoint & a, const FastPoint & b) 
    {
        return a.score > b.score;
    }); 

    // Print sorted list of texture values and indices
    std::print("Top {} FAST features: \n", maxNumFeatures);
    for (int i = 0; i <std::min(maxNumFeatures, (int)candidates.size()); ++i)
    {
        std::print("Index: ({}, {}), Score: {:.2f}\n", candidates[i].pt.x, candidates[i].pt.y, candidates[i].score);
    }

    // Draw all features above threshold
    for (const auto & candidate : candidates)
    {
        if (candidate.score > threshold)
        {
            cv::circle(imgout, candidate.pt, 3, cv::Scalar(0, 0, 255), -1);
        }
    }

    // Draw the top N features with indices
    for (int i = 0; i < std::min(maxNumFeatures, (int)candidates.size()); ++i) {
        cv::circle(imgout, candidates[i].pt, 5, cv::Scalar(0, 255, 0), -1);
        cv::putText(imgout, "id=" + std::to_string(i), candidates[i].pt + cv::Point2f(5, -5), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 1);
    }

    return imgout;
}

cv::Mat detectAndDrawArUco(const cv::Mat & img, int maxNumFeatures)
{
    cv::Mat imgout = img.clone();

    // TODO
    cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);

    cv::aruco::ArucoDetector ArucoDetector(dictionary);

    // Output containers
    std::vector<std::vector<cv::Point2f>> corners;
    std::vector<int> ids;
    std::vector<std::vector<cv::Point2f>> rejected;

    // Detect markers
    ArucoDetector.detectMarkers(img, corners, ids, rejected);

    // Draw detected markers and their IDs
    if (!ids.empty()) {
        cv::aruco::drawDetectedMarkers(imgout, corners, ids);
    }

    // Print detected marker IDs
    std::vector<int> sorted_ids = ids;
    std::sort(sorted_ids.begin(), sorted_ids.end());

    std::print("Detected {} ArUco markers: ", sorted_ids.size());
    for (int id : sorted_ids) {
        std::print("{} ", id);
    }
    std::print("\n");

    // Print a sorted list of marker corner locations, sorted by marker id
    std::vector<std::pair<int, std::vector<cv::Point2f>>> id_corners;
    for (size_t i = 0; i < ids.size(); ++i) {
        id_corners.push_back({ids[i], corners[i]});
    }
    std::sort(id_corners.begin(), id_corners.end(), [](const auto& a, const auto& b) {
        return a.first < b.first;
    });

    std::print("Sorted marker corners by ID:\n");
    for (const auto& [id, pts] : id_corners) {
        std::print("ID {}: ", id);
        for (const auto& pt : pts) {
            std::print("({:.1f}, {:.1f}) ", pt.x, pt.y);
        }
        std::print("\n");
    }

    return imgout;
}
