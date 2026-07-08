#include <filesystem>
#include <string>
#include <print>
#include <iostream>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "BufferedVideo.h"
#include "visualNavigation.h"
#include "Camera.h"
#include "DJIVideoCaption.h"
#include "GaussianInfo.hpp"
#include "SystemVisualNav.h"
#include "plot_visual_odometry.h"
#include "PlotOutdoor.h"
#include "funcmin.hpp"
#include "MeasurementAltimeter.h"
#include "MeasurementOutdoorFlowBundle.h"

// Forward declarations
static Eigen::Vector6d getInitialPose(const DJIVideoCaption & caption0);
double latitudeToMeters(double latitude);
double longitudeToMeters(double longitude);

double rEarth = 6371000.0; // Mean radius of the Earth in meters
double lat0;
double lon0;

void runVisualNavigationFromVideo(const std::filesystem::path & videoPath, const std::filesystem::path & cameraPath, int scenario, int interactive, const std::filesystem::path & outputDirectory)
{
    assert(!videoPath.empty());

    int imgModulus  = 3;                    // Take frames divisible by this number
    int divisor     = 2;                    // Image scaling factor (used for plotting only)

    // Subtitle path for Scenario 4
    std::vector<DJIVideoCaption> djiVideoCaption;
    std::filesystem::path subtitlePath;
    if (scenario == 4)
    {
        subtitlePath = videoPath.parent_path() / (videoPath.stem().string() + ".SRT");
        assert(std::filesystem::exists(subtitlePath));
        std::println("Subtitle file: {}", subtitlePath.string());
        
        // Load and parse subtitle file
        djiVideoCaption = getVideoCaptions(subtitlePath);
    }

    // Output video path
    std::filesystem::path outputPath;
    bool doExport = !outputDirectory.empty();
    if (doExport)
    {
        std::string outputFilename = videoPath.stem().string()
                                   + "_out"
                                   + videoPath.extension().string();
        outputPath = outputDirectory / outputFilename;
    }

    // Load camera calibration
    Camera camera;
    assert(std::filesystem::exists(cameraPath));
    cv::FileStorage fs(cameraPath.string(), cv::FileStorage::READ);
    assert(fs.isOpened());
    fs["camera"] >> camera;

    // Display loaded calibration data
    camera.printCalibration();

    // Compute FOV constants
    camera.computePyramid();

    // Set camera pose w.r.t. body
    Eigen::Matrix3d Rbc;
    Rbc <<  0, 0, 1,     // b1 = c3
            1, 0, 0,   // b2 = c1
            0, 1, 0;   // b3 = c2
    camera.Tbc.rotationMatrix = Rbc;
    camera.Tbc.translationVector = Eigen::Vector3d::Zero();

    // Open input video
    cv::VideoCapture cap(videoPath.string());
    assert(cap.isOpened());
    int nFrames = cap.get(cv::CAP_PROP_FRAME_COUNT);
    assert(nFrames > 0);
    double fps = cap.get(cv::CAP_PROP_FPS);

    BufferedVideoReader bufferedVideoReader(5);
    bufferedVideoReader.start(cap);

    cv::VideoWriter videoOut;
    BufferedVideoWriter bufferedVideoWriter(3);
    if (doExport)
    {
        cv::Size frameSize;
        if (scenario == 4)
        {
            // Single pane for outdoor scenario
            frameSize.width  = cap.get(cv::CAP_PROP_FRAME_WIDTH)/divisor;
            frameSize.height = cap.get(cv::CAP_PROP_FRAME_HEIGHT)/divisor;
        }
        else
        {
            // 2x1 pane for indoor scenario
            frameSize.width     = 2*cap.get(cv::CAP_PROP_FRAME_WIDTH)/divisor;
            frameSize.height    = cap.get(cv::CAP_PROP_FRAME_HEIGHT)/divisor;
        }
        
        double outputFps    = fps/imgModulus;
        int codec = cv::VideoWriter::fourcc('m', 'p', '4', 'v'); // manually specify output video codec
        videoOut.open(outputPath.string(), codec, outputFps, frameSize);
        bufferedVideoWriter.start(videoOut);
    }

    // Visual navigation
    cv::Mat imgk_raw; // Raw images do not have scaling appliedc
    cv::Mat imgkm1_raw;
    Eigen::Matrix<double, 2, Eigen::Dynamic> rQOikm1;

    // Visual navigation initialization
    Eigen::MatrixXd S0 = Eigen::MatrixXd::Zero(18, 18);
    S0.diagonal().setConstant(1e-3);
    auto p0 = GaussianInfo<double>::fromSqrtInfo(Eigen::VectorXd::Zero(18), S0);
    SystemVisualNav system(p0);

    Eigen::VectorXd etakm1(6);
    Eigen::VectorXd etak(6);

    double lastUsedAltitude;
    const double altitudeOffset = 7.0;  // Offset from GPS altitude to AGL
    if (scenario == 4)
    {
        // Initial pose
        etak = getInitialPose(djiVideoCaption[0]);

        // Initialize system state
        Eigen::VectorXd initialState(18);
        initialState.setZero();
        initialState.segment<6>(0) = Eigen::VectorXd::Zero(6);    // Initial velocity
        initialState.segment<6>(6) = etak;                        // Current pose η(t)
        initialState.segment<6>(12) = etak;                       // Previous pose ζ(t) = η(t-Δt) - Note: this will replaced with the exact same vector thanks to k>0 loop
                                                                  // i.e., kindof redundant here but keeps the structure clear

        // Update system with initial state
        Eigen::MatrixXd S0 = Eigen::MatrixXd::Identity(18, 18);
        // Uncertain on velocity and previous pose, certain on initial pose
        S0.diagonal().head(6).setConstant(1e-1);        // High uncertainty on initial velocity
        S0.diagonal().segment<6>(6).setConstant(1e-6);  // Low uncertainty on initial pose
        S0.diagonal().tail(6).setConstant(1e-1);   // High uncertainty on previous pose
        system.density = GaussianInfo<double>::fromSqrtMoment(initialState, S0);

        // Track last used altitude measurement
        lastUsedAltitude = djiVideoCaption[0].altitude;
    }

    // Initialisation
    int frameCount = 0;
    
    // Create plotter
    PlotOutdoor plot;
    cv::Mat imgout;

    // Data collection for trajectory plot
    std::vector<double> gps_north, gps_east;    // GPS measurements (converted to meters)
    std::vector<double> slam_north, slam_east;  // SLAM estimates
    std::vector<double> timeStamps;

    int k = 0; 

    while (true)
    {
        // Get next input frame
        cv::Mat imgk_raw = bufferedVideoReader.read();
        if (imgk_raw.empty())
        {
            break;
        }

        if (frameCount % imgModulus == 0)
        {
            if (k > 0)
            {
                // Process frame
                if (scenario == 4)
                {
                    // Outdoor visual navigation
                    // Get DJI caption for current frame
                    int captionIdx = std::min(frameCount, (int)djiVideoCaption.size() - 1);  
                    const DJIVideoCaption & caption = djiVideoCaption[captionIdx];

                    // Create altimeter measurement with current altitude
                    double currentAltitude = caption.altitude;

                    // Create measurement data from image pair and previous tracked features
                    MeasurementOutdoorFlowBundle measurement(frameCount/fps, camera, system, imgk_raw, imgkm1_raw, rQOikm1, etakm1); // Pass etakm1 here to start update for etak from here

                    // Update tracked features
                    rQOikm1 = measurement.trackedPreviousFeatures();
                    const Eigen::Matrix<double, 2, Eigen::Dynamic> & rQOik = measurement.trackedCurrentFeatures();                    

                    measurement.process(system);
                    
                    // Use AGL measurements everywhere except plotting
                    // measAlt.process is called within altimeter constructor
                    MeasurementAltimeter measAlt(frameCount/fps, camera, system, currentAltitude-altitudeOffset, lastUsedAltitude-altitudeOffset, etakm1);
                    
                    if (std::abs(currentAltitude - lastUsedAltitude) > 0.01) 
                    {
                        // If changed update last used altitude
                        lastUsedAltitude = currentAltitude;
                    }
    
                    // Extract new etak
                    Eigen::VectorXd systemState = system.density.mean();
                    Eigen::Vector6d nuk = systemState.segment<6>(0);
                    etak = systemState.segment<6>(6);
                    etakm1 = systemState.segment<6>(12);                    

                    // Update state for plotting
                    Eigen::VectorXd x(18);
                    x.setZero();
                    x.segment<6>(6) = etak;
                    x.segment<6>(12) = etakm1;

                    // Create visualization
                    cv::resize(imgk_raw, imgout, cv::Size(), 1.0/divisor, 1.0/divisor);

                    // Ensure colour image
                    if (imgout.channels() == 1)
                    {
                        cv::cvtColor(imgout, imgout, cv::COLOR_GRAY2BGR);
                    }

                    // Predicted flow field (used for plotting onto original image)
                    Eigen::Matrix<double, 2, Eigen::Dynamic> rQOik_hat = measurement.predictedFeatures(x, system);

                    // Plotting
                    std::vector<cv::Point2d> rQOikm1_scaled, rQOik_scaled, rQOik_hat_scaled;
                    int np = rQOik.cols();
                    rQOikm1_scaled.resize(np);
                    rQOik_scaled.resize(np);
                    rQOik_hat_scaled.resize(np);
                    for (int j = 0; j < np; ++j)
                    {
                        rQOikm1_scaled[j].x     = rQOikm1(0, j)/divisor;
                        rQOikm1_scaled[j].y     = rQOikm1(1, j)/divisor;

                        rQOik_scaled[j].x       = rQOik(0, j)/divisor;
                        rQOik_scaled[j].y       = rQOik(1, j)/divisor;

                        rQOik_hat_scaled[j].x   = rQOik_hat(0, j)/divisor;
                        rQOik_hat_scaled[j].y   = rQOik_hat(1, j)/divisor;
                    }

                    // Plot flow vectors
                    for (int j = 0; j < rQOik.cols(); ++j)
                    {
                        // Predicted flow
                        cv::arrowedLine(imgout, rQOikm1_scaled[j], rQOik_hat_scaled[j], cv::Scalar(255, 0, 0), 1, cv::LINE_AA);

                        if (measurement.inlierMask()[j])
                        {
                            // Measured inliers
                            cv::arrowedLine(imgout, rQOikm1_scaled[j], rQOik_scaled[j], cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
                        }
                        else
                        {
                            // Measured outliers
                            cv::arrowedLine(imgout, rQOikm1_scaled[j], rQOik_scaled[j], cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
                        }
                    }

                    plot.plotGroundPlane(imgout, etak, camera, divisor);
                    plot.plotHorizon(imgout, etak, camera, divisor);
                    plot.plotCompass(imgout, etak, camera, divisor);
                    plot.plotEpipole(imgout, nuk, etak, etakm1, camera, divisor);
                    plot.setAltitude(lastUsedAltitude, -etak(2)+altitudeOffset); // Use GPS altitude for plotting
                    plot.setNorthEast(etak(0), etak(1));
                    plot.plotText(imgout);

                    // Collect data for trajectory plot
                    plot.addTrajectoryPoint(frameCount/fps, 
                                            etak(0), etak(1),
                                            latitudeToMeters(caption.latitude),
                                            longitudeToMeters(caption.longitude));
                    
                    // Update previous features
                    rQOikm1.resize(2, rQOik.cols());
                    rQOikm1 = rQOik;
                }
                else if (scenario == 5)
                {
                    // TODO: Scenario 5 - Indoor with identical tags
                }
                else if (scenario == 6)
                {
                    // TODO: Scenario 6 - Indoor with point landmarks
                }
                // Update state

                // Update plot
                // Display
                cv::imshow("Visual Navigation", imgout);
                
                char key = -1;
                if (interactive == 2)
                {
                    // Interactive mode 2: Wait on every frame until key press
                    key = cv::waitKey(0); // Block indefinitely
                    if (key == 'q')
                    {
                        std::println("Key 'q' pressed. Terminating program.");
                        break;
                    }
                }
                else if (interactive == 1)
                {
                    // Interactive mode 1: Run without stopping, pause only on last frame
                    if (frameCount >= nFrames - imgModulus) // Check if this is the last frame
                    {
                        std::println("Last frame reached. Press 'q' to quit or any other key to exit.");
                        key = cv::waitKey(0); // Block on last frame
                        if (key == 'q')
                        {
                            std::println("Key 'q' pressed. Terminating program.");
                        }
                        break; // Exit after last frame regardless of key
                    }
                    else
                    {
                        key = cv::waitKey(1); // Non-blocking check
                        if (key == 'q')
                        {
                            std::println("Key 'q' pressed. Terminating program.");
                            break;
                        }
                    }
                }
                else // interactive == 0
                {
                    // Interactive mode 0: Run without stopping, quit on 'q' or when finished
                    key = cv::waitKey(1); // Non-blocking check
                    if (key == 'q')
                    {
                        std::println("Key 'q' pressed. Terminating program.");
                        break;
                    }
                }

                // Write output frame
                if (doExport)
                {
                    // Sanity check before write
                    if (imgout.empty() || imgout.channels() != 3) {
                        std::println("Warning: Invalid frame at {} - skipping write", frameCount);
                    } else {
                        bufferedVideoWriter.write(imgout);
                    }
                    // bufferedVideoWriter.write(imgout);
                }
            }
            // Update previous frame and features
            imgk_raw.copyTo(imgkm1_raw);
            etakm1 = etak;
            k++;  
        }  
        frameCount++;
    }

    if (doExport)
    {
         bufferedVideoWriter.stop();

         // Export trajectory plot
         if (scenario == 4) 
         {
            std::filesystem::path trajectoryOutputPath = outputDirectory / (videoPath.stem().string() + "_trajectory.png");
            plot.exportTrajectoryPlot(trajectoryOutputPath);
         }
    }
    bufferedVideoReader.stop();
}


Eigen::Vector6d getInitialPose(const DJIVideoCaption & caption0)
{
    double h        = caption0.altitude;    // Altitude (GPS) [m]
    double ga       = h - 7;                // Altitude (AGL) [m]

    Eigen::Vector6d eta0;
    eta0 << 0, 0, -ga,
            -0.01, 0.035, -M_PI/3;

    // Also set initial lat and lon readings for trajectory plot
    lat0 = caption0.latitude;
    lon0 = caption0.longitude;

    return eta0;
}

double latitudeToMeters(double latitude)
{
    double dlat = (latitude - lat0) * M_PI / 180.0;
    double north = dlat * rEarth;
    return north;
}

double longitudeToMeters(double longitude)
{
    double dlon = (longitude - lon0) * M_PI / 180.0;
    double east = dlon * rEarth * std::cos(lat0 * M_PI / 180.0);
    return east;

}
