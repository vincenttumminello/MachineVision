#include <cstdlib>
#include <cassert>
#include <string>
#include <filesystem>
#include <print>
#include <opencv2/core.hpp>
#include "calibrate.h"
#include "fieldLocalisation.h"

int main(int argc, char* argv [])
{
    const cv::String keys =
        // Argument names | defaults | help message
        "{help h usage ?  |          | print this help message}"
        "{@input          | <none>   | path to recorded data directory or calibration XML}"
        "{calibrate c     |          | perform camera calibration for given configuration XML}"
        "{robocup r       |          | run RoboCup field localisation on recorded data directory}"
        "{interactive i   | 0        | interactivity (0:none, 1:last frame, 2:all frames)}"
        "{export e        |          | export results}";

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("MCHA4400 RoboCup field localisation");

    if (parser.has("help"))
    {
        parser.printMessage();
        return EXIT_SUCCESS;
    }

    int interactive = parser.get<int>("interactive");
    bool hasExport = parser.has("export");
    bool hasCalibrate = parser.has("calibrate");
    std::filesystem::path inputPath = parser.get<std::string>("@input");

    if (!parser.check())
    {
        parser.printMessage();
        parser.printErrors();
        return EXIT_FAILURE;
    }

    std::filesystem::path outputDirectory;
    if (hasExport)
    {
        std::filesystem::path appPath = parser.getPathToApplication();
        outputDirectory = appPath / ".." / "out";

        // Create output directory if we need to
        if (!std::filesystem::exists(outputDirectory))
        {
            std::println("Creating directory {}", outputDirectory.string());
            std::filesystem::create_directory(outputDirectory);
        }
    }

    if (hasCalibrate)
    {
        std::println("Calibrating camera");
        std::println("Configuration file: {}", inputPath.string());
        calibrateCamera(inputPath);
    }
    else
    {
        assert(0 <= interactive && interactive <= 2);
        std::println("Running RoboCup field localisation");
        std::println("Data directory: {}", inputPath.string());
        runFieldLocalisation(inputPath, interactive, outputDirectory);
    }

    return EXIT_SUCCESS;
}

