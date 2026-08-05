#include <cstdlib>
#include <cassert>
#include <string>
#include <filesystem>
#include <print>
#include <opencv2/core.hpp>
#include "calibrate.h"
#include "CameraLens.h"
#include "FieldMap.h"
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
        "{verbosity v     | 1        | log verbosity (0:silent, 1:one line per event, 2:+optimiser summary and per-hypothesis detail, 3:+per-iteration optimiser trace)}"
        "{lens l          |          | camera calibration to replay with (default: from the frame size)}"
        "{field f         |          | field the recording was made on (default: follows the camera)}"
        "{export e        |          | export results}";

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("MCHA4400 RoboCup field localisation");

    if (parser.has("help"))
    {
        parser.printMessage();
        // The keys string above is a compile-time literal, so the calibrations
        // the build actually knows about are listed here instead.
        std::println("Available --lens calibrations: {}", lensNames());
        std::println("Available --field sizes: {}", fieldNames());
        for (const CameraLens & lens : lensCatalogue())
        {
            std::println("  {}", lens.describe());
        }
        return EXIT_SUCCESS;
    }

    int interactive = parser.get<int>("interactive");
    int verbosity = parser.get<int>("verbosity");
    std::string lensName = parser.get<std::string>("lens");
    std::string fieldName = parser.get<std::string>("field");
    bool hasExport = parser.has("export");
    bool hasCalibrate = parser.has("calibrate");
    std::filesystem::path inputPath = parser.get<std::string>("@input");

    if (!parser.check())
    {
        parser.printMessage();
        parser.printErrors();
        return EXIT_FAILURE;
    }

    if (!lensName.empty() && lensByName(lensName) == nullptr)
    {
        std::println("Unknown --lens '{}'. Available calibrations: {}", lensName, lensNames());
        return EXIT_FAILURE;
    }

    if (!fieldName.empty() && !fieldByName(fieldName))
    {
        std::println("Unknown --field '{}'. Available fields: {}", fieldName, fieldNames());
        return EXIT_FAILURE;
    }

    if (verbosity < 0 || verbosity > 3)
    {
        std::println("--verbosity must be in 0..3, got {}", verbosity);
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
        runFieldLocalisation(inputPath, interactive, outputDirectory, lensName, fieldName, verbosity);
    }

    return EXIT_SUCCESS;
}

